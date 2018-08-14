// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>

#include <gflags/gflags.h>

#include <groute/event_pool.h>
#include <groute/distributed_worklist.h>
#include <groute/worklist_stack.h>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>

#include <groute/graphs/csr_graph.h>
#include <groute/graphs/traversal_algo.h>
#include <groute/cta_work.h>
#include <utils/cuda_utils.h>

#include "bc_common.h"

DEFINE_int32(source_node, 0, "The source node for the BC traversal (clamped to [0, nnodes-1])");

const level_t INF = UINT_MAX;

#define GTID (blockIdx.x * blockDim.x + threadIdx.x)


namespace bc
{
    __global__ void BCInit(level_t *levels,
                           sigma_t *sigmas,
                           int nnodes,
                           index_t source)
    {
        int tid = GTID;
        if (tid < nnodes)
        {
            if (tid == source)
            {
                levels[tid] = 0;
                sigmas[tid] = 1;
            }
            else
            {
                levels[tid] = INF;
                sigmas[tid] = 0;
            }
        }
    }


    template<typename TGraph,
            typename TGraphDatum,
            typename TWorklist,
            typename TWLStack>
    __global__ void BFSKernelFused(TGraph graph,
                                   TGraphDatum levels_datum,
                                   sigma_t *p_node_sigmas,
                                   index_t *p_search_depth,
                                   TWorklist wl1,
                                   TWorklist wl2,
                                   TWLStack wl_stack,
                                   cub::GridBarrier grid_barrier)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size;
        TWorklist *wl_in = &wl1;
        TWorklist *wl_out = &wl2;

        while ((work_size = wl_in->len()) > 0)
        {
            for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
            {
                index_t node = wl_in->read(i);
                level_t next_level = levels_datum.get_item(node) + 1;

                wl_stack.append(node);

                for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node); edge < end_edge; ++edge)
                {
                    index_t dest = graph.edge_dest(edge);
                    level_t prev = atomicMin(levels_datum.get_item_ptr(dest), next_level);

                    if (prev == INF)
                    {
                        atomicAdd(p_node_sigmas + dest, p_node_sigmas[node]);
                        atomicMax(p_search_depth, next_level);
                    }
                    else
                    {
                        if (levels_datum[dest] == next_level)
                        {
                            atomicAdd(p_node_sigmas + dest, p_node_sigmas[node]);
                        }
                    }

                    if (next_level < prev)
                    {
                        wl_out->append(dest);
                    }
                }
            }
            grid_barrier.Sync();
            if (tid == 0)
            {
                wl_in->reset();
                wl_stack.push();
            }
            grid_barrier.Sync();
            auto *tmp = wl_in;
            wl_in = wl_out;
            wl_out = tmp;
        }
    }

    template<typename TGraph,
            typename TGraphDatum,
            typename TWorklist,
            typename TWLStack>
    __global__ void BFSKernelCTAFused(TGraph graph,
                                      TGraphDatum levels_datum,
                                      sigma_t *p_node_sigmas,
                                      index_t *p_search_depth,
                                      TWorklist wl1,
                                      TWorklist wl2,
                                      TWLStack wl_stack,
                                      cub::GridBarrier grid_barrier)
    {
        int tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size;
        TWorklist *wl_in = &wl1;
        TWorklist *wl_out = &wl2;

        while ((work_size = wl_in->len()) > 0)
        {
            uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
            {
                groute::dev::np_local<index_t> np_local = {0, 0, 0};

                if (i < work_size)
                {
                    index_t node = wl_in->read(i);


                    wl_stack.append(node);

                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;
                    np_local.meta_data = node;
                }

                groute::dev::CTAWorkScheduler<level_t>::schedule
                        (np_local, [&graph, &levels_datum, &p_node_sigmas, &p_search_depth, &wl_out](index_t edge,
                                                                                                     index_t node)
                        {
                            level_t next_level = levels_datum.get_item(node) + 1;
                            index_t dest = graph.edge_dest(edge);
                            level_t prev = atomicMin(levels_datum.get_item_ptr(dest), next_level);

                            if (prev == INF)
                            {
                                atomicAdd(p_node_sigmas + dest, p_node_sigmas[node]);
                                atomicMax(p_search_depth, next_level);
                            }
                            else
                            {
                                if (levels_datum[dest] == next_level)
                                {
                                    atomicAdd(p_node_sigmas + dest, p_node_sigmas[node]);
                                }
                            }

                            if (next_level < prev)
                            {
                                wl_out->append(dest);
                            }
                        });
            }
            grid_barrier.Sync();
            if (tid == 0)
            {
                wl_in->reset();
                wl_stack.push();
            }
            grid_barrier.Sync();
            auto *tmp = wl_in;
            wl_in = wl_out;
            wl_out = tmp;
        }
    }

    template<typename Graph,
            typename WLStack,
            typename SourcePath,
            typename Sigmas>
    __global__ void StageTwoDDFused(Graph graph,
                                    WLStack wl_stack,
                                    SourcePath node_source_path,
                                    Sigmas *p_node_sigmas,
                                    Sigmas *p_node_bc_values,
                                    uint32_t *p_search_depth,
                                    cub::GridBarrier barrier)
    {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t curr_depth = *p_search_depth;

        while (curr_depth > 0)
        {
            uint32_t begin_pos = wl_stack.begin_pos(curr_depth);
            uint32_t end_pos = wl_stack.end_pos(curr_depth);

            for (uint32_t idx = tid + begin_pos; idx < end_pos; idx += nthreads)
            {
                index_t node = wl_stack.read(idx);
                index_t src_depth = node_source_path[node];

                for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node);
                     edge < end_edge; edge++)
                {
                    index_t dest = graph.edge_dest(edge);

                    if (node_source_path[dest] == src_depth + 1)
                    {
                        float delta_to = 1.0f * p_node_sigmas[node] / p_node_sigmas[dest] * (1.0f + p_node_bc_values[dest]);

                        atomicAdd(p_node_bc_values + node, delta_to);
                    }
                }
            }
            barrier.Sync();
            curr_depth--;
        }
    }


    template<typename Graph,
            typename WLStack,
            typename SourcePath,
            typename Sigmas>
    __global__ void StageTwoDDCTAFused(Graph graph,
                                       WLStack wl_stack,
                                       SourcePath node_source_path,
                                       Sigmas *p_node_sigmas,
                                       Sigmas *p_node_bc_values,
                                       uint32_t *p_search_depth,
                                       cub::GridBarrier barrier)
    {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t curr_depth = *p_search_depth;

        while (curr_depth > 0)
        {
            uint32_t begin_pos = wl_stack.begin_pos(curr_depth);
            uint32_t end_pos = wl_stack.end_pos(curr_depth);
            uint32_t work_size = end_pos - begin_pos;
            uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
            {
                groute::dev::np_local<index_t> np_local = {0, 0};

                if (i < work_size)
                {
                    index_t node = wl_stack.read(begin_pos + i);

                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;
                    np_local.meta_data = node;
                }

                groute::dev::CTAWorkScheduler<index_t>::schedule
                        (np_local,
                         [&graph, &node_source_path, &p_node_bc_values, &p_node_sigmas](index_t edge,
                                                                                    index_t node)
                         {
                             index_t src_depth = node_source_path[node];
                             index_t dest = graph.edge_dest(edge);

                             if (node_source_path[dest] == src_depth + 1)
                             {
                                 float delta_to = 1.0f * p_node_sigmas[node] / p_node_sigmas[dest] *
                                                  (1.0f + p_node_bc_values[dest]);

                                 atomicAdd(p_node_bc_values + node, delta_to);
                             }
                             return true;
                         });
            }
            barrier.Sync();
            curr_depth--;
        }
    }

    template<typename TGraph, typename TGraphDatum>
    class Problem
    {
    private:
        TGraph m_graph;
        TGraphDatum m_levels_datum;
        sigma_t *m_p_sigmas_datum;
        centrality_t *m_p_bc_value_datum;
        uint32_t *m_search_depth;
    public:
        Problem(const TGraph &graph,
                const TGraphDatum &levels_datum,
                sigma_t *p_sigmas_datum,
                centrality_t *p_bc_value_datum,
                uint32_t *search_depth) :
                m_graph(graph), m_levels_datum(levels_datum),
                m_p_sigmas_datum(p_sigmas_datum),
                m_p_bc_value_datum(p_bc_value_datum),
                m_search_depth(search_depth)
        {
        }

        void Init(index_t source_node, groute::Worklist<index_t> &in_wl, groute::Stream &stream) const
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, m_levels_datum.size);

            BCInit << < grid_dims, block_dims, 0, stream.cuda_stream >> > (m_levels_datum.data_ptr, m_p_sigmas_datum, m_graph.nnodes, source_node);
            in_wl.AppendItemAsync(stream.cuda_stream, source_node);
        }

        template<typename TWorklist, typename TWLStack>
        void Relax(TWorklist &wl1,
                   TWorklist &wl2,
                   TWLStack &wl_stack,
                   groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            int occupancy_per_MP;
            cudaDeviceProp dev_props;
            cub::GridBarrierLifetime barrier;

            GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&dev_props, 0));

            Stopwatch sw_stage1(true);
            if (FLAGS_cta_np)
            {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                              BFSKernelCTAFused<groute::graphs::dev::CSRGraph,
                                                                      groute::graphs::dev::GraphDatum<level_t>,
                                                                      groute::dev::Worklist<index_t>,
                                                                      groute::dev::WorklistStack<index_t >>, FLAGS_block_size, 0);

                int fused_work_blocks = dev_props.multiProcessorCount * occupancy_per_MP;

                grid_dims.x = fused_work_blocks;
                block_dims.x = FLAGS_block_size;

                cub::GridBarrierLifetime barrier;
                barrier.Setup(grid_dims.x);

                Stopwatch sw_stage1(true);

                BFSKernelCTAFused << < grid_dims, block_dims, 0, stream.cuda_stream >> > (m_graph,
                        m_levels_datum,
                        m_p_sigmas_datum,
                        m_search_depth,
                        wl1.DeviceObject(),
                        wl2.DeviceObject(),
                        wl_stack.DeviceObject(),
                        barrier);
                stream.Sync();
            }
            else
            {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                              BFSKernelFused<groute::graphs::dev::CSRGraph,
                                                                      groute::graphs::dev::GraphDatum<level_t>,
                                                                      groute::dev::Worklist<index_t>,
                                                                      groute::dev::WorklistStack<index_t >>, FLAGS_block_size, 0);

                int fused_work_blocks = dev_props.multiProcessorCount * occupancy_per_MP;

                grid_dims.x = fused_work_blocks;
                block_dims.x = FLAGS_block_size;

                barrier.Setup(grid_dims.x);

                BFSKernelFused << < grid_dims, block_dims, 0, stream.cuda_stream >> > (m_graph,
                        m_levels_datum,
                        m_p_sigmas_datum,
                        m_search_depth,
                        wl1.DeviceObject(),
                        wl2.DeviceObject(),
                        wl_stack.DeviceObject(),
                        barrier);
                stream.Sync();
            }

            sw_stage1.stop();

            printf("Time stage1: %f\n", sw_stage1.ms());

            Stopwatch sw_stage2(true);

            if (FLAGS_cta_np)
            {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                              StageTwoDDCTAFused<groute::graphs::dev::CSRGraph,
                                                                      groute::dev::WorklistStack<index_t>,
                                                                      groute::graphs::dev::GraphDatum<level_t>,
                                                                      sigma_t>, FLAGS_block_size, 0);

                grid_dims.x = dev_props.multiProcessorCount * occupancy_per_MP;

                barrier.Setup(grid_dims.x);

                StageTwoDDCTAFused << < grid_dims, block_dims, 0, stream.cuda_stream >> > (m_graph,
                        wl_stack.DeviceObject(),
                        m_levels_datum,
                        m_p_sigmas_datum,
                        m_p_bc_value_datum,
                        m_search_depth,
                        barrier);
                stream.Sync();
            }
            else
            {
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                              StageTwoDDFused<groute::graphs::dev::CSRGraph,
                                                                      groute::dev::WorklistStack<index_t>,
                                                                      groute::graphs::dev::GraphDatum<level_t>,
                                                                      sigma_t>, FLAGS_block_size, 0);

                grid_dims.x = dev_props.multiProcessorCount * occupancy_per_MP;

                barrier.Setup(grid_dims.x);

                StageTwoDDFused << < grid_dims, block_dims, 0, stream.cuda_stream >> > (m_graph,
                        wl_stack.DeviceObject(),
                        m_levels_datum,
                        m_p_sigmas_datum,
                        m_p_bc_value_datum,
                        m_search_depth,
                        barrier);
                stream.Sync();
            }

            sw_stage2.stop();

            printf("Time stage2: %f\n", sw_stage2.ms());
        }


    };

    struct Algo
    {
        static const char *NameLower()
        { return "bc"; }

        static const char *Name()
        { return "BC"; }

    };
}

bool TestBCSingle()
{
    groute::graphs::single::NodeOutputDatum<level_t> levels_datum;
    groute::graphs::traversal::Context<bc::Algo> context(1);
    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    context.SetDevice(0);

    dev_graph_allocator.AllocateDatumObjects(levels_datum);

    context.SyncDevice(0); // graph allocations are on default streams, must sync device

    index_t nnodes = context.nvtxs;

    utils::SharedArray<sigma_t> dev_node_sigmas(nnodes);
    utils::SharedArray<float> dev_node_bc_values(nnodes);
    utils::SharedValue<uint32_t> dev_search_depth;

    bc::Problem<groute::graphs::dev::CSRGraph,
            groute::graphs::dev::GraphDatum<level_t>> problem(dev_graph_allocator.DeviceObject(),
                                                              levels_datum.DeviceObject(),
                                                              dev_node_sigmas.dev_ptr,
                                                              dev_node_bc_values.dev_ptr,
                                                              dev_search_depth.dev_ptr);

    size_t max_work_size = context.host_graph.nedges * FLAGS_wl_alloc_factor;
    if (FLAGS_wl_alloc_abs > 0)
        max_work_size = FLAGS_wl_alloc_abs;

    groute::Stream stream;

    groute::Worklist<index_t> wl1(max_work_size), wl2(max_work_size);
    groute::WorklistStack<index_t> wl_stack(max_work_size * 2);

    wl1.ResetAsync(stream.cuda_stream);
    wl2.ResetAsync(stream.cuda_stream);
    wl_stack.ResetAsync(stream);
    stream.Sync();

    index_t source_node = min(max(0, FLAGS_source_node), context.nvtxs - 1);

    Stopwatch sw(true);

    problem.Init(source_node, wl1, stream);
    problem.Relax(wl1, wl2, wl_stack, stream);

    stream.Sync();

    sw.stop();

    printf("\n%s: %f ms. <filter>\n\n", bc::Algo::Name(), sw.ms());

    dev_node_sigmas.D2H();
    dev_node_bc_values.D2H();

    for (int i = 0; i < dev_node_bc_values.host_vec.size(); i++)
    {
        dev_node_bc_values.host_vec[i] /= 2;
    }

//    for (int i = 0; i < 100; i++)
//    {
//        printf("node: %d %f %f\n", i, dev_node_sigmas.host_vec[i], dev_node_bc_values.host_vec[i]);
//    }
    // Gather

    if (FLAGS_output.length() != 0)
        BCOutput(FLAGS_output.c_str(), dev_node_bc_values.host_vec);

    if (FLAGS_check)
    {
        auto result_pair = BetweennessCentralityHost(context.host_graph, source_node);

        int failed_sigmas = BCCheckErrors(result_pair.second, dev_node_sigmas.host_vec);

        if (failed_sigmas)
        {
            printf("Sigams failed!\n");
        }

        int failed_bc = BCCheckErrors(result_pair.first, dev_node_bc_values.host_vec);

        if (failed_bc)
        {
            printf("BC value failed!\n");
        }
        return failed_sigmas + failed_bc == 0;
    }
    else
    {
        printf("Warning: Result not checked\n");
        return true;
    }

}
