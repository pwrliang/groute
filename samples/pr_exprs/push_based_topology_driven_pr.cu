//
// Created by liang on 2/16/18.
//
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>
#include <random>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.h>
#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/atomic_utils.h>
#include <cub/grid/grid_barrier.cuh>
#include <cub/cub.cuh>
#include <gflags/gflags.h>
#include <utils/cuda_utils.h>
#include <groute/graphs/traversal_algo.h>
#include <groute/cta_work.h>
#include "pr_common.h"

DECLARE_double(threshold);
DECLARE_string(model);
DECLARE_uint32(max_iteration);
DECLARE_bool(offline);
#define OUTLINING

namespace pushbased_pr {
    template<typename WorkSource,
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum>
    __global__ void PageRankInit__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks, ResidualDatum<rank_t> residual, ResidualDatum<rank_t> last_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        for (index_t ii = 0 + tid; ii < work_source.get_size(); ii += nthreads) {
            index_t node = work_source.get_work(ii);

            current_ranks[node] = IDENTITY_ELEMENT;
            last_residual[node] = IDENTITY_ELEMENT;
            residual[node] = (1 - ALPHA);
        }
    }

    template<typename TValue>
    __inline__ __device__ TValue warpReduce(TValue localSum) {
        localSum += __shfl_xor_sync(0xfffffff, localSum, 16);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 8);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 4);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 2);
        localSum += __shfl_xor_sync(0xfffffff, localSum, 1);

        return localSum;
    }

    template<template<typename> class TRankDatum>
    __device__ void PageRankCheck__Single__(TRankDatum<rank_t> current_ranks,
                                            rank_t *block_sum_buffer, rank_t *rtn_sum) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        int laneIdx = threadIdx.x % warpSize;
        int warpIdx = threadIdx.x / warpSize;
        const int SMEMDIM = blockDim.x / warpSize;
        __shared__ rank_t smem[32];

        uint32_t work_size = current_ranks.size;
        rank_t local_sum = 0;

        for (uint32_t node = 0 + tid; node < work_size; node += nthreads) {
            rank_t dist = current_ranks[node];
            if (dist != IDENTITY_ELEMENT)
                local_sum += dist;
        }

        local_sum = warpReduce(local_sum);

        // let every first thread in warps write out the warp-level sum to shared-memory
        if (laneIdx == 0) {
            smem[warpIdx] = local_sum;
        }

        /* "__syncthreads" ensures every first thread in warps in the same block to wait for write-job done
         * if we remove this sync, the threads of first warp will read the old values when last time written
         * The side-effect is : use old values to evaluate the termination which just causes later termination
         * I think this is ok
         */
//        __syncthreads();

        // let first warp get partial sum from shared-memory
        if (warpIdx == 0) {
            local_sum = (threadIdx.x < SMEMDIM) ? smem[threadIdx.x] : 0;
            // let first warp do the reduce again
            local_sum = warpReduce(local_sum);

            // let first thread in the first warp write out the block-level sum
            if (laneIdx == 0) {
                block_sum_buffer[blockIdx.x] = local_sum;
            }
        }

        // let the "master" thread to sum up the block-level sum
        if (tid == 0) {
            uint32_t sum = 0;
            for (int bid = 0; bid < gridDim.x; bid++) {
                sum += block_sum_buffer[bid];
            }
            *rtn_sum = sum;
        }
    }

    template<
            typename WorkSource,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankAsyncKernelCTA__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks, //Pagerank values
            ResidualDatum<rank_t> residual //delta
    ) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> local_work = {0, 0, IDENTITY_ELEMENT};

            if (i < work_size) {
                index_t node = work_source.get_work(i);

                rank_t res = atomicExch(residual.get_item_ptr(node), 0);

                //filter out unnecessary computations
                if (res > 0) {
                    current_ranks[node] += res;

                    local_work.start = graph.begin_edge(node);
                    local_work.size = graph.end_edge(node) - local_work.start;
                    local_work.meta_data = ALPHA * res / local_work.size;
                }
            }

            //use other warps/blocks to help high-degree nodes
            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    local_work,
                    [&graph, &residual](index_t edge, rank_t update) {
                        index_t dest = graph.edge_dest(edge);
                        atomicAdd(residual.get_item_ptr(dest), update);
                    }
            );
        }
    }

    template<
            typename WorkSource,
            typename TGraph, template<typename> class RankDatum,
            template<typename> class ResidualDatum>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankAsyncKernel__Single__(
            WorkSource work_source,
            TGraph graph,
            RankDatum<rank_t> current_ranks, //Pagerank values
            ResidualDatum<rank_t> residual //delta
    ) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;

        uint32_t work_size = work_source.get_size();

        for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {

            index_t node = work_source.get_work(i);
            rank_t old_delta = atomicExch(residual.get_item_ptr(node), 0);

            //filter out unnecessary computations
            if (old_delta > 0) {
                current_ranks[node] += old_delta;

                index_t begin_edge = graph.begin_edge(node),
                        end_edge = graph.end_edge(node),
                        out_degree = end_edge - begin_edge;

                rank_t new_delta = ALPHA * old_delta / out_degree;

                for (index_t edge = begin_edge; edge < end_edge; edge++) {
                    index_t dest = graph.edge_dest(edge);

                    atomicAdd(residual.get_item_ptr(dest), new_delta);
                }
            }

        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankSyncKernelCTA__Single__(
            WorkSource work_source,
            TGraph graph,
            index_t iteration,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_round_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;


        for (index_t i = 0 + tid; i < work_size_rup; i += nthreads) {
            groute::dev::np_local<rank_t> local_work = {0, 0, IDENTITY_ELEMENT};

            if (i < work_size) {
                index_t node = work_source.get_work(i);

                rank_t res;

                if (iteration % 2 == 0) {
                    res = atomicExch(residual.get_item_ptr(node), 0);
                } else {
                    res = atomicExch(last_round_residual.get_item_ptr(node), 0);
                }

                if (res > 0) {
                    current_ranks[node] += res;

                    local_work.start = graph.begin_edge(node);
                    local_work.size = graph.end_edge(node) - local_work.start;
                    local_work.meta_data = ALPHA * res / local_work.size;
                }
            }
            groute::dev::CTAWorkScheduler<rank_t>::template schedule(
                    local_work,
                    [&graph, &residual, &last_round_residual, &iteration](index_t edge, rank_t update) {
                        index_t dest = graph.edge_dest(edge);

                        if (iteration % 2 == 0) {
                            atomicAdd(last_round_residual.get_item_ptr(dest), update);
                        } else {
                            atomicAdd(residual.get_item_ptr(dest), update);
                        }
                    }
            );
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankController__Single__(
            WorkSource work_source,
            TGraph graph,
            cub::GridBarrier grid_barrier,
            int max_iteration,
            bool sync,
            int *running_flag,
            rank_t threshold,
            rank_t *block_sum_buffer,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_round_residual) {

        uint32_t tid = TID_1D;
        uint32_t iteration = 0;
        rank_t pr_sum;

        while (*running_flag) {
            if (sync) {
                PageRankSyncKernelCTA__Single__(work_source,
                                                graph,
                                                iteration,
                                                current_ranks,
                                                residual,
                                                last_round_residual);

            } else {
                PageRankAsyncKernelCTA__Single__(work_source,
                                                 graph,
                                                 current_ranks,
                                                 residual);
            }
            //For Sync or Async model, we both should add a barrier to wait for computation done
            grid_barrier.Sync();//Wait for other threads
            PageRankCheck__Single__(current_ranks, block_sum_buffer, &pr_sum);

            if (tid == 0) {
                printf("%s Iter:%u Current PR sum:%f Normed PR sum:%f\n", sync ? "Sync" : "Async", iteration, pr_sum,
                       pr_sum / work_source.get_size());
                // thread 0 notify other threads to exit.
                if (iteration >= max_iteration || pr_sum / work_source.get_size() >= threshold) {
                    *running_flag = 0;
                }
            }

            __threadfence();
            iteration++;
            grid_barrier.Sync();//make sure other threads can see "running_flag"
        }
    };


    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankControllerOfflineCheck__Single__(
            WorkSource work_source,
            TGraph graph,
            cub::GridBarrier grid_barrier,
            bool sync,
            int *running_flag,
            uint32_t max_iteration,
            uint32_t *rounds_at_end,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_round_residual) {

        uint32_t tid = TID_1D;
        uint32_t iteration = 0;

        while (*running_flag) {
            if (sync) {
                PageRankSyncKernelCTA__Single__(work_source,
                                                graph,
                                                iteration,
                                                current_ranks,
                                                residual,
                                                last_round_residual);
                grid_barrier.Sync();
            } else {
//                PageRankAsyncKernelCTA__Single__(work_source,
//                                                 graph,
//                                                 current_ranks,
//                                                 residual);
                PageRankAsyncKernel__Single__(work_source, graph, current_ranks, residual);
            }
            //For Sync or Async model, we both should add a barrier to wait for computation done

            if (tid == 0) {
                printf("%s Iter:%u\n", sync ? "Sync" : "Async", iteration);
                // thread 0 notify other threads to exit.
                if (iteration >= max_iteration) {
                    *running_flag = 0;
                    printf("maximum iterated rounds reached\n");
                }
            }

            iteration++;
            if (sync) {
                grid_barrier.Sync();
            }
        }
        rounds_at_end[tid] = iteration;
    }

    template<template<typename> class ResidualDatum>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankCopyResidual(ResidualDatum<rank_t> source, ResidualDatum<rank_t> dest) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        assert(source.size == dest.size);

        for (index_t i = 0 + tid; i < source.size; i += nthreads) {
            dest[i] = source[i];
        }
    }

    template<template<typename> class ResidualDatum>
#ifdef OUTLINING
    __device__
#else
    __global__
#endif
    void PageRankCheckResidual(int *convergent,
                               ResidualDatum<rank_t> last_residual,
                               ResidualDatum<rank_t> current_residual) {
        unsigned tid = TID_1D;
        unsigned nthreads = TOTAL_THREADS_1D;

        assert(last_residual.size == current_residual.size);

        for (index_t i = 0 + tid; i < last_residual.size; i += nthreads) {
//            if (abs(last_residual[i] - current_residual[i]) > EPSILON) {
            if (current_residual[i] > EPSILON) {
                atomicAdd(convergent, 1);
            }
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankControllerPartialCheck__Single__(
            WorkSource work_source,
            TGraph graph,
            cub::GridBarrier grid_barrier,
            int *running_flag,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_round_residual) {

        uint32_t tid = TID_1D;
        uint32_t iteration = 0;

        while (*running_flag) {
            PageRankCopyResidual(residual, last_round_residual);
            grid_barrier.Sync();

            PageRankAsyncKernelCTA__Single__(work_source,
                                             graph,
                                             current_ranks, residual);
            //For Sync or Async model, we both should add a barrier to wait for computation done
            *running_flag = 0;
            grid_barrier.Sync();

            PageRankCheckResidual(running_flag, residual, last_round_residual);
            grid_barrier.Sync();

            if (tid == 0) {
                printf("Iter:%u actives:%u\n", iteration, *running_flag);
                // thread 0 notify other threads to exit.
            }
            iteration++;
        }
    };


    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankControllerProfile__Single__(
            WorkSource work_source,
            TGraph graph,
            int peak_clk,
            cub::GridBarrier grid_barrier,
            bool sync,
            float *max_time,
            float *min_time,
            int *running_flag,
            rank_t threshold,
            rank_t *block_sum_buffer,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_round_residual) {

        uint32_t tid = TID_1D;
        uint32_t iteration = 0;
        rank_t pr_sum = 0;

        while (*running_flag) {
            if (tid == 0) {
                *min_time = INT_MAX;
                *max_time = 0;
            }
            grid_barrier.Sync();
            int64_t last_clock = clock64();
            if (sync) {
                PageRankSyncKernelCTA__Single__(work_source,
                                                graph,
                                                iteration,
                                                current_ranks,
                                                residual,
                                                last_round_residual);

            } else {
                PageRankAsyncKernelCTA__Single__(work_source,
                                                 graph,
                                                 current_ranks,
                                                 residual);
            }

            float consumed = (clock64() - last_clock) / ((float) peak_clk);

            utils::atomicMin(min_time, consumed);
            utils::atomicMax(max_time, consumed);
            grid_barrier.Sync();
            if (tid == 0) {
                printf("%s Iter:%d computing time, min:%f max:%f\n", sync ? "Sync" : "Async", iteration, *min_time,
                       *max_time);
                *min_time = INT_MAX;
                *max_time = 0;
            }
            grid_barrier.Sync();

            last_clock = clock64();
            PageRankCheck__Single__(current_ranks, block_sum_buffer, &pr_sum);
            consumed = (clock64() - last_clock) / ((float) peak_clk);
            utils::atomicMin(min_time, consumed);
            utils::atomicMax(max_time, consumed);
            grid_barrier.Sync();

            if (tid == 0) {
//                printf("Iter:%u Current PR sum:%f Normed PR sum:%f consumed:%f ms.\n", iteration, pr_sum,
//                       pr_sum / work_source.get_size(), consumed);

                printf("%s Iter:%d checking time, min: %f max: %f\n", sync ? "Sync" : "Async", iteration, *min_time,
                       *max_time);
                // thread 0 notify other threads to exit.
                if (pr_sum / work_source.get_size() >= threshold) {
                    *running_flag = 0;
                }
            }

            iteration++;
            grid_barrier.Sync();
        }
    }

    template<
            typename TGraph,
            template<typename> class RankDatum,
            template<typename> class ResidualDatum,
            typename WorkSource>
    __global__ void PageRankControllerBarrierless__Single__(
            WorkSource work_source,
            TGraph graph,
            cub::GridBarrier grid_barrier,
            bool sync,
            int *running_flag,
            uint32_t *rounds_at_end,
            rank_t threshold,
            rank_t *block_sum_buffer,
            RankDatum<rank_t> current_ranks,
            ResidualDatum<rank_t> residual,
            ResidualDatum<rank_t> last_round_residual) {
        uint32_t tid = TID_1D;
        uint32_t iteration = 0;
        rank_t pr_sum = 0;

        while (*running_flag) {
            if (sync) {
                PageRankSyncKernelCTA__Single__(work_source,
                                                graph,
                                                iteration,
                                                current_ranks,
                                                residual,
                                                last_round_residual);
                grid_barrier.Sync();
            } else {
                PageRankAsyncKernel__Single__(work_source,
                                              graph,
                                              current_ranks,
                                              residual);
            }

            PageRankCheck__Single__(current_ranks, block_sum_buffer, &pr_sum);
            grid_barrier.Sync();

            if (tid == 0) {
                printf("%s Iter:%u Current PR sum:%f Normed PR sum:%f\n",
                       sync ? "Sync" : "True Barrierless",
                       iteration, pr_sum,
                       pr_sum / work_source.get_size());

                // thread 0 notify other threads to exit.
                if (pr_sum / work_source.get_size() >= threshold) {
                    *running_flag = 0;
                }
            }

            iteration++;
            if (sync) {
                //let every thread to see the lastest running_flag
                __threadfence();
                grid_barrier.Sync();
            }
            grid_barrier.Sync();
        }

        rounds_at_end[tid] = iteration;
    };

    struct Algo {
        static const char *NameLower() { return "pr"; }

        static const char *Name() { return "PR"; }


        template<
                typename TGraphAllocator, typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static const std::vector<rank_t> &Gather(
                TGraphAllocator &graph_allocator, ResidualDatum &residual, RankDatum &current_ranks,
                UnusedData &... data) {
            graph_allocator.GatherDatum(current_ranks);
            return current_ranks.GetHostData();
        }

        template<
                typename ResidualDatum, typename RankDatum, typename...UnusedData>
        static std::vector<rank_t> Host(
                groute::graphs::host::CSRGraph &graph, ResidualDatum &residual, RankDatum &current_ranks,
                UnusedData &... data) {
            return PageRankHost(graph);
        }

        static int Output(const char *file, const std::vector<rank_t> &ranks) {
            return PageRankOutput(file, ranks);
        }

        static int CheckErrors(std::vector<rank_t> &ranks, std::vector<rank_t> &regression) {
            return PageRankCheckErrors(ranks, regression);
        }
    };
}

bool PushBasedTopologyDrivenPR() {
    printf("PushBasedTopologyDrivenPR - %s model", FLAGS_model.data());
    const int device = 0;
    groute::graphs::single::NodeOutputDatum<rank_t> current_ranks;
    groute::graphs::single::NodeOutputDatum<rank_t> residual;
    groute::graphs::single::NodeOutputDatum<rank_t> last_round_residual;

    groute::graphs::traversal::Context<pushbased_pr::Algo> context(1);

    groute::graphs::single::CSRGraphAllocator dev_graph_allocator(context.host_graph);

    context.SetDevice(device);

    dev_graph_allocator.AllocateDatumObjects(current_ranks, residual, last_round_residual);

    context.SyncDevice(device); // graph allocations are on default streams, must sync device

    groute::Stream stream = context.CreateStream(device);
    int peak_clk = 1;
    GROUTE_CUDA_CHECK(cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, device));
    printf("Clock rate %d khz\n", peak_clk);

    groute::graphs::dev::CSRGraph dev_graph = dev_graph_allocator.DeviceObject();

    dim3 grid_dims, block_dims;
    KernelSizing(grid_dims, block_dims, context.host_graph.nnodes);

    pushbased_pr::PageRankInit__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> >
                                                                        (groute::dev::WorkSourceRange<index_t>(
                                                                                dev_graph.owned_start_node(),
                                                                                dev_graph.owned_nnodes()),
                                                                                dev_graph,
                                                                                current_ranks.DeviceObject(),
                                                                                residual.DeviceObject(),
                                                                                last_round_residual.DeviceObject());

    stream.Sync();

    bool sync;
    if (FLAGS_model.compare("sync") == 0) {
        sync = true;
    } else if (FLAGS_model.compare("async") == 0) {
        sync = false;
    } else {
        printf("unsupported flag\n");
        return false;
    }

    printf("max iterated rounds: %u\n", FLAGS_max_iteration);

    //for terminate check
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    utils::SharedValue<rank_t> d_out;


    Stopwatch sw(false);

#ifdef OUTLINING
    grid_dims = {FLAGS_grid_size, 1, 1};
    block_dims = {FLAGS_block_size, 1, 1};

    utils::SharedArray<rank_t> block_sum_buffer(grid_dims.x);
    utils::SharedValue<int> running_flag;
    utils::SharedValue<float> max_time, min_time;

    GROUTE_CUDA_CHECK(cudaMemset(block_sum_buffer.dev_ptr, 0, sizeof(rank_t) * block_sum_buffer.buffer_size));
    running_flag.set_val_H2D(1);
    max_time.set_val_H2D(0);
    min_time.set_val_H2D(INT_MAX);

    printf("grid_size:%d block_size:%d\n", grid_dims.x, block_dims.x);

    cub::GridBarrierLifetime gridBarrierLifetime;
    gridBarrierLifetime.Setup(grid_dims.x);

    printf("Outlining enabled\n");

    utils::SharedArray<uint32_t> rounds_at_end(grid_dims.x * block_dims.x);

    sw.start();

//    pushbased_pr::PageRankControllerProfile__Single__
//            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
//            groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
//                                                  dev_graph.owned_nnodes()),
//                    dev_graph,
//                    peak_clk,
//                    gridBarrierLifetime,
//                    FLAGS_sync,
//                    max_time.dev_ptr,
//                    min_time.dev_ptr,
//                    running_flag.dev_ptr,
//                    FLAGS_threshold,
//                    block_sum_buffer.dev_ptr,
//                    current_ranks.DeviceObject(),
//                    residual.DeviceObject(),
//                    last_round_residual.DeviceObject());
//    pushbased_pr::PageRankControllerPartialCheck__Single__
//            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
//            groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
//                                                  dev_graph.owned_nnodes()),
//                    dev_graph,
//                    gridBarrierLifetime,
//                    running_flag.dev_ptr,
//                    current_ranks.DeviceObject(),
//                    residual.DeviceObject(),
//                    last_round_residual.DeviceObject());

    pushbased_pr::PageRankController__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
            groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                  dev_graph.owned_nnodes()),
                    dev_graph,
                    gridBarrierLifetime,
                    FLAGS_max_iteration,
                    sync,
                    running_flag.dev_ptr,
                    FLAGS_threshold,
                    block_sum_buffer.dev_ptr,
                    current_ranks.DeviceObject(),
                    residual.DeviceObject(),
                    last_round_residual.DeviceObject());

//    if (FLAGS_offline) {
//        pushbased_pr::PageRankControllerOfflineCheck__Single__
//                << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
//                groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
//                                                      dev_graph.owned_nnodes()),
//                        dev_graph,
//                        gridBarrierLifetime,
//                        sync,
//                        running_flag.dev_ptr,
//                        FLAGS_max_iteration,
//                        rounds_at_end.dev_ptr,
//                        current_ranks.DeviceObject(),
//                        residual.DeviceObject(),
//                        last_round_residual.DeviceObject());
//    } else {
//        pushbased_pr::PageRankControllerBarrierless__Single__
//                << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
//                groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
//                                                      dev_graph.owned_nnodes()),
//                        dev_graph,
//                        gridBarrierLifetime,
//                        sync,
//                        running_flag.dev_ptr,
//                        rounds_at_end.dev_ptr,
//                        FLAGS_threshold,
//                        block_sum_buffer.dev_ptr,
//                        current_ranks.DeviceObject(),
//                        residual.DeviceObject(),
//                        last_round_residual.DeviceObject());
//    }
#else
    KernelSizing(grid_dims, block_dims, dev_graph.owned_nnodes());

    printf("Outlining disabled\n");
    printf("grid size %d block size %d\n", grid_dims.x, block_dims.x);

    int iteration = 0;
    bool running = true;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
                           dev_graph.owned_nnodes());

    GROUTE_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    sw.start();

    while (running) {
        if (sync) {
            pushbased_pr::PageRankSyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                    groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                          dev_graph.owned_nnodes()),
                            dev_graph,
                            iteration,
                            current_ranks.DeviceObject(),
                            residual.DeviceObject(),
                            last_round_residual.DeviceObject());
        } else {
            pushbased_pr::PageRankAsyncKernelCTA__Single__ << < grid_dims, block_dims, 0, stream.cuda_stream >> > (
                    groute::dev::WorkSourceRange<index_t>(dev_graph.owned_start_node(),
                                                          dev_graph.owned_nnodes()),
                            dev_graph,
                            current_ranks.DeviceObject(),
                            residual.DeviceObject());
        }

        stream.Sync();

        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
                               dev_graph.owned_nnodes());
        rank_t pr_sum = d_out.get_val_D2H();

        iteration++;

        printf("Iter:%u Current PR sum:%f Normed PR sum:%f\n", iteration, pr_sum,
               pr_sum / dev_graph.owned_nnodes());
        if (pr_sum / dev_graph.owned_nnodes() >= FLAGS_threshold) {
            running = false;
        }

        if (iteration >= FLAGS_max_iteration) {
            printf("maximum reached\n");
            break;
        }
    }

    GROUTE_CUDA_CHECK(cudaFree(d_temp_storage));


#endif

    stream.Sync();

    sw.stop();


//    rounds_at_end.D2H();
//    std::vector<uint32_t> rounds_vec = rounds_at_end.host_vec;
//    uint32_t max_iter = 0, min_iter = UINT32_MAX;
//    for (uint32_t ind = 0; ind < rounds_vec.size(); ind++) {
//        max_iter = max(max_iter, rounds_vec[ind]);
//        min_iter = min(min_iter, rounds_vec[ind]);
//    }
//
//    printf("max iter %u min iter %u\n", max_iter, min_iter);

//    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
//                           dev_graph.owned_nnodes());
//
//    GROUTE_CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
//
//    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, current_ranks.DeviceObject().data_ptr, d_out.dev_ptr,
//                           dev_graph.owned_nnodes());
//    rank_t pr_sum = d_out.get_val_D2H();
//
//    printf("PR sum:%f\n", pr_sum / context.host_graph.nnodes);
    printf("%s model - PR done:%f ms.\n", FLAGS_model.data(), sw.ms());
//    if (pr_sum / context.host_graph.nnodes < FLAGS_threshold) {
//        printf("needs more iterations\n");
//        return false;
//    }

    // Gather
    auto gathered_output = pushbased_pr::Algo::Gather(dev_graph_allocator, residual, current_ranks);
    if (FLAGS_output.length() != 0)
        pushbased_pr::Algo::Output(FLAGS_output.c_str(), gathered_output);

    if (FLAGS_check) {
        auto regression = pushbased_pr::Algo::Host(context.host_graph, residual, current_ranks);
        return pushbased_pr::Algo::CheckErrors(gathered_output, regression) == 0;
    }
    return true;
}
