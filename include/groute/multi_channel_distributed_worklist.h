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

#ifndef __GROUTE_MULTI_CHANNEL_DISTRIBUTED_WORKLIST_H
#define __GROUTE_MULTI_CHANNEL_DISTRIBUTED_WORKLIST_H
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <groute/blocking_queue.h>
#include <groute/context.h>
#include <groute/distributed_worklist.h>
#include <groute/event_pool.h>
#include <groute/groute.h>
#include <groute/internal/cuda_utils.h>
#include <groute/internal/pinned_allocation.h>
#include <groute/internal/worker.h>
#include <groute/worklist.h>
#include <thrust/device_vector.h>
#include <utils/stopwatch.h>

#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

DECLARE_double(wl_alloc_factor_local);
DECLARE_double(wl_alloc_factor_in);
DECLARE_double(wl_alloc_factor_out);
DECLARE_double(wl_alloc_factor_pass);
DECLARE_int32(recv_limit);

namespace groute {

template <typename TLocal, typename TRemote>
struct IMultiChannelDistributedWorklistPeer {
  virtual ~IMultiChannelDistributedWorklistPeer() = default;

  virtual int GetChannelCount() const = 0;

  /// The LocalInputWorklist, exposed for customized usage
  virtual CircularWorklist<TLocal>& GetLocalInputWorklist() = 0;

  /// The RemoteOutputWorklist, exposed for customized usage
  virtual CircularWorklist<TRemote>& GetRemoteOutputWorklist() = 0;

  /// A temp worklist for user-code, just allocated correctly, not used
  /// internally
  virtual Worklist<TLocal>& GetTempWorklist() = 0;

  /// A blocking call for local work segments
  virtual std::vector<Segment<TLocal>> GetLocalWork(Stream& stream) = 0;

  /// Perform split-send, local work will be prepended into the
  /// LocalInputWorklist and remote work will be appended into the
  /// RemoteOutputWorklist (+signal to send thread)
  virtual void PerformSplitSend(Segment<TLocal>& split_work,
                                Stream& stream) = 0;

  /// Signal that work was pushed into the RemoteOutputWorklist
  virtual void SignalRemoteWork(const Event& ev) = 0;
};

template <typename TLocal, typename TRemote, typename SplitOps>
__global__ void SingleSplitSendKernel(
    SplitOps split_ops, TLocal* work_ptr, uint32_t work_size,
    dev::CircularWorklist<TLocal> local_work,
    dev::CircularWorklist<TRemote> remote_work) {
  int tid = TID_1D;

  if (tid < work_size) {
    TLocal work = work_ptr[tid];
    SplitFlags flags = split_ops.on_send(work);

    // no filter counter here
    if (flags & SF_Take) {
      local_work.prepend_warp(work);
    }

    if (flags & SF_Pass) {
      // pack data
      TRemote packed = split_ops.pack(work);

      remote_work.append_warp(packed);
    }
  }
}

template <typename TLocal, typename TRemote, typename SplitOps>
__global__ void SingleSplitReceiveKernel(
    SplitOps split_ops, TRemote* work_ptr, uint32_t work_size,
    dev::CircularWorklist<TLocal> local_work,
    dev::CircularWorklist<TRemote> remote_work, dev::Counter filter_counter) {
  auto tid = TID_1D;
  auto nthreads = TOTAL_THREADS_1D;

  for (auto idx = tid; idx < work_size; idx += nthreads) {
    TRemote work = work_ptr[idx];
    SplitFlags flags = split_ops.on_receive(work);

    int filter_mask = __ballot_sync(__activemask(), flags == SF_None ? 1 : 0);
    int take_mask = __ballot_sync(__activemask(), flags & SF_Take ? 1 : 0);
    int pass_mask = __ballot_sync(__activemask(), flags & SF_Pass ? 1 : 0);
    // never inline the masks into the conditional branching below
    // although it may work. The compiler should optimize this anyhow,
    // but this avoids him from unifying the __ballot's

    if (flags == SF_None) {
      int filter_leader = __ffs(filter_mask) - 1;
      if (lane_id() == filter_leader)
        filter_counter.add(__popc(filter_mask));
    } else {
      if (flags & SF_Take) {
        int take_leader = __ffs(take_mask) - 1;
        int thread_offset = __popc(take_mask & ((1 << lane_id()) - 1));
        local_work.append_warp(split_ops.unpack(work), take_leader,
                               __popc(take_mask), thread_offset);
      }
      // pass on to another device
      if (flags & SF_Pass) {
        int pass_leader = __ffs(pass_mask) - 1;
        int thread_offset = __popc(pass_mask & ((1 << lane_id()) - 1));
        remote_work.append_warp(work, pass_leader, __popc(pass_mask),
                                thread_offset);
      }
    }
  }
}

struct PendingSend {
  int channel;
  std::shared_future<Event> fut;
  size_t len;
  PendingSend(int channel, std::shared_future<Event> fut, size_t len)
      : channel(channel), fut(std::move(fut)), len(len) {}
};

template <typename TLocal, typename TRemote, typename SplitOps>
class MultiChannelDistributedWorklistPeer
    : public IMultiChannelDistributedWorklistPeer<TLocal, TRemote> {
 protected:
  int m_dev, m_ngpus;

 private:
  Context& m_context;
  IDistributedWorklist& m_distributed_worklist;

  SplitOps m_split_ops;
  DistributedWorklistFlags m_flags;
  std::vector<std::shared_ptr<Counter>> m_filter_counters;

  Worklist<TLocal> m_temp_worklist;

  CircularWorklist<TLocal> m_input_worklist;
  CircularWorklist<TRemote> m_send_worklist;
  std::vector<std::shared_ptr<CircularWorklist<TRemote>>> m_pass_worklists;

  std::vector<std::thread> m_receive_threads;
  std::thread m_send_thread;

  // Sync objects
  //
  // Receive:
  //
  // Send (wait any)
  std::mutex m_send_mutex;
  std::condition_variable m_send_cv;
  //
  //  Send-remote: (split-send)
  bool m_send_remote_work = false;
  Event m_send_remote_work_event;
  //
  // Pass-remote: (split-receive)
  BlockingQueue<Event> m_receive_work_events;
  std::vector<std::shared_ptr<BlockingQueue<Event>>> m_pass_remote_work_events;

  //
  // Exit:
  volatile bool m_exit = false;

  int m_send_chunk_size;

  std::vector<Link<TRemote>> m_links_in, m_links_out;

  int m_router_count;

  double m_time_other{};

  double m_time_enqueue{};

  double m_time_recv{};
  double m_time_send{};
  uint32_t m_send_times{};
  std::vector<size_t> m_size_send, m_seg_count;

  void SplitReceive(Counter& counter,
                    const groute::Segment<TRemote>& received_work,
                    groute::CircularWorklist<TLocal>& local_work,
                    groute::CircularWorklist<TRemote>& remote_work,
                    groute::Stream& stream) {
    counter.ResetAsync(stream.cuda_stream);

    dim3 block_dims(DBS, 1, 1);
    dim3 grid_dims((int) round_up(received_work.GetSegmentSize(), block_dims.x),
                   1, 1);

    SingleSplitReceiveKernel<TLocal, TRemote, SplitOps>
        <<<grid_dims, block_dims, 0, stream.cuda_stream>>>(
            m_split_ops, received_work.GetSegmentPtr(),
            received_work.GetSegmentSize(), local_work.DeviceObject(),
            remote_work.DeviceObject(), counter.DeviceObject());

    local_work.SyncAppendAllocAsync(stream.cuda_stream);
    remote_work.SyncAppendAllocAsync(stream.cuda_stream);

    // Report work
    // TODO (later): Try to avoid copies to host
    m_distributed_worklist.ReportWork(
        (int) received_work.GetSegmentSize() - (int) counter.GetCount(stream),
        (int) received_work.GetSegmentSize(), "Filter", m_dev);
  }

  void SplitSend(const groute::Segment<TLocal>& sent_work,
                 groute::CircularWorklist<TLocal>& local_work,
                 groute::CircularWorklist<TRemote>& remote_work,
                 groute::Stream& stream) {
    dim3 block_dims(DBS, 1, 1);
    dim3 grid_dims(round_up(sent_work.GetSegmentSize(), block_dims.x), 1, 1);

    SingleSplitSendKernel<TLocal, TRemote, SplitOps>
        <<<grid_dims, block_dims, 0, stream.cuda_stream>>>(
            m_split_ops, sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
            local_work.DeviceObject(), remote_work.DeviceObject());
    remote_work.SyncAppendAllocAsync(stream.cuda_stream);
  }

  void ReceiveLoop(int channel) {
    m_context.SetDevice(m_dev);
    Stream stream = m_context.CreateStream(
        m_dev, (m_flags & DW_HighPriorityReceive) ? SP_High : SP_Default);
    auto& counter = *(m_filter_counters[channel]);
    Stopwatch sw;
    auto& pass_wl = *(m_pass_worklists[channel]);
    auto& pass_work_events = *(m_pass_remote_work_events[channel]);

    while (true) {
      auto fut = m_links_in[channel].Receive();
      auto seg = fut.get();

      if (seg.Empty()) {
        break;
      }
      // queue a wait on stream
      seg.Wait(stream.cuda_stream);

      SplitReceive(counter, seg, m_input_worklist, pass_wl, stream);

      Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);
      m_receive_work_events.push(split_ev);

      pass_work_events.push(split_ev);
      m_send_cv.notify_one();
      m_links_in[channel].ReleaseBuffer(seg, split_ev);
    }
    stream.Sync();

    // Signal exit
    {
      std::lock_guard<std::mutex> guard(m_send_mutex);
      m_exit = true;
      m_send_cv.notify_one();
    }

    {
      m_exit = true;
      m_receive_work_events.push(Event());
    }
  }

  void SendLoop() {
    m_context.SetDevice(m_dev);
    Stream stream = m_context.CreateStream(m_dev);
    int source = 0;
    Stopwatch sw, sw1;

    while (true) {
      /*
       * 1. A worklist from SendSplit
       * 2. multiple worklists from ReceiveSplit
       */

      auto forward_works = [this, &stream]() {
        bool ready = true;

        for (int channel = 0; channel < m_router_count; channel++) {
          ready &= !m_pass_remote_work_events[channel]->empty();
        }

        if (!ready) {
          return false;
        }
        std::vector<std::vector<Segment<TRemote>>> grouped_segs(m_router_count);

        for (int channel = 0; channel < m_router_count; channel++) {
          m_pass_remote_work_events[channel]->pop().Wait(stream.cuda_stream);
          auto& worklist = m_pass_worklists[channel];

          for (auto seg : worklist->ToSegs(stream)) {
            // Split seg into equal chunks, then number it.
            grouped_segs[channel].push_back(seg);
          }
        }

        std::vector<PendingSend> send_ops;
        // Emitting segs at the same time
        for (int channel = 0; channel < m_router_count; channel++) {
          for (auto& seg : grouped_segs[channel]) {
            auto fut = m_links_out[channel].Send(seg, Event());
            send_ops.template emplace_back(channel, fut, seg.GetSegmentSize());
          }
        }

        std::vector<size_t> aggregated_size(m_router_count, 0);

        for (auto& send_op : send_ops) {
          aggregated_size[send_op.channel] += send_op.len;
          send_op.fut.get().Wait(stream.cuda_stream);
        }

        for (int channel = 0; channel < m_router_count; channel++) {
          m_pass_worklists[channel]->PopItemsAsync(aggregated_size[channel],
                                                   stream.cuda_stream);
        }
        return true;
      };

      auto send_works = [this, &stream]() {
        if (m_send_remote_work) {
          m_send_remote_work = false;
          m_send_remote_work_event.Wait(stream.cuda_stream);

          std::vector<Segment<TRemote>> segs;

          for (auto seg : m_send_worklist.ToSegs(stream)) {
            auto avg_seg_len =
                (seg.GetSegmentSize() + m_router_count - 1) / m_router_count;
            auto split_segs = seg.Split(avg_seg_len);

            segs.insert(segs.begin(), split_segs.begin(), split_segs.end());
          }

          size_t avg_seg_count =
              (segs.size() + m_router_count - 1) / m_router_count;
          std::vector<PendingSend> send_ops;

          for (int channel = 0; channel < m_router_count; channel++) {
            size_t begin = std::min(channel * avg_seg_count, segs.size());
            size_t end = std::min((channel + 1) * avg_seg_count, segs.size());

            for (size_t i = begin; i < end; i++) {
              auto& seg = segs[i];
              auto fut = m_links_out[channel].Send(seg, Event());

              send_ops.template emplace_back(channel, fut,
                                             seg.GetSegmentSize());
            }
          }

          size_t aggregated_size = 0;

          for (auto& send_op : send_ops) {
            aggregated_size += send_op.len;
            send_op.fut.get().Wait(stream.cuda_stream);
          }
          m_send_worklist.PopItemsAsync(aggregated_size, stream.cuda_stream);
          return true;
        }
        return false;
      };

      {
        std::unique_lock<std::mutex> guard(m_send_mutex);

        // This loop just find out a matched work_ev which comes from
        // upstream, and a corresponding worklist
        while (true) {
          if (m_exit) {
            break;
          }
          // we alternate source for giving each worklist a fair chance
          // This procedure will be triggered by ReceiveLoop
          if (source == 0) {
            if (!send_works()) {
              forward_works();
            }
          } else {
            if (!forward_works()) {
              send_works();
            }
          }
          // Notified by SignalRemoteWork or ReceiveLoop
          m_send_cv.wait(guard);
        }
      }

      if (m_exit)
        break;

      source = 1 - source;

      m_send_times++;
    }
  }

 public:
  MultiChannelDistributedWorklistPeer(
      Context& context,
      std::vector<std::shared_ptr<router::IRouter<TRemote>>>& routers,
      IDistributedWorklist& distributed_worklist, const SplitOps& split_ops,
      DistributedWorklistFlags flags, device_t dev, int ngpus,
      size_t max_work_size, size_t max_exch_size, size_t exch_buffs)
      : m_context(context),
        m_dev(dev),
        m_ngpus(ngpus),
        m_distributed_worklist(distributed_worklist),
        m_split_ops(split_ops),
        m_flags(flags),
        m_send_chunk_size(max_work_size),
        m_router_count(routers.size()) {
    for (auto& router : routers) {
      m_links_in.emplace_back(*router, dev, max_exch_size, exch_buffs);
      m_links_out.emplace_back(dev, *router);
    }

    void* mem_buffer;
    size_t mem_size;
    Stream stream = m_context.CreateStream(m_dev);

    mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_in, mem_size, AF_PO2);
    m_input_worklist = groute::CircularWorklist<TLocal>(
        (TLocal*) mem_buffer, mem_size / sizeof(TLocal));
    m_input_worklist.ResetAsync(stream.cuda_stream);

    mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_out, mem_size, AF_PO2);
    m_send_worklist = groute::CircularWorklist<TRemote>(
        (TRemote*) mem_buffer, mem_size / sizeof(TRemote));
    m_send_worklist.ResetAsync(stream.cuda_stream);

    for (int i = 0; i < m_router_count; i++) {
      {
        mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_pass / m_router_count,
                                   mem_size, AF_PO2);
        auto wl = std::make_shared<groute::CircularWorklist<TRemote>>(
            (TRemote*) mem_buffer, mem_size / sizeof(TRemote));
        wl->ResetAsync(stream.cuda_stream);
        m_pass_worklists.push_back(wl);
      }
      m_filter_counters.push_back(std::make_shared<Counter>());
      m_pass_remote_work_events.push_back(
          std::make_shared<BlockingQueue<Event>>());
    }

    mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_local, mem_size);
    m_temp_worklist = groute::Worklist<TLocal>((TLocal*) mem_buffer,
                                               mem_size / sizeof(TLocal));
    m_temp_worklist.ResetAsync(stream.cuda_stream);

    stream.Sync();

    for (int channel = 0; channel < m_router_count; channel++) {
      m_receive_threads.push_back(
          std::thread([this, channel]() { ReceiveLoop(channel); }));
    }
    m_send_thread = std::thread([this]() { SendLoop(); });
    m_size_send.resize(m_router_count, 0);
    m_seg_count.resize(m_router_count, 0);

    if (m_dev == 0) {
      std::cout << "Packet size: " << max_exch_size / 1024.0 / 1024.0
                << " MB num buffers: " << exch_buffs << std::endl;
    }
  }

  ~MultiChannelDistributedWorklistPeer() {
    for (auto& th : m_receive_threads) {
      th.join();
    }
    m_send_thread.join();

    std::stringstream ss;
    size_t total_size = 0;

    for (auto size : m_size_send) {
      total_size += size;
    }

    ss << "Dev " << m_dev << " SendOP enqueue times: " << m_send_times
       << " send time: " << m_time_send << " split-recv time: " << m_time_recv
       << " enqueue time: " << m_time_enqueue
       << " Total comm: " << total_size / 1024.0 / 1024.0 << " MB "
       << " Bandwidth: " << total_size / 1024.0 / 1024.0 / (m_time_send / 1024)
       << " MB/s" << std::endl;

    for (int i = 0; i < m_router_count; i++) {
      auto size_in_mb = m_size_send[i] / 1024.0 / 1024;
      ss << "    Ring " << i << " Size: " << size_in_mb << " MB"
         << " Seg count: " << m_seg_count[i]
         << " Avg size: " << size_in_mb / m_seg_count[i] << " MB" << std::endl;
    }
    std::cout << ss.str();
  }

  int GetChannelCount() const { return m_router_count; }

  CircularWorklist<TLocal>& GetLocalInputWorklist() override {
    return m_input_worklist;
  }
  // Can be directly accessed by outside caller. This functio is used in the
  // example of SSSP for storing vertices which don't belong to current subgraph
  CircularWorklist<TRemote>& GetRemoteOutputWorklist() override {
    return m_send_worklist;
  }
  // This variable doesn't not interact with any other variables
  Worklist<TLocal>& GetTempWorklist() override { return m_temp_worklist; }

  // convert m_local_input_worklist to segment for consuming purpose
  std::vector<Segment<TLocal>> GetLocalWork(Stream& stream) override {
    auto segs = m_input_worklist.ToSegs(stream);

    while (segs.empty()) {
      m_receive_work_events.pop().Wait(stream.cuda_stream);

      if (m_exit)
        return segs;

      segs = m_input_worklist.ToSegs(stream);
    }
    return segs;
  }

  void PerformSplitSend(Segment<TLocal>& split_work, Stream& stream) override {
    if (split_work.Empty())
      return;
    // All work items in m_local_input_worklist are split
    // the content in m_send_remote_output_worklist will be sent out by SendLoop
    SplitSend(split_work, m_input_worklist, m_send_worklist, stream);
    Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);
    SignalRemoteWork(split_ev);
  }

  void SignalRemoteWork(const Event& ev) override {
    // Signal
    std::lock_guard<std::mutex> guard(m_send_mutex);
    m_send_remote_work = true;
    m_send_remote_work_event = ev;
    m_send_cv.notify_one();
  }
};

template <typename TLocal, typename TRemote>
class MultiChannelDistributedWorklist : public IDistributedWorklist {
 private:
  Context& m_context;
  std::vector<std::shared_ptr<router::IRouter<TRemote>>> m_routers;
  int m_ngpus;

  std::atomic<int> m_active_peers_counter;
  std::atomic<int> m_work_counter;

  // Workitem counter
  std::atomic<unsigned int> m_reported_work;
  std::vector<unsigned int> m_ctr;

 public:
  unsigned int GetCurrentWorkCount(device_t dev) { return m_ctr[dev + 1]; }

 public:
  std::mutex log_gate;

 public:
  MultiChannelDistributedWorklist(
      Context& context,
      std::vector<std::shared_ptr<router::IRouter<TRemote>>>& routers,
      int ngpus)
      : m_context(context),
        m_routers(routers),
        m_ngpus(ngpus),
        m_work_counter(0),
        m_active_peers_counter(ngpus),
        m_reported_work(0) {
    if (false) {
      m_ctr.resize(ngpus + 1, 0);
    }
  }

  virtual ~MultiChannelDistributedWorklist() {
    if (false) {
      printf("Work performed by each GPU:\n");
      for (size_t i = 1; i < m_ctr.size(); ++i)
        printf("  GPU %llu: %lu witems\n", i, m_ctr[i]);
      int repwork = m_reported_work;
      printf("Total witems: %lu\n", repwork);
    }
  }

  template <typename SplitOps>
  std::unique_ptr<IMultiChannelDistributedWorklistPeer<TLocal, TRemote>>
  CreatePeer(device_t dev, const SplitOps& split_ops, size_t max_work_size,
             size_t max_exch_size, size_t exch_buffs,
             DistributedWorklistFlags flags =
                 (DistributedWorklistFlags) (DW_WarpAppend |
                                             DW_HighPriorityReceive)) {
    m_context.SetDevice(dev);

    return groute::make_unique<
        MultiChannelDistributedWorklistPeer<TLocal, TRemote, SplitOps>>(
        m_context, m_routers, *this, split_ops, flags, dev, m_ngpus,
        max_work_size, max_exch_size, exch_buffs);
  }

  void ReportPeerTermination() {
    if (--m_active_peers_counter == 0) {
      for (auto& router : m_routers) {
        router->Shutdown();
      }
    }
  }

  void ReportWork(int new_work, int performed_work, const char* caller,
                  device_t dev) override {
    int work = new_work - performed_work;

    if (false) {
      m_reported_work += performed_work;
      m_ctr[dev + 1] += performed_work;
    }

    if (work == 0)
      return;

    int current_work = (m_work_counter += work);

    //    {
    //      std::lock_guard<std::mutex> lock(log_gate);
    //
    //      std::cout << std::endl
    //                << '[' << std::this_thread::get_id() << ",\t" << caller <<
    //                ']'
    //                << "\tNew: " << new_work << ",\tPerformed: " <<
    //                performed_work
    //                << ",\tCurrent: " << current_work;
    //    }

    if (current_work == 0) {
      for (auto& router : m_routers) {
        router->Shutdown();
      }
    }
  }

  void ReportWork(int work) override {
    if (work == 0)
      return;

    int current_work = (m_work_counter += work);

    //    {
    //      std::lock_guard<std::mutex> lock(log_gate);
    //
    //      std::cout << std::endl
    //                << '[' << std::this_thread::get_id() << ']'
    //                << "\t\tWork: " << work << ",\t\tCurrent: " <<
    //                current_work;
    //    }

    if (current_work == 0) {
      for (auto& router : m_routers) {
        router->Shutdown();
      }
    }
  }

  bool HasWork() const override { return m_work_counter > 0; }

  bool HasActivePeers() override { return m_active_peers_counter > 0; }
};
}  // namespace groute

#endif  // __GROUTE_MULTI_CHANNEL_DISTRIBUTED_WORKLIST_H
