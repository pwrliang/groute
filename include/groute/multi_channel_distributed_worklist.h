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
  virtual ~IMultiChannelDistributedWorklistPeer() {}

  virtual int GetChannelCount() const = 0;

  /// The LocalInputWorklist, exposed for customized usage
  virtual CircularWorklist<TLocal>& GetLocalInputWorklist() = 0;

  /// The RemoteOutputWorklist, exposed for customized usage
  virtual CircularWorklist<TRemote>& GetRemoteOutputWorklist(int channel) = 0;

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
__global__ void MultiSplitSendKernel(
    SplitOps split_ops, TLocal* work_ptr, uint32_t work_size,
    dev::CircularWorklist<TLocal> local_work,
    dev::CircularWorklist<TRemote> remote_works) {
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

      remote_works.append_warp(packed);
    }
  }
}

template <typename TLocal, typename TRemote, typename SplitOps>
__global__ void MultiSplitReceiveKernel(
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

      if (flags & SF_Pass)
      // pass on to another device
      {
        int pass_leader = __ffs(pass_mask) - 1;
        int thread_offset = __popc(pass_mask & ((1 << lane_id()) - 1));
        remote_work.append_warp(work, pass_leader, __popc(pass_mask),
                                thread_offset);
      }
    }
  }
}

template <typename T>
class BlockingQueue {
 private:
  std::mutex d_mutex;
  std::condition_variable d_condition;
  std::deque<T> d_queue;

 public:
  void push(T const& value) {
    {
      std::unique_lock<std::mutex> lock(this->d_mutex);
      d_queue.push_front(value);
    }
    this->d_condition.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(this->d_mutex);
    this->d_condition.wait(lock, [=] { return !this->d_queue.empty(); });
    T rc(std::move(this->d_queue.back()));
    this->d_queue.pop_back();
    return rc;
  }

  bool empty() {
    std::unique_lock<std::mutex> lock(this->d_mutex);
    return d_queue.empty();
  }
};

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
  struct Pop {
    std::shared_future<Event> future;
    size_t size;

    Pop(std::shared_future<Event> future, size_t size)
        : future(std::move(future)), size(size) {}
    Pop() : future(), size(0) {}
  };

  Context& m_context;
  IDistributedWorklist& m_distributed_worklist;

  SplitOps m_split_ops;
  DistributedWorklistFlags m_flags;
  std::vector<std::shared_ptr<Counter>> m_filter_counters;

  CircularWorklist<TLocal> m_local_input_worklist;
  Worklist<TLocal> m_temp_worklist;

  std::vector<std::shared_ptr<CircularWorklist<TRemote>>>
      m_send_remote_output_worklists,  // From local work (split-send)
      m_pass_remote_output_worklists;  // From previous device on the ring
                                       // (split-receive), passing on

  thrust::device_vector<dev::CircularWorklist<TRemote>>
      m_dev_send_remote_output_worklists, m_dev_pass_remote_output_worklists;

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
  BlockingQueue<Event> m_receive_work_events;
  bool m_send_remote_work = false;
  Event m_send_remote_work_event;
  //
  // Pass-remote: (split-receive)
  BlockingQueue<Pop> m_submitted_ops;

  std::mutex m_receive_mutex;
  std::mutex m_split_recv_mutex;

  //
  // Exit:
  volatile bool m_exit = false;

  int m_send_chunk_size;

  std::vector<Link<TRemote>> m_links_in, m_links_out;

  Stream m_send_stream;
  std::vector<typename CircularWorklist<TRemote>::Bounds> m_send_bounds;

  int m_router_count;

  double m_time_other{};

  double m_time_enqueue{};

  double m_time_recv{};
  double m_time_send{};
  uint32_t m_send_times{};
  std::vector<size_t> m_size_send, m_seg_count;

  void SplitReceive(
      Counter& counter, const groute::Segment<TRemote>& received_work,
      groute::CircularWorklist<TLocal>& local_work,
      std::vector<std::shared_ptr<groute::CircularWorklist<TRemote>>>&
          remote_works,
      thrust::device_vector<groute::dev::CircularWorklist<TRemote>>&
          dev_remote_works,
      int channel, groute::Stream& stream) {
    counter.ResetAsync(stream.cuda_stream);

    dim3 block_dims(DBS, 1, 1);
    dim3 grid_dims(round_up(received_work.GetSegmentSize(), block_dims.x), 1,
                   1);

    MultiSplitReceiveKernel<TLocal, TRemote, SplitOps>
        <<<grid_dims, block_dims, 0, stream.cuda_stream>>>(
            m_split_ops, received_work.GetSegmentPtr(),
            received_work.GetSegmentSize(), local_work.DeviceObject(),
            remote_works[channel]->DeviceObject(), counter.DeviceObject());

    local_work.SyncAppendAllocAsync(stream.cuda_stream);
    remote_works[channel]->SyncAppendAllocAsync(stream.cuda_stream);

    // Report work
    // TODO (later): Try to avoid copies to host
    m_distributed_worklist.ReportWork(
        (int) received_work.GetSegmentSize() - (int) counter.GetCount(stream),
        (int) received_work.GetSegmentSize(), "Filter", m_dev);
  }

  void SplitSend(
      const groute::Segment<TLocal>& sent_work,
      groute::CircularWorklist<TLocal>& local_work,
      std::vector<std::shared_ptr<groute::CircularWorklist<TRemote>>>&
          remote_works,
      thrust::device_vector<groute::dev::CircularWorklist<TRemote>>&
          dev_remote_works,
      int channel, groute::Stream& stream) {
    dim3 block_dims(DBS, 1, 1);
    dim3 grid_dims(round_up(sent_work.GetSegmentSize(), block_dims.x), 1, 1);

    MultiSplitSendKernel<TLocal, TRemote, SplitOps>
        <<<grid_dims, block_dims, 0, stream.cuda_stream>>>(
            m_split_ops, sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
            local_work.DeviceObject(), remote_works[channel]->DeviceObject());
    //    local_work.SyncAppendAllocAsync(stream.cuda_stream);
    //    for (auto& remote_work : remote_works) {
    remote_works[channel]->SyncAppendAllocAsync(stream.cuda_stream);
    //    }
  }

  void ReceiveLoop(int channel) {
    m_context.SetDevice(m_dev);
    Stream stream = m_context.CreateStream(
        m_dev, (m_flags & DW_HighPriorityReceive) ? SP_High : SP_Default);

    std::deque<Pop> issued_send_ops;
    CircularWorklist<TRemote>& pass_wl =
        *(m_pass_remote_output_worklists[channel]);
    auto bounds = pass_wl.GetBounds(stream);
    while (true) {
      auto& link_in = m_links_in[channel];
      auto seg = link_in.Receive().get();

      if (seg.Empty()) {
        break;
      }

      size_t pop_count = 0;

      // pop sent segment as much as possible
      while (!issued_send_ops.empty() &&
             groute::is_ready(issued_send_ops.front().future)) {
        auto pop = issued_send_ops.front();
        // since we check the front is whether ready, no blocking here
        auto e = pop.future.get();
        if (!e.Query()) {
          break;
        }
        pop_count += pop.size;
        issued_send_ops.pop_front();
      }

      size_t space = pass_wl.GetSpace(bounds);
      size_t work = seg.GetSegmentSize();

      // If we don't have enough space, loop over future to wait for
      while (pop_count + space < work) {
        auto pop = issued_send_ops.front();
        auto e = pop.future.get();
        e.Wait(stream.cuda_stream);
        pop_count += pop.size;
        issued_send_ops.pop_front();
      }
      pass_wl.PopItemsAsync(pop_count, stream.cuda_stream);

      seg.Wait(stream.cuda_stream);
      SplitReceive(*(m_filter_counters[channel]), seg, m_local_input_worklist,
                   m_pass_remote_output_worklists,
                   m_dev_pass_remote_output_worklists, channel, stream);
      //      std::cout << "pass len: " << pass_wl.GetLength(stream) <<
      //      std::endl;
      // generate an event for synchronizing purpose
      Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);

      // Notify the GetLocalWork function that we got available data in
      // m_local_input_worklist
      m_receive_work_events.push(split_ev);
      link_in.ReleaseBuffer(seg, split_ev);

      // Get the latest bounds in which the end variable is expended
      auto current = pass_wl.GetBounds(stream);
      auto exclude = current.Exclude(
          bounds);  // Exclude last bound to get the bound of new-added elements
      bounds = current;

      // Now, forward work with sender and push Pop objects
      for (auto seg_to_send : pass_wl.GetSegs(exclude)) {
        auto sent_ev = m_links_out[channel].Send(seg_to_send, split_ev);

        issued_send_ops.push_back(Pop(sent_ev, seg_to_send.GetSegmentSize()));
      }
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
    int channel = 0;

    while (m_exit) {
      if (!m_submitted_ops.empty()) {
        Pop pop = m_submitted_ops.pop();
        auto e = pop.future.get();

        e.Wait(stream.cuda_stream);
        m_send_remote_output_worklists[channel]->PopItemsAsync(
            pop.size, stream.cuda_stream);
      } else {
        std::this_thread::yield();
      }
      channel = (channel + 1) % m_router_count;
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

    mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_in, mem_size, AF_PO2);
    m_local_input_worklist = groute::CircularWorklist<TLocal>(
        (TLocal*) mem_buffer, mem_size / sizeof(TLocal));
    m_local_input_worklist.ResetAsync((cudaStream_t) 0);

    m_send_stream = m_context.CreateStream(m_dev);
    //    m_send_bounds = m_pass_queue.GetBounds(m_send_stream);

    for (int i = 0; i < m_router_count; i++) {
      {
        mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_out / m_router_count,
                                   mem_size, AF_PO2);
        auto wl = std::make_shared<groute::CircularWorklist<TRemote>>(
            (TRemote*) mem_buffer, mem_size / sizeof(TRemote), "Send Queue");
        wl->ResetAsync((cudaStream_t) 0);
        m_dev_send_remote_output_worklists.push_back(wl->DeviceObject());
        m_send_remote_output_worklists.push_back(wl);
      }
      {
        mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_pass / m_router_count,
                                   mem_size, AF_PO2);
        auto wl = std::make_shared<groute::CircularWorklist<TRemote>>(
            (TRemote*) mem_buffer, mem_size / sizeof(TRemote), "Pass Queue");
        wl->ResetAsync((cudaStream_t) 0);
        m_dev_pass_remote_output_worklists.push_back(wl->DeviceObject());
        m_pass_remote_output_worklists.push_back(wl);
        m_send_bounds.push_back(wl->GetBounds(m_send_stream));
      }
      m_filter_counters.push_back(std::make_shared<Counter>());
    }

    mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_local, mem_size);
    m_temp_worklist = groute::Worklist<TLocal>((TLocal*) mem_buffer,
                                               mem_size / sizeof(TLocal));
    m_temp_worklist.ResetAsync((cudaStream_t) 0);

    GROUTE_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t) 0));  // just in case

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
    return m_local_input_worklist;
  }
  // Can be directly accessed by outside caller. This functio is used in the
  // example of SSSP for storing vertices which don't belong to current subgraph
  CircularWorklist<TRemote>& GetRemoteOutputWorklist(int channel) override {
    return *m_send_remote_output_worklists[channel];
  }
  // This variable doesn't not interact with any other variables
  Worklist<TLocal>& GetTempWorklist() override { return m_temp_worklist; }

  // convert m_local_input_worklist to segment for consuming purpose
  std::vector<Segment<TLocal>> GetLocalWork(Stream& stream) override {
    auto segs = m_local_input_worklist.ToSegs(stream);

    while (segs.empty()) {
      m_receive_work_events.pop().Wait(stream.cuda_stream);

      if (m_exit)
        return segs;

      segs = m_local_input_worklist.ToSegs(stream);
    }

    return segs;
  }

  int m_channel = 0;

  void PerformSplitSend(Segment<TLocal>& split_work, Stream& stream) override {
    if (split_work.Empty())
      return;

    m_channel = (m_channel + 1) % m_router_count;
    // All work items in m_local_input_worklist are split
    // the content in m_send_remote_output_worklist will be sent out by SendLoop
    SplitSend(split_work, m_local_input_worklist,
              m_send_remote_output_worklists,
              m_dev_send_remote_output_worklists, m_channel, stream);
    Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);
    SignalRemoteWork(split_ev);
  }

  void SignalRemoteWork(const Event& ev) override {
    ev.Wait(m_send_stream.cuda_stream);
    auto current =
        m_send_remote_output_worklists[m_channel]->GetBounds(m_send_stream);
    auto exclude = current.Exclude(m_send_bounds[m_channel]);
    m_send_bounds[m_channel] = current;

    auto segs = m_send_remote_output_worklists[m_channel]->GetSegs(exclude);

    for (auto seg : segs) {
      auto sent_ev = m_links_out[m_channel].Send(seg, Event());

      m_submitted_ops.push(Pop(std::move(sent_ev), seg.GetSegmentSize()));
    }
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
