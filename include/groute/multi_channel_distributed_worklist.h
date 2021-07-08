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
#include <vector>

DECLARE_double(wl_alloc_factor_local);
DECLARE_double(wl_alloc_factor_in);
DECLARE_double(wl_alloc_factor_out);
DECLARE_double(wl_alloc_factor_pass);

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
    SplitOps split_ops, TLocal* work_ptr, uint32_t work_size, int nrings,
    dev::CircularWorklist<TLocal> local_work,
    dev::CircularWorklist<TRemote>* remote_works) {
  int tid = TID_1D;

  if (tid < work_size) {
    TLocal work = work_ptr[tid];
    SplitFlags flags = split_ops.on_send(work);

    // no filter counter here
    if (flags & SF_Take) {
      local_work.prepend(work);
    }

    if (flags & SF_Pass) {
      // pack data
      TRemote packed = split_ops.pack(work);
      int wl_idx = packed % nrings;

      remote_works[wl_idx].append(packed);
    }
  }
}

template <typename TLocal, typename TRemote, typename SplitOps>
__global__ void MultiSplitReceiveKernel(
    SplitOps split_ops, TRemote* work_ptr, uint32_t work_size, int nrings,
    dev::CircularWorklist<TLocal> local_work,
    dev::CircularWorklist<TRemote>* remote_works, dev::Counter filter_counter) {
  int tid = TID_1D;
  if (tid < work_size) {
    TRemote work = work_ptr[tid];
    SplitFlags flags = split_ops.on_receive(work);

    if (flags == SF_None) {
      filter_counter.add(1);
    } else {
      if (flags & SF_Take) {
        local_work.append(split_ops.unpack(work));
      }

      if (flags & SF_Pass) {
        int wl_idx = work % nrings;
        // belongs to another device
        remote_works[wl_idx].append(work);
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

template <typename T>
struct PendingSend {
  std::shared_future<Event> event;
  size_t len{};
  int channel;
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

  CircularWorklist<TLocal> m_local_input_worklist;
  Worklist<TLocal> m_temp_worklist;

  std::vector<std::shared_ptr<CircularWorklist<TRemote>>>
      m_send_remote_output_worklists,  // From local work (split-send)
      m_pass_remote_output_worklists;  // From previous device on the ring
                                       // (split-receive), passing on

  thrust::device_vector<dev::CircularWorklist<TRemote>>
      m_dev_send_remote_output_worklists, m_dev_pass_remote_output_worklists;

  std::vector<std::thread> m_receive_threads;
  std::vector<std::thread> m_send_threads;

  // Sync objects
  //
  // Receive:
  //
  // Send (wait any)
  std::mutex m_send_mutex;
  std::condition_variable m_send_cv;
  //
  //  Send-remote: (split-send)
  std::vector<std::shared_ptr<BlockingQueue<Event>>> m_receive_work_events;

  std::vector<std::shared_ptr<BlockingQueue<Event>>> m_send_remote_work_events;
  //
  // Pass-remote: (split-receive)
  std::vector<std::shared_ptr<BlockingQueue<Event>>> m_pass_remote_work_events;

  //
  // Exit:
  volatile bool m_exit = false;

  int m_send_chunk_size;

  std::vector<Link<TRemote>> m_links_in, m_links_out;

  int m_router_count;

  double m_time_other{};

  double m_time_enqueue{};

  double m_time_split{};
  double m_time_send{};
  uint32_t m_send_times{};
  std::vector<size_t> m_size_send, m_seg_count;
  std::vector<double> m_split_recv_time, m_split_send_time;

  std::vector<PendingSend<TRemote>> m_pending_send;

  size_t m_max_send_size{};
  size_t m_max_recv_size{};

  void SplitReceive(
      const groute::Segment<TRemote>& received_work,
      groute::CircularWorklist<TLocal>& local_work,
      std::vector<std::shared_ptr<groute::CircularWorklist<TRemote>>>&
          remote_works,
      thrust::device_vector<groute::dev::CircularWorklist<TRemote>>&
          dev_remote_works,
      Counter& filter_counter, groute::Stream& stream) {
    filter_counter.ResetAsync(stream.cuda_stream);

    dim3 block_dims(DBS, 1, 1);
    dim3 grid_dims(round_up(received_work.GetSegmentSize(), block_dims.x), 1,
                   1);

    MultiSplitReceiveKernel<TLocal, TRemote, SplitOps>
        <<<grid_dims, block_dims, 0, stream.cuda_stream>>>(
            m_split_ops, received_work.GetSegmentPtr(),
            received_work.GetSegmentSize(), m_router_count,
            local_work.DeviceObject(),
            thrust::raw_pointer_cast(dev_remote_works.data()),
            filter_counter.DeviceObject());

    local_work.SyncAppendAllocAsync(stream.cuda_stream);
    for (auto& remote_work : remote_works) {
      remote_work->SyncAppendAllocAsync(stream.cuda_stream);
    }
    // Report work
    // TODO (later): Try to avoid copies to host
    m_distributed_worklist.ReportWork((int) received_work.GetSegmentSize() -
                                          (int) filter_counter.GetCount(stream),
                                      (int) received_work.GetSegmentSize(),
                                      "Filter", m_dev);
  }

  void SplitSend(
      const groute::Segment<TLocal>& sent_work,
      groute::CircularWorklist<TLocal>& local_work,
      std::vector<std::shared_ptr<groute::CircularWorklist<TRemote>>>&
          remote_works,
      thrust::device_vector<groute::dev::CircularWorklist<TRemote>>&
          dev_remote_works,
      groute::Stream& stream) {
    dim3 block_dims(DBS, 1, 1);
    dim3 grid_dims(round_up(sent_work.GetSegmentSize(), block_dims.x), 1, 1);

    MultiSplitSendKernel<TLocal, TRemote, SplitOps>
        <<<grid_dims, block_dims, 0, stream.cuda_stream>>>(
            m_split_ops, sent_work.GetSegmentPtr(), sent_work.GetSegmentSize(),
            m_router_count, local_work.DeviceObject(),
            thrust::raw_pointer_cast(dev_remote_works.data()));

    for (auto& remote_work : remote_works) {
      remote_work->SyncAppendAllocAsync(stream.cuda_stream);
    }
  }

  void ReceiveLoop(int channel) {
    m_context.SetDevice(m_dev);
    Stream stream = m_context.CreateStream(
        m_dev, (m_flags & DW_HighPriorityReceive) ? SP_High : SP_Default);
    Stopwatch sw;

    while (true) {
      auto fut = m_links_in[channel].Receive();
      auto seg = fut.get();

      if (seg.Empty()) {
        std::cout << "Dev: " << m_dev << " empty seg" << std::endl;
        break;
      }

      // queue a wait on stream
      seg.Wait(stream.cuda_stream);

      sw.start();
      SplitReceive(seg, m_local_input_worklist, m_pass_remote_output_worklists,
                   m_dev_pass_remote_output_worklists,
                   *(m_filter_counters[channel]), stream);
      stream.Sync();
      sw.stop();
      m_split_recv_time[channel] += sw.ms();

      std::stringstream ss;

      ss << "Dev: " << m_dev << " Channel:" << channel << " recv "
         << seg.GetSegmentSize()
         << " local size: " << m_local_input_worklist.GetLength(stream)
         << " pass size: "
         << m_pass_remote_output_worklists[channel]->GetLength(stream)
         << std::endl;

      std::cout << ss.str();

      // generate an event for synchronizing purpose
      Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);
      // Signal SendLoop that it can send m_pass_remote_output_worklist with
      // link_out
      m_pass_remote_work_events[channel]->push(split_ev);
      // Notify sender
      m_send_cv.notify_all();
      // Notify the GetLocalWork function that we got available data in
      // m_local_input_worklist
      m_receive_work_events[channel]->push(split_ev);
      // We use split_ev to let the deeper function to know that SplitReceive
      // is done
      m_links_in[channel].ReleaseBuffer(seg, split_ev);
    }
    std::stringstream ss;
    ss << "Dev: " << m_dev << " Channel:" << channel << " recv exit"
       << std::endl;
    std::cout << ss.str();
    stream.Sync();

    if (channel == 0) {
      // Signal exit
      {
        std::lock_guard<std::mutex> guard(m_send_mutex);
        m_exit = true;
        m_send_cv.notify_all();
      }

      {
        m_exit = true;
        m_receive_work_events[channel]->push(Event());
      }
    }
  }

  void SendLoop(int channel) {
    m_context.SetDevice(m_dev);
    Stream stream = m_context.CreateStream(m_dev);

    int source = 0;

    while (true) {
      std::shared_ptr<CircularWorklist<TRemote>> worklist;
      {
//        std::unique_lock<std::mutex> guard(m_send_mutex);

        // This loop just find out a matched work_ev which comes from
        // upstream, and a corresponding worklist
        while (true) {
          if (m_exit)
            break;

          // we alternate source for giving each worklist a fair chance
          // This procedure will be triggered by ReceiveLoop
          if (source == 0) {
            // we first check the pass list at this round
            if (!m_pass_remote_work_events[channel]->empty()) {
              m_pass_remote_work_events[channel]->pop().Wait(
                  stream.cuda_stream);
              worklist = m_pass_remote_output_worklists[channel];
              break;
            }
            // this procedure will be triggered by SignalRemoteWork
            if (!m_send_remote_work_events[channel]->empty()) {
              m_send_remote_work_events[channel]->pop().Wait(
                  stream.cuda_stream);
              worklist = m_send_remote_output_worklists[channel];
              break;
            }
          } else {
            // we first check the send list at this round
            // Same, SignalRemoteWork modifies m_send_remote_work
            if (!m_send_remote_work_events[channel]->empty()) {
              m_send_remote_work_events[channel]->pop().Wait(
                  stream.cuda_stream);
              worklist = m_send_remote_output_worklists[channel];
              break;
            }
            // This branch has the same logic but with different sequence of
            // execution
            if (!m_pass_remote_work_events[channel]->empty()) {
              m_pass_remote_work_events[channel]->pop().Wait(
                  stream.cuda_stream);
              worklist = m_pass_remote_output_worklists[channel];
              break;
            }
          }
          // Notified by SignalRemoteWork or ReceiveLoop
//          m_send_cv.wait(guard);
        }
      }

      if (m_exit)
        break;

      source = 1 - source;

      Stopwatch sw;
      Stopwatch sw1;

      m_send_times++;
      sw.start();

      auto segs = worklist->ToSegs(stream);

      std::stringstream ss;

      ss << "Dev: " << m_dev << " Channel:" << channel << " send "
         << worklist->GetLength(stream) << std::endl;

      std::cout << ss.str();

      for (Segment<TRemote> output_seg : segs) {
        std::shared_future<Event> ft =
            m_links_out[channel].Send(output_seg, Event());
        ft.get().Wait(stream.cuda_stream);

        worklist->PopItemsAsync(output_seg.GetSegmentSize(),
                                stream.cuda_stream);
        m_size_send[channel] += output_seg.GetSegmentSize() * sizeof(TRemote);
        m_seg_count[channel]++;
        m_max_send_size = std::max(
            m_max_send_size, output_seg.GetSegmentSize() * sizeof(TRemote));
      }
      sw.stop();
      stream.Sync();
      m_time_send += sw.ms();
    }

    std::stringstream ss;
    ss << "Dev: " << m_dev << " Channel:" << channel << " send exit";
    std::cout << ss.str();
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
        m_router_count(routers.size()),
        m_max_recv_size(max_exch_size) {
    for (auto& router : routers) {
      auto counter = std::make_shared<Counter>();

      m_links_in.emplace_back(*router, dev, max_exch_size, exch_buffs);
      m_links_out.emplace_back(dev, *router);
      m_filter_counters.push_back(counter);
      m_pass_remote_work_events.push_back(
          std::make_shared<BlockingQueue<Event>>());
      m_receive_work_events.push_back(std::make_shared<BlockingQueue<Event>>());
      m_send_remote_work_events.push_back(
          std::make_shared<BlockingQueue<Event>>());
    }

    void* mem_buffer;
    size_t mem_size;

    mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_in, mem_size, AF_PO2);
    m_local_input_worklist = groute::CircularWorklist<TLocal>(
        (TLocal*) mem_buffer, mem_size / sizeof(TLocal));
    m_local_input_worklist.ResetAsync((cudaStream_t) 0);

    for (int i = 0; i < m_router_count; i++) {
      {
        mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_out / m_router_count,
                                   mem_size, AF_PO2);
        auto wl = std::make_shared<groute::CircularWorklist<TRemote>>(
            (TRemote*) mem_buffer, mem_size / sizeof(TRemote));
        wl->ResetAsync((cudaStream_t) 0);
        m_dev_send_remote_output_worklists.push_back(wl->DeviceObject());
        m_send_remote_output_worklists.push_back(wl);
      }
      {
        mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_pass / m_router_count,
                                   mem_size, AF_PO2);
        auto wl = std::make_shared<groute::CircularWorklist<TRemote>>(
            (TRemote*) mem_buffer, mem_size / sizeof(TRemote));
        wl->ResetAsync((cudaStream_t) 0);
        m_dev_pass_remote_output_worklists.push_back(wl->DeviceObject());
        m_pass_remote_output_worklists.push_back(wl);
      }
    }

    mem_buffer = context.Alloc(FLAGS_wl_alloc_factor_local, mem_size);
    m_temp_worklist = groute::Worklist<TLocal>((TLocal*) mem_buffer,
                                               mem_size / sizeof(TLocal));
    m_temp_worklist.ResetAsync((cudaStream_t) 0);

    GROUTE_CUDA_CHECK(cudaStreamSynchronize((cudaStream_t) 0));  // just in case

    for (int channel = 0; channel < m_router_count; channel++) {
      m_receive_threads.push_back(
          std::thread([this, channel]() { ReceiveLoop(channel); }));
      m_send_threads.push_back(
          std::thread([this, channel]() { SendLoop(channel); }));
    }

    m_size_send.resize(m_router_count, 0);
    m_seg_count.resize(m_router_count, 0);
    m_split_send_time.resize(m_router_count, 0);
    m_split_recv_time.resize(m_router_count, 0);
  }

  ~MultiChannelDistributedWorklistPeer() {
    for (auto& th : m_receive_threads) {
      th.join();
    }
    for (auto& th : m_send_threads) {
      th.join();
    }

    std::stringstream ss;
    size_t total_size = 0;

    for (auto size : m_size_send) {
      total_size += size;
    }

    ss << "Dev " << m_dev << " SendOP enqueue times: " << m_send_times
       << " send time: " << m_time_send << " split time: " << m_time_split
       << " enqueue time: " << m_time_enqueue
       << " Total comm: " << total_size / 1024.0 / 1024.0 << " MB "
       << " Bandwidth: " << total_size / 1024.0 / 1024.0 / (m_time_send / 1024)
       << " MB/s"
       << " MaxSendSegSize " << m_max_send_size / 1024.0 / 1024.0 << " MB"
       << " MaxRecvSegSize " << m_max_recv_size / 1024.0 / 1024 << " MB"
       << std::endl;

    for (int i = 0; i < m_router_count; i++) {
      auto size_in_mb = m_size_send[i] / 1024.0 / 1024;
      ss << "    Ring " << i << " Size: " << size_in_mb << " MB"
         << " Seg count: " << m_seg_count[i]
         << " Avg size: " << size_in_mb / m_seg_count[i] << " MB"
         << " split recv time: " << m_split_recv_time[i] << std::endl;
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
      // FIXME: Consumer stuck on here
      while (!m_exit) {
        bool done = false;
        // waiting split is done
        for (int channel = 0; channel < m_router_count; channel++) {
          if (!m_receive_work_events[channel]->empty()) {
            m_receive_work_events[channel]->pop().Wait(stream.cuda_stream);
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }

      if (m_exit)
        return segs;

      segs = m_local_input_worklist.ToSegs(stream);
    }

    size_t len = 0;

    for (auto& seg : segs) {
      len += seg.GetSegmentSize();
    }

    std::cout << "Dev " << m_dev << " get size: " << len << std::endl;

    return segs;
  }

  void PerformSplitSend(Segment<TLocal>& split_work, Stream& stream) override {
    if (split_work.Empty())
      return;
    // All work items in m_local_input_worklist are split
    // the content in m_send_remote_output_worklist will be sent out by SendLoop
    SplitSend(split_work, m_local_input_worklist,
              m_send_remote_output_worklists,
              m_dev_send_remote_output_worklists, stream);
    Event split_ev = m_context.RecordEvent(m_dev, stream.cuda_stream);
    SignalRemoteWork(split_ev);
  }

  void SignalRemoteWork(const Event& ev) override {
    // Signal
    for (auto& q : m_send_remote_work_events) {
      q->push(ev);
    }
    std::cout << "signal " << std::endl;
    //    m_send_cv.notify_all();
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

    {
      std::lock_guard<std::mutex> lock(log_gate);

      std::cout << std::endl
                << '[' << std::this_thread::get_id() << ']'
                << "\t\tWork: " << work << ",\t\tCurrent: " << current_work
                << std::endl;
    }

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
