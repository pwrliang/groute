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

#ifndef __GROUTE_POLICY_H
#define __GROUTE_POLICY_H
#include <groute/router.h>

#include <algorithm>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "gflags/gflags.h"

DECLARE_int32(nrings);

namespace groute {
namespace router {

/**
 * @brief A general purpose Policy object based on a topology
 */
class Policy : public IPolicy {
 private:
  std::vector<RoutingTable> m_tables;
  RouteStrategy m_strategy;

 public:
  Policy(const RoutingTable& topology, RouteStrategy strategy = Availability)
      : m_tables{topology}, m_strategy(strategy) {}

  Policy(std::vector<RoutingTable> tables,
         RouteStrategy strategy = Availability)
      : m_tables(std::move(tables)), m_strategy(strategy) {}

  RoutingTable GetRoutingTable() override {
    RoutingTable full_table;

    for (auto& table : m_tables) {
      for (auto& e : table) {
        auto src = e.first;
        auto& dsts = e.second;

        for (auto dst : dsts) {
          auto& merged_dsts = full_table[src];

          if (std::find(merged_dsts.begin(), merged_dsts.end(), dst) ==
              merged_dsts.end()) {
            merged_dsts.push_back(dst);
          }
        }
      }
    }

    return full_table;
  }

  Route GetRoute(device_t src_dev, int message_metadata) override {
    RoutingTable topology;
    if (m_tables.size() == 1 || message_metadata == -1) {
      topology = m_tables[0];
    } else {
      topology = m_tables[message_metadata];
    }

    //    std::cout << "ring id: " << message_metadata << std::endl;

    assert(topology.find(src_dev) != topology.end());

    return Route(topology.at(src_dev), m_strategy);
  }

  int GetRouteNum() const override { return m_tables.size(); }

  static std::shared_ptr<IPolicy> CreateBroadcastPolicy(
      device_t src_dev, const std::vector<device_t>& dst_devs) {
    RoutingTable topology;
    topology[src_dev] = dst_devs;
    return std::make_shared<Policy>(topology, Broadcast);
  }

  static std::shared_ptr<IPolicy> CreateScatterPolicy(
      device_t src_dev, const std::vector<device_t>& dst_devs) {
    RoutingTable topology;
    topology[src_dev] = dst_devs;
    return std::make_shared<Policy>(topology, Availability);
  }

  static std::shared_ptr<IPolicy> CreateP2PPolicy(device_t src_dev,
                                                  device_t dst_dev) {
    RoutingTable topology;
    topology[src_dev] = {dst_dev};
    return std::make_shared<Policy>(topology, Availability);
  }

  static std::shared_ptr<IPolicy> CreateGatherPolicy(
      device_t dst_dev, const std::vector<device_t>& src_devs) {
    RoutingTable topology;
    for (const device_t& src_dev : src_devs)
      topology[src_dev] = {dst_dev};
    return std::make_shared<Policy>(topology, Availability);
  }

  static std::shared_ptr<IPolicy> CreateOneWayReductionPolicy(int ndevs) {
    assert(ndevs > 0);

    // Each device N can send to devices [0...N-1]

    RoutingTable topology;

    for (device_t i = 0; i < ndevs; i++) {
      topology[i] = range(i);
    }
    topology[0].push_back(Device::Host);

    return std::make_shared<Policy>(topology, Availability);
  }

  static std::shared_ptr<IPolicy> CreateTreeReductionPolicy(int ndevs) {
    assert(ndevs > 0);

    RoutingTable topology;

    // 0
    // 1 -> 0
    // 2 -> 0
    // 3 -> 2
    // 4 -> 0
    // 5 -> 4
    // 6 -> 4
    // 7 -> 6
    // ..

    unsigned int p = next_power_2((unsigned int) ndevs) / 2;
    unsigned int stride = 1;

    while (p > 0) {
      for (int i = 0; i < p; i++) {
        int to = stride * (2 * i);
        int from = stride * (2 * i + 1);

        from = std::min(ndevs - 1, from);
        if (from <= to)
          continue;

        topology[(device_t) from].push_back((device_t) to);
      }

      p /= 2;
      stride *= 2;
    }

    // add host as a receiver for the drain device
    topology[0].push_back(Device::Host);

    return std::make_shared<Policy>(topology, Availability);
  }

  static std::shared_ptr<IPolicy> CreateRingPolicy(int ndevs) {
    assert(ndevs > 0);

    RoutingTable topology;

    for (device_t i = 0; i < ndevs; i++) {
      topology[i] = {(i + 1) % ndevs};
    }
    //    if (ndevs == 8) {
    //      topology[0] = {3};
    //      topology[1] = {7};
    //      topology[2] = {1};
    //      topology[3] = {2};
    //      topology[4] = {5};
    //      topology[5] = {6};
    //      topology[6] = {0};
    //      topology[7] = {4};
    //    }
    // Instead of pushing to GPU 0, we push tasks to the first available device,
    // this is beneficial for the case where the first device is already
    // utilized with a prior task.
    topology[Device::Host] = range(ndevs);  // for initial work from host

    return std::make_shared<Policy>(topology, Availability);
  }

  static std::shared_ptr<IPolicy> CreateMultiRingsPolicy(int ndevs) {
    assert(ndevs > 0);

    if (ndevs != 8) {
      return CreateRingPolicy(ndevs);
    }
    std::vector<std::vector<int>> seqs{{0, 3, 2, 1, 7, 4, 5, 6},
                                       {0, 6, 5, 4, 7, 1, 2, 3},
                                       {0, 2, 4, 6, 7, 5, 3, 1},
                                       {0, 1, 3, 5, 7, 6, 4, 2}};

    if (FLAGS_nrings > seqs.size()) {
      std::cerr << "Too many rings" << std::endl;
      std::exit(1);
    }
    seqs.resize(FLAGS_nrings);

    std::cout << "seqs " << seqs.size() << std::endl;

    std::vector<RoutingTable> tables;

    for (auto& seq : seqs) {
      RoutingTable topology;

      for (int i = 0; i < seq.size(); i++) {
        topology[seq[i]] = {seq[(i + 1) % seq.size()]};
      }

      // Instead of pushing to GPU 0, we push tasks to the first available
      // device, this is beneficial for the case where the first device is
      // already utilized with a prior task.
      topology[Device::Host] = range(ndevs);  // for initial work from host
      tables.push_back(topology);
    }

    return std::make_shared<Policy>(tables, Availability);
  }
};

class SimplePolicy : public IPolicy {
 private:
  RoutingTable m_table;
  RouteStrategy m_strategy;

 public:
  SimplePolicy(const RoutingTable& topology,
               RouteStrategy strategy = Availability)
      : m_table{topology}, m_strategy(strategy) {}

  RoutingTable GetRoutingTable() override { return m_table; }

  Route GetRoute(device_t src_dev, int message_metadata) override {
    assert(m_table.find(src_dev) != m_table.end());

    return Route(m_table.at(src_dev), m_strategy);
  }

  int GetRouteNum() const override { return 1; }

  static std::shared_ptr<IPolicy> CreateRingPolicy(
      const std::vector<int>& dev_seq) {
    RoutingTable topology;

    for (device_t i = 0; i < dev_seq.size(); i++) {
      topology[dev_seq[i]] = {dev_seq[(i + 1) % dev_seq.size()]};
    }
    // Instead of pushing to GPU 0, we push tasks to the first available device,
    // this is beneficial for the case where the first device is already
    // utilized with a prior task.
    topology[Device::Host] = dev_seq;  // for initial work from host

    return std::make_shared<Policy>(topology, Availability);
  }
};
}  // namespace router
}  // namespace groute

#endif  // __GROUTE_POLICY_H
