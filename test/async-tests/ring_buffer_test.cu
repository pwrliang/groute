#include <groute/event_pool.h>
#include <groute/groute.h>
#include <groute/internal/pinned_allocation.h>
#include <groute/internal/worker.h>
#include <gtest/gtest.h>
#include <utils/ring_buffer.h>

#include <cmath>

TEST(Async, RingBuffer) {
  groute::RingBuffer<int32_t> ring_buffer(0, 1000);

  for (int x = 0; x < 10; x++) {
    for (int _ = 0; _ < 2; _++) {
      ring_buffer.GetWritableBuffer(80);
    }
    ring_buffer.CommitPending();
    std::cout << "Readable size: " << ring_buffer.GetReadableSize() << std::endl;
    auto segs = ring_buffer.GetReadableBuffer();
//    ring_buffer.PrintOffsets();
    std::cout << segs.size() << std::endl;
  }
}