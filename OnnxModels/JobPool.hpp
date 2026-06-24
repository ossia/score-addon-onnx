#pragma once
// Lock-free recycling pool for the worker inference jobs.
//
// score::TaskPool's queue element is a smallfun::function<void(), 128>; the
// posted lambda already captures ~32 bytes of context, so a worker job passed
// by value must stay under ~90 bytes — our InferJob structs (with their staged
// input vectors and recurrent-state buffers) are several times that. Instead
// the job lives on the heap and only a unique_ptr (8 bytes) travels through
// the queue.
//
// To keep request() RT-clean on the DSP thread, jobs are recycled through
// ossia::object_pool (ossia/detail/buffer_pool.hpp), which is backed by the
// lock-free moodycamel mpmc_queue: acquire() is a lock-free dequeue and the
// recycled job's internal vectors keep their capacity, so the steady state
// allocates nothing. The cold path (empty pool -> make_unique) matches the
// first-frame allocation profile we already have. release() happens on the
// worker thread, after work() has moved the results out.
//
// The pool type (ossia::object_pool) is provided by a vendored, API-compatible
// copy under Onnx/helpers/compat so this header carries no ossia/ include; in a
// score build the compat header transparently uses moodycamel's ConcurrentQueue
// (same lock-free backing as the real ossia header), standalone falls back to a
// mutex-guarded queue with the same semantics.
#include <Onnx/helpers/compat/buffer_pool.hpp>

#include <memory>

namespace OnnxModels
{
template <typename Job>
struct JobPool
{
  ossia::object_pool<std::unique_ptr<Job>> pool;

  std::unique_ptr<Job> acquire()
  {
    auto j = pool.acquire();
    if(!j)
      j = std::make_unique<Job>();
    return j;
  }

  void release(std::unique_ptr<Job> j)
  {
    if(j)
      pool.release(std::move(j));
  }

  static JobPool& instance()
  {
    static JobPool p;
    return p;
  }
};
}
