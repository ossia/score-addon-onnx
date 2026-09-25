#pragma once
// ossia::object_pool for JobPool.hpp.
//
// In a score build libossia is on the include path: its own header is used,
// so the pool is not defined twice when another header of the build includes
// ossia/detail/buffer_pool.hpp too.
//
// Standalone builds have no libossia: a vendored, API-compatible copy follows.
// JobPool only uses ossia::object_pool<std::unique_ptr<Job>>, whose backing
// store is a lock-free MPMC queue (ossia::mpmc_queue == moodycamel's
// ConcurrentQueue). When the moodycamel header is available it is used
// directly, keeping the lock-free acquire()/release(); otherwise a small
// mutex-guarded queue with the same API (acquire pops, release pushes) keeps
// the behaviour, if not the lock-freedom.

#if __has_include(<ossia/detail/buffer_pool.hpp>)
#include <ossia/detail/buffer_pool.hpp>
#else

#include <mutex>
#include <utility>
#if __has_include(<concurrentqueue/concurrentqueue.h>)
#define OSSIA_COMPAT_HAS_MOODYCAMEL 1
#include <concurrentqueue/concurrentqueue.h>
#elif __has_include(<concurrentqueue.h>)
#define OSSIA_COMPAT_HAS_MOODYCAMEL 1
#include <concurrentqueue.h>
#else
#define OSSIA_COMPAT_HAS_MOODYCAMEL 0
#include <vector>
#endif

namespace ossia
{
#if OSSIA_COMPAT_HAS_MOODYCAMEL
template <typename T>
using mpmc_queue = moodycamel::ConcurrentQueue<T>;
#else
// Minimal mutex-guarded MPMC queue with the subset of the moodycamel API that
// object_pool needs (try_dequeue + enqueue).
template <typename T>
class mpmc_queue
{
public:
  bool try_dequeue(T& t)
  {
    std::lock_guard lock{m_mutex};
    if(m_impl.empty())
      return false;
    t = std::move(m_impl.back());
    m_impl.pop_back();
    return true;
  }

  void enqueue(T&& t)
  {
    std::lock_guard lock{m_mutex};
    m_impl.push_back(std::move(t));
  }

private:
  std::mutex m_mutex;
  std::vector<T> m_impl;
};
#endif

template <typename Obj_T>
struct object_pool
{
  mpmc_queue<Obj_T> buffers;

  Obj_T acquire()
  {
    Obj_T b;
    buffers.try_dequeue(b);
    return b;
  }

  void release(Obj_T b) { buffers.enqueue(std::move(b)); }
};
}
#endif
