#include "AllocCounter.hpp"

#include <cstdlib>
#include <new>

namespace
{
thread_local bool g_counting = false;
thread_local std::size_t g_min = 0;
thread_local int g_count = 0;
}

void AllocCounter::begin(std::size_t min_bytes)
{
  g_min = min_bytes;
  g_count = 0;
  g_counting = true;
}

int AllocCounter::end()
{
  g_counting = false;
  return g_count;
}

void* operator new(std::size_t n)
{
  if(g_counting && n >= g_min)
    g_count++;
  if(void* p = std::malloc(n ? n : 1))
    return p;
  throw std::bad_alloc{};
}
void operator delete(void* p) noexcept
{
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept
{
  std::free(p);
}
