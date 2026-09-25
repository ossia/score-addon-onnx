#pragma once
// Counts the allocations of at least `min_bytes` made on the calling thread
// between begin() and end() (the global operator new is replaced for the
// node test binary, in alloc_counter.cpp). ORT's own small objects stay
// under a few hundred bytes; the buffers the tests watch are larger.
#include <cstddef>

namespace AllocCounter
{
void begin(std::size_t min_bytes);
int end();
}
