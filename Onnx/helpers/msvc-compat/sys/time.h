#pragma once
// MSVC / bare-clang-Windows lack <sys/time.h>. ctx actually reaches
// gettimeofday() (clock init), so provide a real implementation via C11
// timespec_get (present in the UCRT and clang). struct timeval is defined here
// because we deliberately do not pull <winsock2.h>.
#include <time.h>

// Use the same guard MSVC's own headers (winsock2.h / <sys/types.h>) use, so
// our definition coexists with theirs if either is ever pulled into this TU.
#ifndef _TIMEVAL_DEFINED
#define _TIMEVAL_DEFINED
struct timeval
{
  long tv_sec;
  long tv_usec;
};
#endif

// static (internal linkage): never emitted as a global symbol, so it cannot
// collide at link with a gettimeofday defined elsewhere in the codebase.
static __inline int gettimeofday(struct timeval* tv, void* tz)
{
  struct timespec ts;
  (void)tz;
  timespec_get(&ts, TIME_UTC);
  tv->tv_sec = (long)ts.tv_sec;
  tv->tv_usec = (long)(ts.tv_nsec / 1000);
  return 0;
}
