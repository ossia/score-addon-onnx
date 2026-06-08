#pragma once
// MSVC / clang-cl lack <unistd.h>. ctx references a few POSIX names that its
// MSVC libc doesn't provide; supply them here (this header is only on the
// include path for WIN32 && !MINGW). With CTX_EVENTS/THREADS/PTY=0 none of the
// terminal glue actually executes, so the function stubs only need to link.
#include <stdint.h>

// POSIX integer types ctx uses in always-compiled VT struct glue.
#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif
#ifndef _PID_T_DEFINED
#define _PID_T_DEFINED
typedef int pid_t; // not provided by the MSVC CRT
#endif

// Insurance: ctx uses M_PI; the proper fix is _USE_MATH_DEFINES before <math.h>
// (set in CMake), this guarded fallback covers any include-order surprise.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static __inline long write(int fd, const void* buf, unsigned long n)
{
  (void)fd;
  (void)buf;
  return (long)n;
}
static __inline int usleep(unsigned int usec)
{
  (void)usec;
  return 0;
}
