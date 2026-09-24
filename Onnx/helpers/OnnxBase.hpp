#pragma once

#include <cstdlib>
#include <version>

#if !defined(_MSC_VER)
  #ifndef _Frees_ptr_opt_
    #define _Frees_ptr_opt_
  #endif
#ifndef _Return_type_success_
#define _Return_type_success_(...)
#endif
#else
  #define __restrict__
#endif

#include <onnxruntime_cxx_api.h>
#if __APPLE__
#include <coreml_provider_factory.h>
#endif

#include <boost/container/vector.hpp>

#include <array>
#include <cstdio>
#include <mutex>
#include <string>

namespace Onnx
{
// ORT's log goes through this sink rather than its default one:
// - a thread can mute it for a moment (QuietOrtLog), to try something that is
//   expected to fail, without muting that session for its whole life as a
//   session log level would;
// - a message identical to one of the last few is dropped, so a model that
//   fails on every frame logs its error once, not at the frame rate.
namespace log_detail
{
inline thread_local int quiet = 0;

inline void ORT_API_CALL
sink(void*, OrtLoggingLevel severity, const char*, const char* logid,
     const char* where, const char* message)
{
  if(quiet > 0 || !message)
    return;
  static std::mutex mutex;
  static std::array<std::string, 8> recent;
  static std::size_t next = 0;
  std::lock_guard lock{mutex};
  for(const auto& r : recent)
    if(r == message)
      return;
  recent[next] = message;
  next = (next + 1) % recent.size();
  static constexpr char levels[] = "VIWEF";
  const int l = (int)severity;
  std::fprintf(
      stderr, "[%c:onnxruntime:%s, %s] %s\n", (l >= 0 && l < 5) ? levels[l] : '?',
      logid ? logid : "", where ? where : "", message);
}
}

// Mutes ORT's log on this thread while alive.
struct QuietOrtLog
{
  QuietOrtLog() noexcept { ++log_detail::quiet; }
  ~QuietOrtLog() { --log_detail::quiet; }
  QuietOrtLog(const QuietOrtLog&) = delete;
  QuietOrtLog& operator=(const QuietOrtLog&) = delete;
};

// ORT's environment is a process-wide singleton and the first creation decides
// whether it has global thread pools; a session with
// use_per_session_threads=false (the WebAssembly default) requires them.
inline Ort::Env make_env(const char* logid)
{
  static Ort::ThreadingOptions threading = [] {
    Ort::ThreadingOptions opts;
    opts.SetGlobalIntraOpNumThreads(1);
    opts.SetGlobalInterOpNumThreads(1);
    return opts;
  }();

  return Ort::Env{threading, &log_detail::sink, nullptr, ORT_LOGGING_LEVEL_WARNING, logid};
}

// An ORT input tensor plus the float buffer backing it. The buffer is reused
// across frames (steady-state zero-alloc), so it must outlive `value`.
struct FloatTensor
{
  boost::container::vector<float> storage;
  Ort::Value value;
};
}
