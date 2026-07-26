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

namespace Onnx
{
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

  return Ort::Env{threading, ORT_LOGGING_LEVEL_WARNING, logid};
}

// An ORT input tensor plus the float buffer backing it. The buffer is reused
// across frames (steady-state zero-alloc), so it must outlive `value`.
struct FloatTensor
{
  boost::container::vector<float> storage;
  Ort::Value value;
};
}
