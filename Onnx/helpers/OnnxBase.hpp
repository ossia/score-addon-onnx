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
// An ORT input tensor plus the float buffer backing it. The buffer is reused
// across frames (steady-state zero-alloc), so it must outlive `value`.
struct FloatTensor
{
  boost::container::vector<float> storage;
  Ort::Value value;
};
}
