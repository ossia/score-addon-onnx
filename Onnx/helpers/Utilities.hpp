#pragma once
#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace Onnx
{

static int calculate_product(const std::vector<std::int64_t>& v)
{
  int total = 1;
  for (auto& i : v)
    total *= i;
  return total;
}

template <typename T>
void nhwc_to_nchw(
    size_t width,
    size_t height,
    size_t step,
    const void* data,
    T* __restrict__ r_plane,
    T* __restrict__ g_plane,
    T* __restrict__ b_plane,
    std::array<T, 3> mean,
    std::array<T, 3> std)
{
  const uint8_t* src = static_cast<const uint8_t*>(data);
  step -= 3 * width;
#pragma omp simd
  for (size_t y = 0; y < height; ++y)
  {
    for (size_t x = 0; x < width; ++x)
    {
      *r_plane++ = (T(*src++) - mean[0]) / std[0];
      *g_plane++ = (T(*src++) - mean[1]) / std[1];
      *b_plane++ = (T(*src++) - mean[2]) / std[2];
    }
    src += step;
  }
}

template <typename T>
Ort::Value
vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape)
{
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  return Ort::Value::CreateTensor<T>(
      mem_info, data.data(), data.size(), shape.data(), shape.size());
}

inline void softmax(std::span<const float> in, std::vector<float>& out)
{
  out.clear();
  out.resize(in.size());
#pragma omp simd
  for (int k = 0; k < in.size(); k++)
  {
    out[k] = std::exp(in[k]);
  }
  float esum = std::reduce(out.begin(), out.end());
  if (!(esum > 0.f))
    return;

#pragma omp simd
  for (int k = 0; k < in.size(); k++)
  {
    out[k] /= esum;
  }
}
}
