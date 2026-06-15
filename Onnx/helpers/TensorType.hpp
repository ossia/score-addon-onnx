#pragma once
// Dependency-free mirror of the ONNX/ORT tensor element types we handle. Keeping
// this out of <onnxruntime_cxx_api.h> lets ModelSpec and the tensor->texture
// (output) path stay ORT-free and standalone-compilable, in the same spirit as
// CoreTypes.hpp. The ORT enum is mapped to this in OnnxContext (the one place
// that already includes ORT).
#include <cstdint>

namespace Onnx
{
enum class TensorElemType : uint8_t
{
  Unknown = 0,
  Float,    // float32
  Float16,  // IEEE half
  BFloat16, // bfloat16
  Double,   // float64
  Uint8,
  Int8,
  Uint16,
  Int16,
  Uint32,
  Int32,
  Uint64,
  Int64,
  Bool,
};

// Bytes per element (0 for Unknown). Float16/BFloat16 are 2 bytes.
inline constexpr int elemSize(TensorElemType t) noexcept
{
  switch(t)
  {
    case TensorElemType::Float:
    case TensorElemType::Uint32:
    case TensorElemType::Int32:
      return 4;
    case TensorElemType::Float16:
    case TensorElemType::BFloat16:
    case TensorElemType::Uint16:
    case TensorElemType::Int16:
      return 2;
    case TensorElemType::Double:
    case TensorElemType::Uint64:
    case TensorElemType::Int64:
      return 8;
    case TensorElemType::Uint8:
    case TensorElemType::Int8:
    case TensorElemType::Bool:
      return 1;
    case TensorElemType::Unknown:
    default:
      return 0;
  }
}
} // namespace Onnx
