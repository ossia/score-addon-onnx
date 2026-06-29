#pragma once
// Qt-free debug helpers: map the ORT enums to human-readable strings. These
// used to be QDebug operator<< overloads; now they return const char* so the
// (compiled) standalone objects don't pull in <QDebug>. Callers print with
// std::fprintf(stderr, ...).
#include <Onnx/helpers/OnnxBase.hpp>

namespace Onnx
{
inline const char* to_string(ONNXType t)
{
  switch (t)
  {
    case ONNX_TYPE_UNKNOWN:
      return "unknown";
    case ONNX_TYPE_TENSOR:
      return "tensor";
    case ONNX_TYPE_SEQUENCE:
      return "sequence";
    case ONNX_TYPE_MAP:
      return "map";
    case ONNX_TYPE_OPAQUE:
      return "opaque";
    case ONNX_TYPE_SPARSETENSOR:
      return "sparse tensor";
    case ONNX_TYPE_OPTIONAL:
      return "optional";
  }
  return "?";
}

inline const char* to_string(ONNXTensorElementDataType t)
{
  switch (t)
  {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      return "UNDEFINED";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "UINT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "INT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "UINT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "INT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "INT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "INT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "STRING";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "BOOL";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "FLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "DOUBLE";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return "UINT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return "UINT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return "COMPLEX64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return "COMPLEX128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "BFLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      return "FLOAT8E4M3FN";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      return "FLOAT8E4M3FNUZ";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return "FLOAT8E5M2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      return "FLOAT8E5M2FNUZ";
  }
  return "?";
}
}
