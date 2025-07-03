#pragma once
#include <QDebug>

#include <Onnx/helpers/OnnxBase.hpp>

inline QDebug operator<<(QDebug s, ONNXType t)
{
  switch (t)
  {
    case ONNX_TYPE_UNKNOWN:
      return s << "unknown";
    case ONNX_TYPE_TENSOR:
      return s << "tensor";
    case ONNX_TYPE_SEQUENCE:
      return s << "sequence";
    case ONNX_TYPE_MAP:
      return s << "map";
    case ONNX_TYPE_OPAQUE:
      return s << "opaque";
    case ONNX_TYPE_SPARSETENSOR:
      return s << "sparse tensor";
    case ONNX_TYPE_OPTIONAL:
      return s << "optional";
  }
  return s;
}

inline QDebug operator<<(QDebug s, ONNXTensorElementDataType t)
{
  switch (t)
  {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      return s << "UNDEFINED";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return s << "FLOAT"; // maps to c type float
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
      return s << "UINT4"; // ??
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return s << "UINT8"; // maps to c type uint8_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      return s << "INT4"; // ??
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return s << "INT8"; // maps to c type int8_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return s << "UINT16"; // maps to c type uint16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return s << "INT16"; // maps to c type int16_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return s << "INT32"; // maps to c type int32_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return s << "INT64"; // maps to c type int64_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return s << "STRING"; // maps to c++ type std::string
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return s << "BOOL";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return s << "FLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return s << "DOUBLE"; // maps to c type double
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return s << "UINT32"; // maps to c type uint32_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return s << "UINT64"; // maps to c type uint64_t
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      return s
             << "COMPLEX64"; // complex with float32 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      return s
             << "COMPLEX128"; // complex with float64 real and imaginary components
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return s
             << "BFLOAT16"; // Non-IEEE floating-point format based on IEEE754 single-precision
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
      return s
             << "FLOAT8E4M3FN"; // Non-IEEE floating-point format based on IEEE754 single-precision
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
      return s
             << "FLOAT8E4M3FNUZ"; // Non-IEEE floating-point format based on IEEE754 single-precision
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
      return s
             << "FLOAT8E5M2"; // Non-IEEE floating-point format based on IEEE754 single-precision
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
      return s
             << "FLOAT8E5M2FNUZ"; // Non-IEEE floating-point format based on IEEE754 single-precision
  }
  return s;
}

inline QDebug operator<<(QDebug s, const Ort::ConstTensorTypeAndShapeInfo& t)
{
  s << "\n - element type: " << t.GetElementType() << "["
    << t.GetElementCount() << "]";
  s << "\n - dimensions: " << t.GetDimensionsCount();
  s << "\n - shape: " << t.GetShape();
  return s;
}

inline QDebug operator<<(QDebug s, const Ort::ConstSequenceTypeInfo& t)
{
  s << "\n - element type: " << t.GetSequenceElementType();
  return s;
}

inline QDebug operator<<(QDebug s, const Ort::ConstMapTypeInfo& t)
{
  s << "\n - element type: " << t.GetMapKeyType() << " -> "
    << t.GetMapValueType();
  return s;
}

inline QDebug operator<<(QDebug s, const Ort::ConstOptionalTypeInfo& t)
{
  s << "\n - element type: " << t.GetOptionalElementType();
  return s;
}

inline QDebug operator<<(QDebug s, const Ort::ConstTypeInfo& t)
{
  switch (t.GetONNXType())
  {
    case ONNX_TYPE_UNKNOWN:
      return s << "unknown";
    case ONNX_TYPE_TENSOR:
      return s << "tensor: " << t.GetTensorTypeAndShapeInfo();
    case ONNX_TYPE_SEQUENCE:
      return s << "sequence: " << t.GetSequenceTypeInfo();
    case ONNX_TYPE_MAP:
      return s << "map: " << t.GetMapTypeInfo();
    case ONNX_TYPE_OPAQUE:
      return s << "opaque";
    case ONNX_TYPE_SPARSETENSOR:
      return s << "sparse tensor";
    case ONNX_TYPE_OPTIONAL:
      return s << "optional: " << t.GetOptionalTypeInfo();
  }
  return s;
}

inline QDebug operator<<(QDebug s, const Ort::TypeInfo& t)
{
  return s << t.GetConst();
}
