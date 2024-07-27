#include "Process.hpp"
#include <Process/Dataflow/WidgetInlets.hpp>
#include <score/tools/File.hpp>
#include <wobjectimpl.h>

W_OBJECT_IMPL(Onnx::Model)

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This sample application demonstrates how to use components of the experimental C++ API
 * to query for model inputs/outputs and how to run inferrence on a model.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *  2) The onnx model should have float input
 *
 *
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) generate random data for an input tensor
 *  4) pass tensor through the model and check the resulting tensor
 *
 */

#include <onnxruntime_cxx_api.h>
#undef READ
#undef WRITE
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
QDebug operator<<(QDebug s, ONNXType t)
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

QDebug operator<<(QDebug s, ONNXTensorElementDataType t)
{
  switch (t)
  {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
      return s << "UNDEFINED";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return s << "FLOAT"; // maps to c type float
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return s << "UINT8"; // maps to c type uint8_t
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

QDebug operator<<(QDebug s, const Ort::ConstTensorTypeAndShapeInfo& t)
{
  s << "\n - element type: " << t.GetElementType() << "["
    << t.GetElementCount() << "]";
  s << "\n - dimensions: " << t.GetDimensionsCount();
  s << "\n - shape: " << t.GetShape();
  return s;
}

QDebug operator<<(QDebug s, const Ort::ConstSequenceTypeInfo& t)
{
  s << "\n - element type: " << t.GetSequenceElementType();
  return s;
}
QDebug operator<<(QDebug s, const Ort::ConstMapTypeInfo& t)
{
  s << "\n - element type: " << t.GetMapKeyType() << " -> "
    << t.GetMapValueType();
  return s;
}

QDebug operator<<(QDebug s, const Ort::ConstOptionalTypeInfo& t)
{
  s << "\n - element type: " << t.GetOptionalElementType();
  return s;
}
QDebug operator<<(QDebug s, const Ort::ConstTypeInfo& t)
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
QDebug operator<<(QDebug s, const Ort::TypeInfo& t)
{
  return s << t.GetConst();
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t> &v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int calculate_product(const std::vector<std::int64_t> &v)
{
    int total = 1;
    for (auto &i : v)
        total *= i;
    return total;
}

Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
    OrtAllocatorType::OrtArenaAllocator,
    OrtMemType::OrtMemTypeDefault);
template <typename T>
Ort::Value
vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape)
{
  return Ort::Value::CreateTensor<T>(
      mem_info, data.data(), data.size(), shape.data(), shape.size());
}

struct ModelSpec
{
    struct Port
    {
        QString name;
        Process::PortType port_type{};
        ossia::val_type data_type{};
        std::vector<int64_t> shape;
    };
    std::vector<Port> inputs;
    std::vector<Port> outputs;

    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char;
    std::vector<const char *> output_names_char;
};

struct OnnxRunContext
{
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;

    // print name/shape of inputs
    Ort::AllocatorWithDefaultOptions allocator;

    ModelSpec readModelSpec()
    {
        ModelSpec spec;

        for (std::size_t i = 0; i < session.GetInputCount(); i++) {
            const std::string name = session.GetInputNameAllocated(i, allocator).get();
            const Ort::TypeInfo &input_type = session.GetInputTypeInfo(i);
            const Ort::ConstTensorTypeAndShapeInfo &input_tensor_type
                = input_type.GetTensorTypeAndShapeInfo();

            spec.inputs.push_back({.name = QString::fromStdString(name),
                                   .port_type = {},
                                   .data_type = {},
                                   .shape = input_tensor_type.GetShape()});

            // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
            for (int64_t& dim : spec.inputs.back().shape)
              if (dim < 0)
                dim = 1;

            spec.input_names.push_back(std::move(name));
        }

        for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
            const std::string name = session.GetOutputNameAllocated(i, allocator).get();
            const Ort::TypeInfo &output_type = session.GetOutputTypeInfo(i);
            const Ort::ConstTensorTypeAndShapeInfo &output_tensor_type
                = output_type.GetTensorTypeAndShapeInfo();

            spec.outputs.push_back({.name = QString::fromStdString(name),
                                    .port_type = {},
                                    .data_type = {},
                                    .shape = output_tensor_type.GetShape()});

            spec.output_names.push_back(std::move(name));
        }
        spec.input_names_char.resize(spec.input_names.size());
        spec.output_names_char.resize(spec.output_names.size());
        std::transform(std::begin(spec.input_names),
                       std::end(spec.input_names),
                       std::begin(spec.input_names_char),
                       [&](const std::string &str) { return str.c_str(); });

        std::transform(std::begin(spec.output_names),
                       std::end(spec.output_names),
                       std::begin(spec.output_names_char),
                       [&](const std::string &str) { return str.c_str(); });

        return spec;
    }

    explicit OnnxRunContext(QString name)
        : env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer")
        , session(env, name.toStdString().c_str(), session_options)
    {
        // Assume model has 1 input node and 1 output node.
        // assert(input_names.size() == 1 && output_names.size() == 1);
    }

    struct FloatTensor
    {
        std::vector<float> storage;
        Ort::Value value;
    };

    FloatTensor createRandomTensor(ModelSpec::Port &port)
    {
        // Create a single Ort tensor of random numbers
        auto &input_shape = port.shape;
        auto total_number_elements = calculate_product(input_shape);

        // generate random numbers in the range [0, 255]
        std::vector<float> input_tensor_values(total_number_elements);
        std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] {
            return rand() % 255;
        });

        FloatTensor f{
            .storage = {},
            .value = vec_to_tensor<float>(input_tensor_values, input_shape)};
        f.storage = std::move(input_tensor_values);
        return f;
    }

    template <typename T>
    void convert_rgb24_packed_to_planar(
        size_t width,
        size_t height,
        size_t step,
        const void* data,
        T* r_plane,
        T* g_plane,
        T* b_plane)
    {
      const uint8_t* src = static_cast<const uint8_t*>(data);
      step -= 3 * width;
      for (size_t y = 0; y < height; ++y)
      {
        for (size_t x = 0; x < width; ++x)
        {
          *r_plane++ = T(*src++) / 255.f;
          *g_plane++ = T(*src++) / 255.f;
          *b_plane++ = T(*src++) / 255.f;
        }
        src += step;
      }
    }

    FloatTensor tensorFromImage(
        ModelSpec::Port& port,
        int w,
        int h,
        bool normalize_resnet = false)
    {
      auto& input_shape = port.shape;
      QImage img("/home/jcelerier/Documents/ossia/score/packages/horse.jpeg");
      img = img.scaled(
          w,
          h,
          Qt::AspectRatioMode::KeepAspectRatioByExpanding,
          Qt::SmoothTransformation);
      img = img.copy(0, 0, w, h);
      img = img.convertToFormat(QImage::Format_RGB888);

      std::vector<float> input_tensor_values(3 * w * h);

      auto ptr = (unsigned char*)img.constBits();
      if (normalize_resnet)
      {
        for (int i = 0; i < 3 * w * h;)
        {
          // clang-format off
          // https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
          input_tensor_values[i + 0] = (float(*ptr & 0x00ff0000 >> 16) / 255.f - 0.485f) / 0.229f;
          input_tensor_values[i + 1] = (float(*ptr & 0x0000ff00 >> 8)  / 255.f - 0.456f) / 0.224f;
          input_tensor_values[i + 2] = (float(*ptr & 0x000000ff >> 0)  / 255.f - 0.406f) / 0.225f;
          // clang-format on
          i += 3;
          ptr++;
        }
      }
      else
      {
        auto dst = input_tensor_values.data();
        auto dst_r = dst;
        auto dst_g = dst_r + w * h;
        auto dst_b = dst_g + w * h;
        convert_rgb24_packed_to_planar<float>(
            w, h, 3 * w, ptr, dst_r, dst_g, dst_b);
      }

      FloatTensor f{
          .storage = {},
          .value = vec_to_tensor<float>(input_tensor_values, input_shape)};
      f.storage = std::move(input_tensor_values);
      return f;
    }
    void processOutput_yolo(
        const ModelSpec& spec,
        const std::vector<Ort::Value>& outputTensor)
    {
      if (outputTensor.size() > 0)
      {
        const float* arr = outputTensor.front().GetTensorData<float>();
        for (int i = 0; i < outputTensor.front()
                                .GetTensorTypeAndShapeInfo()
                                .GetElementCount();
             i += 7)
        {
          int class_type = static_cast<int>(arr[i + 5]);
          int accuracy = (arr[i + 6]);
          int ModelInputImageSize_width = 640;
          int ModelInputImageSize_height = 640;
          int original_cols = 640;
          int original_rows = 640;
          // clang-format off
          int x = arr[i + 1] / (float)ModelInputImageSize_width * original_cols;
          int y = arr[i + 2] / (float)ModelInputImageSize_height * original_rows;
          int w = (arr[i + 3] - arr[i + 1]) / (float)ModelInputImageSize_width * (float)original_cols;
          int h = (arr[i + 4] - arr[i + 2]) / (float)ModelInputImageSize_height * (float)original_rows;
          // clang-format on

          qDebug() << class_type << accuracy << x << y << w << h;
        }
      }
    }

    void processOutput_resnet(
        const ModelSpec& spec,
        const std::vector<Ort::Value>& output_tensors)
    {
      QFile f("/opt/models/imagenet_classes.txt");
      f.open(QIODevice::ReadOnly);
      auto classes = f.readAll().split('\n');

      for (const Ort::Value& ot : output_tensors)
      {
        std::cout << spec.output_names_char[0] << ": " << ot.GetTypeInfo()
                  << std::endl;
        std::span<const float> res = std::span(
            ot.GetTensorData<float>(),
            ot.GetTensorTypeAndShapeInfo().GetElementCount());
        int k = 0;
        for (auto val : res)
        {
          if (val > 5)
            qDebug() << k << classes[k] << val;
          k++;
        }
        qDebug() << std::distance(
            res.begin(), std::max_element(res.begin(), res.end()));
      }
    }

    void infer(const ModelSpec& spec, std::span<Ort::Value> input_tensors)
    {
      try
      {
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            spec.input_names_char.data(),
            input_tensors.data(),
            spec.input_names_char.size(),
            spec.output_names_char.data(),
            spec.output_names_char.size());

        processOutput_resnet(spec, output_tensors);
        // processOutput_yolo(spec, output_tensors);
        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(
            output_tensors.size() == spec.output_names.size()
            && output_tensors[0].IsTensor());

        auto info = output_tensors[0].GetTypeInfo();
        qDebug() << info;
      }
      catch (const Ort::Exception& exception)
      {
        std::cout << "ERROR running model inference: " << exception.what()
                  << std::endl;
        exit(-1);
      }
    }
};

auto x = []
{
  OnnxRunContext ctx(
      "/opt/models/models/validated/vision/classification/resnet/model/"
      "resnet101-v1-7.onnx");

  //OnnxRunContext ctx(
  //    "/opt/models/PINTO_model_zoo/307_YOLOv7/yolov7_640x640.onnx");
  //OnnxRunContext ctx("/opt/models/yolov7/yolov7.onnx");

  auto spec = ctx.readModelSpec();
  if (spec.inputs.empty())
  {
    qDebug() << "No input port";
    return 0;
  }

  if (std::ranges::any_of(spec.inputs[0].shape, [](int x) { return x <= 0; }))
  {
    qDebug() << "invalid shape" << spec.inputs[0].shape;
    return 0;
  }

  qDebug() << "Infering...";

  auto t = ctx.tensorFromImage(spec.inputs[0], 224, 224, false);

  Ort::Value input_tensors[1]{std::move(t.value)};
  qDebug() << input_tensors[0].GetTypeInfo();
  assert(
      input_tensors[0].IsTensor()
      && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()
             == spec.inputs[0].shape);
  std::cout << "\ninput_tensor shape: "
            << print_shape(
                   input_tensors[0].GetTensorTypeAndShapeInfo().GetShape())
            << std::endl;

  ctx.infer(spec, input_tensors);
  std::exit(0);
  return 0;
}();
namespace Onnx
{

Model::Model(const TimeVal &duration,
             const QString &name,
             const Id<Process::ProcessModel> &id,
             QObject *parent)
    : Process::ProcessModel{duration, id, "OnnxProcess", parent}
    , m_text{name}
{
  qDebug() << "Loading...";

  metadata().setInstanceName(*this);
  OnnxRunContext ctx(
      score::locateFilePath(name, score::IDocument::documentContext(*parent)));
  auto spec = ctx.readModelSpec();
  if (spec.inputs.empty())
  {
    qDebug() << "No input port";
    return;
  }

  if (std::ranges::any_of(spec.inputs[0].shape, [](int x) { return x <= 0; }))
  {
    qDebug() << "invalid shape" << spec.inputs[0].shape;
    return;
  }

  qDebug() << "Infering...";
  auto t = ctx.createRandomTensor(spec.inputs[0]);
  Ort::Value in[1]{std::move(t.value)};
  ctx.infer(spec, in);
}

Model::~Model() { }

QString Model::prettyName() const noexcept
{
  return tr("Onnx Process");
}
} // namespace Onnx
template <>
void DataStreamReader::read(const Onnx::Model& proc)
{
  insertDelimiter();
}

template <>
void DataStreamWriter::write(Onnx::Model& proc)
{
  checkDelimiter();
}

template <>
void JSONReader::read(const Onnx::Model& proc)
{}

template <>
void JSONWriter::write(Onnx::Model& proc)
{}
