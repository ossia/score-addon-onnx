#pragma once
#include <QFile>
#include <QImage>

#include <Onnx/helpers/Debug.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/Utilities.hpp>
#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace Onnx
{
struct OnnxRunContext
{
  Ort::Env env;

  Ort::SessionOptions session_options = []
  {
    Ort::SessionOptions session_options;
    const auto& api = Ort::GetApi();
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));

    // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#shape-inference-for-tensorrt-subgraphs
    /*
    OrtTensorRTProviderOptionsV2* tensorrt_options;
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
    std::unique_ptr<
        OrtTensorRTProviderOptionsV2,
        decltype(api.ReleaseTensorRTProviderOptions)>
        rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
        static_cast<OrtSessionOptions*>(session_options),
        rel_trt_options.get()));
*/
    return session_options;
  }();
  Ort::Session session;

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;

  explicit OnnxRunContext(std::string_view name)
      : env(ORT_LOGGING_LEVEL_WARNING, "example")
      , session(env, name.data(), session_options)
  {
  }

  ModelSpec readModelSpec()
  {
    ModelSpec spec;

    for (std::size_t i = 0; i < session.GetInputCount(); i++)
    {
      const std::string name
          = session.GetInputNameAllocated(i, allocator).get();
      const Ort::TypeInfo& input_type = session.GetInputTypeInfo(i);
      const Ort::ConstTensorTypeAndShapeInfo& input_tensor_type
          = input_type.GetTensorTypeAndShapeInfo();

      spec.inputs.push_back(
          {.name = QString::fromStdString(name),
           .port_type = {},
           .data_type = {},
           .shape = input_tensor_type.GetShape()});

      // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
      for (int64_t& dim : spec.inputs.back().shape)
        if (dim < 0)
          dim = 1;

      spec.input_names.push_back(std::move(name));
    }

    for (std::size_t i = 0; i < session.GetOutputCount(); i++)
    {
      const std::string name
          = session.GetOutputNameAllocated(i, allocator).get();
      const Ort::TypeInfo& output_type = session.GetOutputTypeInfo(i);
      const Ort::ConstTensorTypeAndShapeInfo& output_tensor_type
          = output_type.GetTensorTypeAndShapeInfo();

      spec.outputs.push_back(
          {.name = QString::fromStdString(name),
           .port_type = {},
           .data_type = {},
           .shape = output_tensor_type.GetShape()});

      spec.output_names.push_back(std::move(name));
    }
    spec.input_names_char.resize(spec.input_names.size());
    spec.output_names_char.resize(spec.output_names.size());
    std::transform(
        std::begin(spec.input_names),
        std::end(spec.input_names),
        std::begin(spec.input_names_char),
        [&](const std::string& str) { return str.c_str(); });

    std::transform(
        std::begin(spec.output_names),
        std::end(spec.output_names),
        std::begin(spec.output_names_char),
        [&](const std::string& str) { return str.c_str(); });

    return spec;
  }

  void infer(
      const ModelSpec& spec,
      std::span<Ort::Value> input_tensors,
      std::span<Ort::Value> output_values)
  {
    try
    {
      session.Run(
          Ort::RunOptions{nullptr},
          spec.input_names_char.data(),
          input_tensors.data(),
          spec.input_names_char.size(),
          spec.output_names_char.data(),
          output_values.data(),
          spec.output_names_char.size());
    }
    catch (const Ort::Exception& exception)
    {
      std::cout << "ERROR running model inference: " << exception.what()
                << std::endl;
      exit(-1);
    }
  }
  /*
  // processOutput_resnet(spec, output_tensors);
  processOutput_yolo(spec, output_tensors);
  // double-check the dimensions of the output tensors
  // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
  assert(
      output_tensors.size() == spec.output_names.size()
      && output_tensors[0].IsTensor());

  auto info = output_tensors[0].GetTypeInfo();
  qDebug() << info;
} catch (const Ort::Exception& exception)
{
  std::cout << "ERROR running model inference: " << exception.what()
            << std::endl;
  exit(-1);
}
}
*/
};
}
