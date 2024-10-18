#pragma once
#include <ossia/detail/algorithms.hpp>

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
struct Options
{
  std::string provider = "cuda";
  int device_id = 0;
};

static Ort::SessionOptions create_session_options(const Options& opts)
{
  Ort::SessionOptions session_options;

  static constexpr const char* device_ids[10]
      = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

  const char* device_id_str = device_ids[std::clamp(opts.device_id, 0, 8)];
  const OrtApi& api = Ort::GetApi();
  auto p = Ort::GetAvailableProviders();
  for (std::string& s : p)
  {
    if (s.ends_with("ExecutionProvider"))
      s.resize(s.size() - strlen("ExecutionProvider"));
    for (char& c : s)
      c = std::tolower(c);
  }

  if (opts.provider == "cuda" && ossia::contains(p, "cuda"))
  {
    using namespace Ort;

    OrtCUDAProviderOptionsV2* cuda_option_v2 = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_option_v2));
    const std::vector keys{
        "device_id",
        "gpu_mem_limit",
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
        "cudnn_conv1d_pad_to_nc1d",
        "enable_cuda_graph",
        "enable_skip_layer_norm_strict_mode"};
    const std::vector values{
        device_id_str,
        "2147483648",
        "kNextPowerOfTwo",
        "EXHAUSTIVE",
        "1",
        "1",
        "1",
        "0",
        "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cuda_option_v2, keys.data(), values.data(), keys.size()));
    // FIXME release options
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_option_v2);
  }

  if (opts.provider == "tensorrt" && ossia::contains(p, "tensorrt"))
  {
    using namespace Ort;
    const std::vector keys{
        "device_id",
        "trt_engine_cache_enable",
        "trt_timing_cache_enable",
    };
    const std::vector values{device_id_str, "1", "1"};

    // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#shape-inference-for-tensorrt-subgraphs
    OrtTensorRTProviderOptionsV2* options{};
    Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&options));
    Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
        options, keys.data(), values.data(), keys.size()));
    session_options.AppendExecutionProvider_TensorRT_V2(*options);
    // FIXME release options
  }

  if (opts.provider == "rocm" && ossia::contains(p, "rocm"))
  {
    using namespace Ort;
    OrtROCMProviderOptions* options{};
    Ort::ThrowOnError(api.CreateROCMProviderOptions(&options));
    options->device_id = opts.device_id;
    session_options.AppendExecutionProvider_ROCM(*options);
    // FIXME release options
  }

  if (opts.provider == "openvino" && ossia::contains(p, "openvino"))
  {
    using namespace Ort;

    std::unordered_map<std::string, std::string> options;
    options["device_type"] = "GPU";
    options["precision"] = "FP32";
    session_options.AppendExecutionProvider("OpenVINO", options);

    // https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#onnxruntime-graph-level-optimization
    session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  }

#if _WIN32
  if (opts.provider == "dml" && ossia::contains(p, "dml"))
  {
    using namespace Ort;

    std::unordered_map<std::string, std::string> options;
    session_options.AppendExecutionProvider("DML", options);
  }
#endif

#if __APPLE__
  if (opts.provider == "coreml" && ossia::contains(p, "coreml"))
  {
    using namespace Ort;

    uint32_t coreml_flags = 0;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
        session_options, coreml_flags));
  }
#endif
  return session_options;
}
struct OnnxRunContext
{
  Options opts;
  Ort::Env env;

  Ort::SessionOptions session_options;
  Ort::Session session;

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;

  explicit OnnxRunContext(std::string_view name)
      : env(ORT_LOGGING_LEVEL_WARNING, "example")
      , session_options(create_session_options(opts))
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
