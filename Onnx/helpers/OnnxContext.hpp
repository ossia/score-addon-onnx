#pragma once
#include <ossia/detail/algorithms.hpp>

#include <Onnx/helpers/Debug.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/Profile.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace Onnx
{
struct Options
{
  std::string provider = "default";
  int device_id = 0;
};

static Ort::SessionOptions create_session_options(const Options& opts)
try
{
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(
      kOrtSessionOptionsConfigUseORTModelBytesDirectly, "1");
  session_options.SetIntraOpNumThreads(
      1); // FIXME seemed to cause issues with Fast-VLM
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Sequential is ORT's default and most-tested teardown path. We run with
  // InterOpNumThreads(1), so ORT_PARALLEL bought no inter-op parallelism anyway
  // but did take the parallel-executor teardown path — a known crash class when
  // a node owning several sessions (a two-stage PoseDetector: landmark+detector)
  // is destroyed on stop. Use sequential to avoid it.
  session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options.DisableProfiling();
  static constexpr const char* device_ids[10]
      = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

  const char* device_id_str = device_ids[std::clamp(opts.device_id, 0, 8)];
  const OrtApi& api = Ort::GetApi();
  auto p = Ort::GetAvailableProviders();
  for (std::string& s : p)
  {
    std::fprintf(stderr, "Available provider: %s\n", s.c_str());
    if (s.ends_with("ExecutionProvider"))
      s.resize(s.size() - strlen("ExecutionProvider"));
    for (char& c : s)
      c = std::tolower(c);
  }

  std::string requested_provider = opts.provider;
  if (const char* env = std::getenv("SCORE_ONNX_FORCE_PROVIDER");
      env && *env)
  {
    std::string e = env;
    // trim
    auto notspace = [](unsigned char c) { return !std::isspace(c); };
    e.erase(e.begin(), std::find_if(e.begin(), e.end(), notspace));
    e.erase(std::find_if(e.rbegin(), e.rend(), notspace).base(), e.end());
    for (char& c : e)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (!e.empty())
      requested_provider = e;
  }
  if (requested_provider == "default")
  {
    if (ossia::contains(p, "cuda"))
      requested_provider = "cuda";
#if defined(_WIN32)
    else if (ossia::contains(p, "dml"))
      requested_provider = "dml";
#endif
    else if (ossia::contains(p, "rocm"))
      requested_provider = "rocm";
    else if (ossia::contains(p, "openvino"))
      requested_provider = "openvino";
#if defined(__APPLE__)
    else if (ossia::contains(p, "coreml"))
      requested_provider = "coreml";
#endif
    else if (ossia::contains(p, "webgpu"))
      requested_provider = "webgpu";
    else if (ossia::contains(p, "cpu"))
      requested_provider = "cpu";
  }

  if (requested_provider == "cuda" && ossia::contains(p, "cuda"))
  {
    using namespace Ort;

    OrtCUDAProviderOptionsV2* cuda_option_v2 = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_option_v2));
    const std::vector keys{
        "device_id",
        "arena_extend_strategy",
        "cudnn_conv_algo_search",
        "do_copy_in_default_stream",
        "cudnn_conv_use_max_workspace",
        "cudnn_conv1d_pad_to_nc1d",
        "enable_cuda_graph",
        "enable_skip_layer_norm_strict_mode"};
    const std::vector values{
        device_id_str,
        "kNextPowerOfTwo",
        "EXHAUSTIVE",
        "1",
        "1",
        "1",
        "0",
        "1"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cuda_option_v2, keys.data(), values.data(), keys.size()));
    // FIXME release options
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_option_v2);
  }

  if (requested_provider == "tensorrt" && ossia::contains(p, "tensorrt"))
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

  if (requested_provider == "rocm" && ossia::contains(p, "rocm"))
  {
    using namespace Ort;
    OrtROCMProviderOptions* options{};
    Ort::ThrowOnError(api.CreateROCMProviderOptions(&options));
    options->device_id = opts.device_id;
    session_options.AppendExecutionProvider_ROCM(*options);
    // FIXME release options
  }

  if (requested_provider == "openvino" && ossia::contains(p, "openvino"))
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
  if (requested_provider == "dml" && ossia::contains(p, "dml"))
  {
    using namespace Ort;

    std::unordered_map<std::string, std::string> options;
    session_options.AppendExecutionProvider("DML", options);
  }
#endif

#if __APPLE__
  if (requested_provider == "coreml" && ossia::contains(p, "coreml"))
  {
    using namespace Ort;

    std::unordered_map<std::string, std::string> options;

    // Note: https://github.com/apple/coremltools/issues/2301
    // options["ModelFormat"] = std::string("MLProgram");
    options["MLComputeUnits"] = "ALL";
    options["RequireStaticInputShapes"] = "0";
    options["EnableOnSubgraphs"] = "1";
    session_options.AppendExecutionProvider("CoreML", options);
  }
#endif

  if (requested_provider == "cpu" && ossia::contains(p, "cpu"))
  {
    int cpus = std::thread::hardware_concurrency();
    session_options.SetIntraOpNumThreads(std::max(cpus / 2, 1));
    session_options.SetInterOpNumThreads(std::max(cpus / 2, 1));
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  }

  // FIXME RKNPU
  // FIXME ARMNN, etc.

  return session_options;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "Onnxruntime: falling back to CPU: %s\n", e.what());
  return create_session_options(Options{.provider = "cpu", .device_id = 0});
}
catch (...)
{
  std::fprintf(stderr, "OnnxRuntime: falling back to CPU: unknown error\n");
  return create_session_options(Options{.provider = "cpu", .device_id = 0});
}

inline TensorElemType fromOrtElementType(ONNXTensorElementDataType t) noexcept
{
  switch(t)
  {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return TensorElemType::Float;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return TensorElemType::Float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:return TensorElemType::BFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return TensorElemType::Double;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return TensorElemType::Uint8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return TensorElemType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return TensorElemType::Uint16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return TensorElemType::Int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return TensorElemType::Uint32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return TensorElemType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return TensorElemType::Uint64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return TensorElemType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return TensorElemType::Bool;
    default:                                    return TensorElemType::Unknown;
  }
}

struct OnnxRunContext
{
  Options opts;
  Ort::Env env;

  Ort::SessionOptions session_options;
  Ort::Session session;

  Ort::AllocatorWithDefaultOptions allocator;

  // bytes is not the filename, it is the raw model binary data
  explicit OnnxRunContext(std::string_view bytes)
      : env(ORT_LOGGING_LEVEL_WARNING, "ossia")
      , session_options(create_session_options(opts))
      , session(env, bytes.data(), bytes.size(), session_options)
  {
    // The session (and therefore its I/O spec) is immutable for the context's
    // lifetime, so build the spec ONCE here. readModelSpec() then hands back a
    // reference instead of re-querying ORT and re-allocating names/vectors on
    // every frame — this is the per-frame hot path of every node.
    m_spec = buildModelSpec();
  }

  // Cached, immutable model I/O spec. Hot per-frame callers should bind with
  // `const auto&` to stay allocation-free; the worker threads hold the context
  // alive via shared_ptr, so the reference (and its name char*) stay valid.
  const ModelSpec& readModelSpec() const noexcept { return m_spec; }

private:
  ModelSpec buildModelSpec()
  {
    ONNX_PROF_SCOPE(ReadSpec);
    ModelSpec spec;

    for (std::size_t i = 0; i < session.GetInputCount(); i++)
    {
      const std::string name
          = session.GetInputNameAllocated(i, allocator).get();
      const Ort::TypeInfo& input_type = session.GetInputTypeInfo(i);
      const Ort::ConstTensorTypeAndShapeInfo& input_tensor_type
          = input_type.GetTensorTypeAndShapeInfo();

      spec.inputs.push_back(
          {.name = name,
           .port_type = {},
           .data_type = {},
           .shape = input_tensor_type.GetShape(),
           .elem_type = fromOrtElementType(input_tensor_type.GetElementType())});

      // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
      if (auto& tensor = spec.inputs.back();
          tensor.shape.size() == 4) // NCHW or NHCW
        if (tensor.shape[0] == -1)
          tensor.shape[0] = 1;

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
          {.name = name,
           .port_type = {},
           .data_type = {},
           .shape = output_tensor_type.GetShape(),
           .elem_type = fromOrtElementType(output_tensor_type.GetElementType())});

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

    spec.rebuildCharPointers();
    return spec;
  }

  ModelSpec m_spec; // built once in the ctor; returned by readModelSpec()

public:
  void infer(
      const ModelSpec& spec,
      std::span<Ort::Value> input_tensors,
      std::span<Ort::Value> output_values)
  {
    ONNX_PROF_SCOPE(Infer);
    try
    {
      // Counts MUST come from the caller-provided spans, not the model's full
      // declared name lists: callers pass fixed-size stack arrays sized to the
      // outputs they actually read. Using the full declared count would make ORT
      // write/read past those arrays for a model with more I/O than expected.
      session.Run(
          Ort::RunOptions{nullptr},
          spec.input_names_char.data(),
          input_tensors.data(),
          input_tensors.size(),
          spec.output_names_char.data(),
          output_values.data(),
          output_values.size());
    }
    catch (const Ort::Exception& exception)
    {
      // Per-frame failure (bad/odd output shape on a frame, ORT hiccup): log and
      // rethrow so the node's operator() catch skips this frame. NEVER exit() —
      // that would kill the whole host process on a single transient throw.
      std::fprintf(stderr, "ERROR running model inference: %s\n", exception.what());
      throw;
    }
  }
};

inline ModelSpec readModelSpec(std::string_view model_path)
{
  OnnxRunContext ctx{model_path};
  return ctx.readModelSpec();
}
}
