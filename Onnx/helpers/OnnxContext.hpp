#pragma once
#include <Onnx/helpers/Debug.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/Profile.hpp>
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>
#include <onnxruntime_session_options_config_keys.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <ranges>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace Onnx
{
// Replaces ossia::contains (formerly ossia/detail/algorithms.hpp): a plain
// "does this range contain this value" check, kept here so the helpers stay
// free of any ossia/score include in a standalone build.
template <typename Range, typename Value>
inline bool contains(const Range& r, const Value& v)
{
  return std::ranges::find(r, v) != std::ranges::end(r);
}

struct Options
{
  std::string provider = "default";
  int device_id = 0;
  // CUDA runs convolutions and matmuls in TF32 by default on Ampere and
  // newer (10-bit mantissa): harmless for images, audible for audio, where a
  // model that boosts part of the spectrum amplifies the noise floor.
  bool tf32 = true;

  // For the audio nodes: full float32 precision.
  static Options precise() { return {.tf32 = false}; }
};

// resolved (optional): receives the execution provider actually appended
// ("cuda", "tensorrt", ..., "cpu"), after the SCORE_ONNX_FORCE_PROVIDER
// override and the "default" pick; "cpu" when nothing else was available.
static Ort::SessionOptions
create_session_options(const Options& opts, std::string* resolved = nullptr)
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
    if (contains(p, "cuda"))
      requested_provider = "cuda";
#if defined(_WIN32)
    else if (contains(p, "dml"))
      requested_provider = "dml";
#endif
    else if (contains(p, "rocm"))
      requested_provider = "rocm";
    else if (contains(p, "openvino"))
      requested_provider = "openvino";
#if defined(__APPLE__)
    else if (contains(p, "coreml"))
      requested_provider = "coreml";
#endif
    else if (contains(p, "webgpu"))
      requested_provider = "webgpu";
    else if (contains(p, "webnn"))
      requested_provider = "webnn";
    else if (contains(p, "cpu"))
      requested_provider = "cpu";
  }

  if (requested_provider == "cuda" && contains(p, "cuda"))
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
        "enable_skip_layer_norm_strict_mode",
        "use_tf32"};
    const std::vector values{
        device_id_str,
        "kNextPowerOfTwo",
        "EXHAUSTIVE",
        "1",
        "1",
        "1",
        "0",
        "1",
        opts.tf32 ? "1" : "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
        cuda_option_v2, keys.data(), values.data(), keys.size()));
    // FIXME release options
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_option_v2);
  }

  if (requested_provider == "tensorrt" && contains(p, "tensorrt"))
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

  if (requested_provider == "rocm" && contains(p, "rocm"))
  {
    using namespace Ort;
    OrtROCMProviderOptions* options{};
    Ort::ThrowOnError(api.CreateROCMProviderOptions(&options));
    options->device_id = opts.device_id;
    session_options.AppendExecutionProvider_ROCM(*options);
    // FIXME release options
  }

  if (requested_provider == "openvino" && contains(p, "openvino"))
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
  if (requested_provider == "dml" && contains(p, "dml"))
  {
    using namespace Ort;

    std::unordered_map<std::string, std::string> options;
    session_options.AppendExecutionProvider("DML", options);
  }
#endif

#if __APPLE__
  if (requested_provider == "coreml" && contains(p, "coreml"))
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

  if (requested_provider == "cpu" && contains(p, "cpu"))
  {
    int cpus = std::thread::hardware_concurrency();
    session_options.SetIntraOpNumThreads(std::max(cpus / 2, 1));
    session_options.SetInterOpNumThreads(std::max(cpus / 2, 1));
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  }

  // FIXME RKNPU
  // FIXME ARMNN, etc.

  if (resolved)
    *resolved = contains(p, requested_provider) ? requested_provider : "cpu";
  return session_options;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "Onnxruntime: falling back to CPU: %s\n", e.what());
  return create_session_options(Options{.provider = "cpu", .device_id = 0}, resolved);
}

catch (...)
{
  std::fprintf(stderr, "OnnxRuntime: falling back to CPU: unknown error\n");
  return create_session_options(Options{.provider = "cpu", .device_id = 0}, resolved);
}

// Session creation with a fallback: some fp16 exports (e.g. the FastVLM-0.5B
// vision_encoder_fp16) crash ORT's extended-level graph fusions during
// session initialization ("Tensor type mismatch. T != MLFloat16" from
// tensor.h:210, a float-only fusion kernel touching an fp16 tensor). Basic
// optimizations initialize and run those models fine, so retry with them.
// The first attempt is silenced, or ORT logs that recovered failure as an
// error; the retry keeps the caller's options (provider, threads) and logs as
// usual, so a model that fails both ways is still reported.
template <typename PathString>
inline std::unique_ptr<Ort::Session> create_session_with_fallback(
    Ort::Env& env,
    const PathString& path,
    const Ort::SessionOptions& sessionOptions)
{
  try
  {
    // Muted only for this attempt: the session keeps its normal log level,
    // so its warnings and run errors are still printed afterwards.
    QuietOrtLog quiet;
    return std::make_unique<Ort::Session>(env, path.data(), sessionOptions);
  }
  catch (const Ort::Exception& e)
  {
    auto fallback = sessionOptions.Clone();
    fallback.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);
    auto session = std::make_unique<Ort::Session>(env, path.data(), fallback);
    std::fprintf(
        stderr,
        "Onnxruntime: loaded with basic graph optimizations (extended failed: "
        "%s)\n",
        e.what());
    return session;
  }
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

inline ONNXTensorElementDataType toOrtElementType(TensorElemType t) noexcept
{
  switch(t)
  {
    case TensorElemType::Float16:  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case TensorElemType::BFloat16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case TensorElemType::Double:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case TensorElemType::Uint8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case TensorElemType::Int8:     return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case TensorElemType::Uint16:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case TensorElemType::Int16:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case TensorElemType::Uint32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case TensorElemType::Int32:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case TensorElemType::Uint64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case TensorElemType::Int64:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case TensorElemType::Bool:     return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case TensorElemType::Float:
    default:                       return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
}

struct OnnxRunContext
{
  Options opts;
  // The execution provider the session was created for ("cuda", "cpu", ...;
  // see create_session_options). ORT may still place single nodes on the CPU.
  std::string provider;
  Ort::Env env;

  Ort::SessionOptions session_options;
  Ort::Session session;

  Ort::AllocatorWithDefaultOptions allocator;

  // bytes is not the filename, it is the raw model binary data.
  // model_path is the file the bytes were read from, if known: a session built
  // from a buffer has no base directory, so without it the external data of a
  // model (model.onnx_data next to model.onnx) is looked up in the process's
  // working directory and not found.
  explicit OnnxRunContext(
      std::string_view bytes, std::string_view model_path = {}, Options o = {})
      : opts(std::move(o))
      , env(make_env("ossia"))
      , session_options(
            withModelFolder(create_session_options(opts, &provider), model_path))
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

  // Transparent comparator: lookups by string_view don't build a std::string.
  using MetadataMap = std::map<std::string, std::string, std::less<>>;

  // The graph's custom metadata (ModelProto.metadata_props, e.g. InstantHMR's
  // `cliff_focal` / `image_size`). Read once (call_once, for the worker threads
  // sharing a context) and cached: later calls return the same map by
  // reference, without allocating. An unreadable block yields an empty map.
  const MetadataMap& metadata() const
  {
    std::call_once(m_metadata_once, [this] {
      try
      {
        Ort::AllocatorWithDefaultOptions alloc;
        Ort::ModelMetadata md = session.GetModelMetadata();
        auto keys = md.GetCustomMetadataMapKeysAllocated(alloc);
        for(auto& k : keys)
        {
          if(!k)
            continue;
          auto v = md.LookupCustomMetadataMapAllocated(k.get(), alloc);
          m_metadata.emplace(k.get(), v ? v.get() : "");
        }
      }
      catch(...)
      {
        m_metadata.clear();
      }
    });
    return m_metadata;
  }

  // One custom metadata value, or `fallback` when the key is absent. The
  // returned view points into the cached map (valid for the context's life).
  std::string_view
  metadataValue(std::string_view key, std::string_view fallback = {}) const
  {
    const auto& md = metadata();
    const auto it = md.find(key);
    return it != md.end() ? std::string_view(it->second) : fallback;
  }

private:
  static Ort::SessionOptions
  withModelFolder(Ort::SessionOptions so, std::string_view model_path)
  {
    if(!model_path.empty())
    {
      const auto folder = std::filesystem::path(model_path).parent_path().string();
      if(!folder.empty())
        // kOrtSessionOptionsModelExternalInitializersFileFolderPath, spelled out
        // so that older onnxruntime headers without the constant still build.
        so.AddConfigEntry(
            "session.model_external_initializers_file_folder_path", folder.c_str());
    }
    return so;
  }

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
           .shape = input_tensor_type.GetShape(),
           .elem_type = fromOrtElementType(input_tensor_type.GetElementType())});

      // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
      if (auto& tensor = spec.inputs.back();
          tensor.shape.size() == 4) // NCHW or NHCW
        if (tensor.shape[0] == -1)
        {
          tensor.shape[0] = 1;
          tensor.dynamic_batch = true;
        }

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
           .shape = output_tensor_type.GetShape(),
           .elem_type = fromOrtElementType(output_tensor_type.GetElementType())});

      spec.output_names.push_back(std::move(name));
    }
    spec.rebuildCharPointers();
    return spec;
  }

  ModelSpec m_spec; // built once in the ctor; returned by readModelSpec()
  // Lazily-read custom metadata (see metadata()); mutable: filled on first use.
  mutable std::once_flag m_metadata_once;
  mutable MetadataMap m_metadata;

public:
  void infer(
      const ModelSpec& spec,
      std::span<Ort::Value> input_tensors,
      std::span<Ort::Value> output_values)
  {
    ONNX_PROF_SCOPE(Infer);
    // Counts MUST come from the caller-provided spans, not the model's full
    // declared name lists: callers pass fixed-size stack arrays sized to the
    // outputs they actually read. Using the full declared count would make ORT
    // write/read past those arrays for a model with more I/O than expected.
    // A failure throws to the node, which reports it and skips the frame.
    session.Run(
        Ort::RunOptions{nullptr},
        spec.input_names_char.data(),
        input_tensors.data(),
        input_tensors.size(),
        spec.output_names_char.data(),
        output_values.data(),
        output_values.size());
  }
};

// The I/O of a model given as raw bytes (not a path).
inline ModelSpec readModelSpec(std::string_view model_bytes)
{
  OnnxRunContext ctx{model_bytes};
  return ctx.readModelSpec();
}

// A session for the model file at `path`, read on the calling thread (a
// worker: the nodes' file ports may have unmapped the file by then).
inline std::shared_ptr<OnnxRunContext>
loadRunContext(const std::string& path, Options o = {})
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if(!f)
    throw std::runtime_error("cannot read the file");
  std::string bytes((std::size_t)f.tellg(), '\0');
  f.seekg(0);
  if(!f.read(bytes.data(), (std::streamsize)bytes.size()))
    throw std::runtime_error("cannot read the file");
  return std::make_shared<OnnxRunContext>(bytes, path, std::move(o));
}
}
