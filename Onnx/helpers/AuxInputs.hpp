#pragma once
// Values for the model inputs a node does not own ("aux" inputs): everything
// but its primary data input and its recurrent states.
//
// planAuxInputs() decides once per model, from the name, type and shape of each
// input, what it gets; fillAux() writes the values into storage typed as the
// model declares and returns the tensor. Rules, first match wins:
//   sr / sample_rate / sampling_rate / fs      -> the sample rate
//   *mask*, valid*                             -> ones, shaped like the primary
//   int *length* / *lens / *_len / seq_len     -> the primary input's length
//   bool                                       -> false
//   float latent, more than one value (opt-in) -> seeded N(0,1) noise
//   scalar ([], [1], [1,1])                    -> the next free Param
//   anything else                              -> zeros
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/TensorType.hpp>

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Onnx
{
enum class AuxFill : uint8_t
{
  Zeros,
  Ones,         // masks
  SampleRate,   // sr / sample_rate
  Param,        // a scalar driven by Param k
  SeededNoise,  // extra latents of a generator
  LinkedLength, // the length of the primary input
};

struct AuxPlan
{
  int index = -1; // model input index
  TensorElemType dt = TensorElemType::Float;
  std::vector<int64_t> shape; // declared (dynamic dims <= 0)
  AuxFill fill = AuxFill::Zeros;
  int param_index = -1; // AuxFill::Param
  bool link = false;    // dynamic dims take the primary input's (masks)
};

struct AuxOptions
{
  int nparams = 2;           // how many Params the node exposes
  bool seeded_noise = false; // generator nodes: extra float latents get noise
};

// What the node supplies at run time.
struct AuxHost
{
  std::span<const float> params;
  double sample_rate = 16000.;
  uint32_t seed = 0;
  float noise_scale = 1.f;
  std::span<const int64_t> primary_shape; // for masks and lengths
  int64_t primary_length = 0;             // for LinkedLength; 0 = primary's last dim
  int64_t batch = 1; // a dynamic leading dim of a rank >= 2 input
};

namespace aux_detail
{
inline std::string lower(std::string_view s)
{
  std::string r(s);
  for(auto& c : r)
    c = (char)std::tolower((unsigned char)c);
  return r;
}

// Whole-token match: "sr" matches "sr" or "input_sr", not "sru" or "rate".
inline bool hasToken(const std::string& name, std::string_view tok)
{
  for(std::size_t pos = name.find(tok); pos != std::string::npos;
      pos = name.find(tok, pos + 1))
  {
    const bool start = pos == 0 || !std::isalnum((unsigned char)name[pos - 1]);
    const std::size_t e = pos + tok.size();
    const bool end = e == name.size() || !std::isalnum((unsigned char)name[e]);
    if(start && end)
      return true;
  }
  return false;
}

inline bool isInt(TensorElemType t) noexcept
{
  switch(t)
  {
    case TensorElemType::Uint8:
    case TensorElemType::Int8:
    case TensorElemType::Uint16:
    case TensorElemType::Int16:
    case TensorElemType::Uint32:
    case TensorElemType::Int32:
    case TensorElemType::Uint64:
    case TensorElemType::Int64:
      return true;
    default:
      return false;
  }
}

inline bool isFloat(TensorElemType t) noexcept
{
  return t == TensorElemType::Float || t == TensorElemType::Float16
         || t == TensorElemType::BFloat16 || t == TensorElemType::Double;
}

// Store v as element i of a buffer of type t.
inline void store(uint8_t* buf, TensorElemType t, std::size_t i, double v) noexcept
{
  auto put = [&]<typename T>(T x) { std::memcpy(buf + i * sizeof(T), &x, sizeof(T)); };
  const double r = std::round(v);
  switch(t)
  {
    case TensorElemType::Float16:  put(floatToHalf((float)v)); break;
    case TensorElemType::BFloat16: put(floatToBFloat16((float)v)); break;
    case TensorElemType::Double:   put(v); break;
    case TensorElemType::Uint8:    put((uint8_t)std::clamp(r, 0., 255.)); break;
    case TensorElemType::Int8:     put((int8_t)std::clamp(r, -128., 127.)); break;
    case TensorElemType::Uint16:   put((uint16_t)std::clamp(r, 0., 65535.)); break;
    case TensorElemType::Int16:    put((int16_t)std::clamp(r, -32768., 32767.)); break;
    case TensorElemType::Uint32:   put((uint32_t)std::max(r, 0.)); break;
    case TensorElemType::Int32:    put((int32_t)r); break;
    case TensorElemType::Uint64:   put((uint64_t)std::max(r, 0.)); break;
    case TensorElemType::Int64:    put((int64_t)r); break;
    case TensorElemType::Bool:     put((uint8_t)(v > 0.5)); break;
    case TensorElemType::Float:
    default:                       put((float)v); break;
  }
}
} // namespace aux_detail

// An input tensor of element type `dt` over `values`: the floats themselves
// for float32 (ORT does not write to inputs), else converted into `staging`
// (ints rounded and saturated). Both must outlive the tensor. No allocation
// once staging has reached its size.
inline Ort::Value typedTensor(
    std::span<const float> values, const std::vector<int64_t>& shape,
    TensorElemType dt, std::vector<uint8_t>& staging)
{
  using namespace aux_detail;
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  if(dt == TensorElemType::Float || elemSize(dt) <= 0)
    return Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(values.data()), values.size(), shape.data(),
        shape.size());
  staging.resize(values.size() * (std::size_t)elemSize(dt));
  for(std::size_t i = 0; i < values.size(); ++i)
    store(staging.data(), dt, i, values[i]);
  return Ort::Value::CreateTensor(
      mem, staging.data(), staging.size(), shape.data(), shape.size(), toOrtElementType(dt));
}

// Plan every input not listed in `owned` (the node's primary and state inputs).
inline std::vector<AuxPlan> planAuxInputs(
    const std::vector<ModelSpec::Port>& inputs, std::span<const int> owned,
    const AuxOptions& opt = {})
{
  using namespace aux_detail;
  std::vector<AuxPlan> plans;
  int next_param = 0;
  for(int i = 0; i < (int)inputs.size(); ++i)
  {
    if(std::find(owned.begin(), owned.end(), i) != owned.end())
      continue;
    const auto& in = inputs[i];
    const std::string n = lower(in.name);
    AuxPlan p;
    p.index = i;
    p.dt = in.elem_type;
    p.shape = in.shape;

    const bool scalar
        = in.shape.size() <= 2
          && std::all_of(in.shape.begin(), in.shape.end(), [](int64_t d) { return d == 1; });
    const bool numeric = isInt(in.elem_type) || isFloat(in.elem_type);

    if(numeric
       && (hasToken(n, "sr") || hasToken(n, "sample_rate") || hasToken(n, "sampling_rate")
           || hasToken(n, "fs")))
      p.fill = AuxFill::SampleRate;
    else if(n.find("mask") != std::string::npos || n.starts_with("valid"))
    {
      p.fill = AuxFill::Ones;
      p.link = true;
    }
    else if(
        isInt(in.elem_type)
        && (n.find("length") != std::string::npos || n.ends_with("lens")
            || n.ends_with("_len") || hasToken(n, "seq_len")))
      p.fill = AuxFill::LinkedLength;
    else if(in.elem_type == TensorElemType::Bool)
      p.fill = AuxFill::Zeros;
    else if(opt.seeded_noise && isFloat(in.elem_type) && flatSize(in.shape) > 1)
      p.fill = AuxFill::SeededNoise;
    else if(scalar && numeric && next_param < opt.nparams)
    {
      p.fill = AuxFill::Param;
      p.param_index = next_param++;
    }
    plans.push_back(std::move(p));
  }
  return plans;
}

// The concrete shape an aux input is fed at, into `s`: its declared rank, with
// the dynamic dims taken from the primary input when linked (a mask follows the
// token sequence), else 1.
inline void auxShape(const AuxPlan& p, const AuxHost& h, std::vector<int64_t>& s)
{
  s.assign(p.shape.begin(), p.shape.end());
  const bool link = p.link && h.primary_shape.size() == s.size();
  for(std::size_t k = 0; k < s.size(); ++k)
    if(s[k] <= 0)
      s[k] = link && h.primary_shape[k] > 0 ? h.primary_shape[k] : 1;
  if(h.batch > 1 && s.size() >= 2 && p.shape[0] <= 0)
    s[0] = h.batch;
}

// Write the plan's values into `storage` (typed as declared) and return a tensor
// over it; `shape` receives the concrete shape. Both must outlive the tensor.
// No allocation once storage has reached its size.
inline Ort::Value fillAux(
    const AuxPlan& p, const AuxHost& h, std::vector<uint8_t>& storage,
    std::vector<int64_t>& shape)
{
  using namespace aux_detail;
  auxShape(p, h, shape);
  const int64_t n = flatSize(shape);
  const int es = std::max(elemSize(p.dt), 1);
  storage.resize((std::size_t)n * es);

  double v = 0.;
  switch(p.fill)
  {
    case AuxFill::Ones:
      v = 1.;
      break;
    case AuxFill::SampleRate:
      v = h.sample_rate;
      break;
    case AuxFill::Param:
      v = p.param_index >= 0 && p.param_index < (int)h.params.size()
              ? h.params[p.param_index]
              : 0.;
      break;
    case AuxFill::LinkedLength:
      v = h.primary_length > 0
              ? (double)h.primary_length
              : (h.primary_shape.empty() ? 1. : (double)h.primary_shape.back());
      break;
    default:
      break;
  }

  if(p.fill == AuxFill::SeededNoise)
  {
    std::mt19937 rng(h.seed + (uint32_t)p.index);
    std::normal_distribution<float> nd(0.f, 1.f);
    for(int64_t i = 0; i < n; ++i)
      store(storage.data(), p.dt, (std::size_t)i, nd(rng) * h.noise_scale);
  }
  else
  {
    for(int64_t i = 0; i < n; ++i)
      store(storage.data(), p.dt, (std::size_t)i, v);
  }

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return Ort::Value::CreateTensor(
      mem, storage.data(), storage.size(), shape.data(), shape.size(), toOrtElementType(p.dt));
}
} // namespace Onnx
