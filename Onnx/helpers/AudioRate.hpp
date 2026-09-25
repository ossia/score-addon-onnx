#pragma once
// The sample rate an audio model runs at. Almost no export declares it, and
// its port names rarely say what it is (Demucs: mix/x, DTLN: input_2, ...), but
// its file name usually does. In order: the "sample_rate" / "sr" /
// "sampling_rate" metadata property, a family name in the port names, the same
// in the file name, else the caller's fallback.
#include <Onnx/helpers/OnnxContext.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace Onnx
{
namespace audio_rate_detail
{
inline std::string lower(std::string_view s)
{
  std::string r(s);
  for(auto& c : r)
    c = (char)std::tolower((unsigned char)c);
  return r;
}

// A family name in `text`, lower case; 0 when none matches.
inline double rateOfName(const std::string& text)
{
  struct Family
  {
    const char* key;
    double rate;
  };
  static constexpr Family families[]{
      {"demucs", 44100.}, {"spleeter", 44100.},
      {"whisper", 16000.}, {"crepe", 16000.},  {"silero", 16000.},
      {"dtln", 16000.},    {"yamnet", 16000.}, {"wav2vec", 16000.},
      {"hubert", 16000.},  {"wavlm", 16000.},
      {"hifigan", 22050.}, {"tacotron", 22050.}, {"waveglow", 22050.},
      {"encodec", 24000.}, {"vocos", 24000.},
      {"clap", 48000.},    {"rave", 48000.},   {"deepfilter", 48000.},
      {"dfnet", 48000.},
  };
  for(const auto& f : families)
    if(text.find(f.key) != std::string::npos)
      return f.rate;
  return 0.;
}
}

inline double
audioModelRate(OnnxRunContext& ctx, std::string_view model_path, double fallback)
{
  using namespace audio_rate_detail;
  try
  {
    Ort::AllocatorWithDefaultOptions alloc;
    auto meta = ctx.session.GetModelMetadata();
    for(const char* key : {"sample_rate", "sampling_rate", "sr"})
      if(auto v = meta.LookupCustomMetadataMapAllocated(key, alloc))
        if(const double r = std::atof(v.get()); r >= 1000. && r <= 384000.)
          return r;
  }
  catch(...)
  {
  }

  const auto& spec = ctx.readModelSpec();
  for(const auto* ports : {&spec.inputs, &spec.outputs})
    for(const auto& p : *ports)
      if(const double r = rateOfName(lower(p.name)); r > 0.)
        return r;

  const auto slash = model_path.find_last_of("/\\");
  const auto file = model_path.substr(slash == std::string_view::npos ? 0 : slash + 1);
  if(const double r = rateOfName(lower(file)); r > 0.)
    return r;
  return fallback;
}
}
