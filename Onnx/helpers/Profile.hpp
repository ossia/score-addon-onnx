#pragma once
// Frame profiler for the pose pipeline. Compiled out entirely unless
// ONNX_PROFILE is defined at build time (-DONNX_PROFILE), so it costs nothing
// in normal builds. When enabled it accumulates per-bucket wall time and prints
// an averaged breakdown to stderr every PRINT_EVERY frames.
//
// Buckets are sequential and non-overlapping within a frame, so their sum is
// <= Total; the remainder (decode + tracking + drawing + overhead) is printed
// as "other".

#if defined(ONNX_PROFILE)
  #include <array>
  #include <chrono>
  #include <cstdio>

namespace Onnx::prof
{
enum Bucket
{
  ReadSpec = 0, // per-frame ORT model introspection
  Warp,         // warpAffine (ROI crops, cover-resize)
  Resize,       // LANCIR resize (letterbox detector input)
  TensorBuild,  // RGBA -> normalized planar/interleaved float
  Infer,        // ORT session.Run
  Total,        // whole processed frame
  NBuckets
};

inline const char* name(int b)
{
  static const char* n[]
      = {"readspec", "warp", "resize", "tensorbuild", "infer", "TOTAL"};
  return n[b];
}

struct Accum
{
  std::array<double, NBuckets> us{};
  long frames{};
};
inline Accum& accum()
{
  static thread_local Accum a;
  return a;
}

inline void maybePrint()
{
  constexpr long PRINT_EVERY = 60;
  auto& a = accum();
  if(a.frames < PRINT_EVERY)
    return;
  const double f = a.frames;
  double accounted = 0;
  for(int i = 0; i < Total; ++i)
    accounted += a.us[i];
  std::fprintf(stderr, "[onnx-prof] %ld frames, avg us/frame:", a.frames);
  for(int i = 0; i < NBuckets; ++i)
    std::fprintf(stderr, " %s=%.1f", name(i), a.us[i] / f);
  std::fprintf(
      stderr, " other=%.1f  (~%.0f fps)\n", (a.us[Total] - accounted) / f,
      a.us[Total] > 0 ? 1e6 / (a.us[Total] / f) : 0.0);
  a = Accum{};
}

using clock = std::chrono::steady_clock;

struct Scope
{
  Bucket b;
  clock::time_point t0;
  explicit Scope(Bucket b_) : b(b_), t0(clock::now()) { }
  ~Scope()
  {
    accum().us[b]
        += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
  }
};

// Counts a processed frame on destruction (handles early-return / throw paths).
struct FrameScope
{
  clock::time_point t0;
  FrameScope() : t0(clock::now()) { }
  ~FrameScope()
  {
    auto& a = accum();
    a.us[Total]
        += std::chrono::duration<double, std::micro>(clock::now() - t0).count();
    ++a.frames;
    maybePrint();
  }
};
} // namespace Onnx::prof

  #define ONNX_PROF_CAT2(a, b) a##b
  #define ONNX_PROF_CAT(a, b) ONNX_PROF_CAT2(a, b)
  #define ONNX_PROF_SCOPE(bucket) \
    ::Onnx::prof::Scope ONNX_PROF_CAT(_onnx_prof_, __LINE__)(::Onnx::prof::bucket)
  #define ONNX_PROF_FRAME() \
    ::Onnx::prof::FrameScope ONNX_PROF_CAT(_onnx_frame_, __LINE__)
#else
  #define ONNX_PROF_SCOPE(bucket) \
    do                            \
    {                             \
    } while(0)
  #define ONNX_PROF_FRAME() \
    do                      \
    {                       \
    } while(0)
#endif
