#pragma once
// Frame profiler for the pose pipeline. Compiled out entirely unless
// ONNX_PROFILE is defined at build time (-DONNX_PROFILE), so it costs nothing
// in normal builds. When enabled it accumulates per-bucket wall time and prints
// an averaged breakdown to stderr every PRINT_EVERY frames.
//
// Buckets are sequential and non-overlapping within a frame, so their sum is
// <= Total; the remainder (decode + tracking + overhead) is printed as "other".

// The bucket enum is always defined (even with profiling off) so it can be
// passed as a function parameter (e.g. to tag a fused sampler as det vs crop);
// only the timing machinery below is gated on ONNX_PROFILE.
namespace Onnx::prof
{
enum Bucket
{
  ReadSpec = 0, // per-frame ORT model introspection
  WarpDet,      // fused sample+normalize of the WHOLE frame (detector/RTMO/YOLO)
  WarpCrop,     // fused sample+normalize of per-ROI crops (landmark / reid)
  Resize,       // LANCIR resize (unused by the pose path now)
  TensorBuild,  // legacy normalize pass (now folded into Warp; reads ~0)
  Infer,        // ORT session.Run
  Draw,         // fillCanvas (input->output copy) + ctx overlay rasterization
  Total,        // whole processed frame
  NBuckets
};

inline const char* name(int b)
{
  static const char* n[]
      = {"readspec", "warp:det", "warp:crop", "resize",
         "tensorbuild", "infer", "draw", "TOTAL"};
  return n[b];
}
} // namespace Onnx::prof

#if defined(ONNX_PROFILE)
  #include <array>
  #include <chrono>
  #include <cstdio>

namespace Onnx::prof
{
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
  // Scope on a runtime Bucket value (already-qualified expression).
  #define ONNX_PROF_SCOPE_VAR(bucketExpr) \
    ::Onnx::prof::Scope ONNX_PROF_CAT(_onnx_prof_, __LINE__)(bucketExpr)
  #define ONNX_PROF_FRAME() \
    ::Onnx::prof::FrameScope ONNX_PROF_CAT(_onnx_frame_, __LINE__)
#else
  #define ONNX_PROF_SCOPE(bucket) \
    do                            \
    {                             \
    } while(0)
  #define ONNX_PROF_SCOPE_VAR(bucketExpr) \
    do                                    \
    {                                     \
    } while(0)
  #define ONNX_PROF_FRAME() \
    do                      \
    {                       \
    } while(0)
#endif
