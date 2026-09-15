#pragma once
// Vendored, self-contained copies of the ossia math primitives the pose
// tracker uses (from libossia: ossia/math/filters.hpp and
// ossia/math/tracking.hpp). Same arrangement as compat/safe_math.hpp beside
// this file.
//
// libossia is only on the include path in a score build. The standalone
// back-ends - max, touchdesigner, godot, python, and the SDK-free `dump`
// target the portability smoke builds - never fetch it, and they compile
// PoseDetector.hpp just as the score build does. Including ossia headers
// unconditionally would break every one of them, so this defers to the real
// header when it is reachable and supplies an equivalent otherwise.
//
// Defining these in namespace ossia is only safe because the two are mutually
// exclusive: when libossia is present we include it and define nothing, so
// there is never a second definition of the same inline entity.
//
// Keep API- and behaviour-compatible with libossia. Only what the tracker
// actually calls is mirrored here; the canonical versions carry the full
// documentation and the rest of the family.
#if __has_include(<ossia/math/filters.hpp>)
#include <ossia/math/filters.hpp>
#include <ossia/math/tracking.hpp>
#else
#include <cmath>

namespace ossia
{

inline constexpr double two_pi = 6.283185307179586476925286766559005768;

//! First-order low-pass. The coefficient is passed per-sample so a varying dt
//! needs no internal recomputation.
template <typename T = float>
struct one_pole_filter
{
  using value_type = T;

  T y{};
  //! False until the first sample: it is passed through rather than smoothed
  //! up from an arbitrary zero.
  bool primed{};

  inline void reset() noexcept
  {
    y = T{};
    primed = false;
  }

  inline void reset(T x) noexcept
  {
    y = x;
    primed = true;
  }

  inline void assign_parameters(const one_pole_filter&) noexcept { }

  [[nodiscard]] inline T operator()(T x, T alpha) noexcept
  {
    if(!primed) [[unlikely]]
    {
      primed = true;
      return y = x;
    }
    return y += alpha * (x - y);
  }
};

//! One-euro filter (Casiez, Roussel, Vogel, CHI 2012): a low-pass whose cutoff
//! rises with the speed of the signal, so it is smooth at rest and responsive
//! in motion. The derivative is taken against the previous *raw* input, as in
//! the reference implementation.
template <typename T = float>
struct one_euro_filter
{
  using value_type = T;

  T min_cutoff{1};
  T beta{0};
  T d_cutoff{1};

  one_pole_filter<T> x_filter{};
  one_pole_filter<T> dx_filter{};
  T x_prev{};

  inline void reset() noexcept
  {
    x_filter.reset();
    dx_filter.reset();
    x_prev = T{};
  }

  inline void assign_parameters(const one_euro_filter& p) noexcept
  {
    min_cutoff = p.min_cutoff;
    beta = p.beta;
    d_cutoff = p.d_cutoff;
  }

  //! A dt <= 0 returns the previous output rather than dividing by zero, so a
  //! duplicated timestamp cannot poison the state.
  [[nodiscard]] inline T operator()(T x, T dt) noexcept
  {
    if(dt <= T(0)) [[unlikely]]
      return x_filter.primed ? x_filter.y : x;

    const T k = T(two_pi) * dt;

    const T dx = x_filter.primed ? (x - x_prev) / dt : T(0);
    x_prev = x;

    const T kd = k * d_cutoff;
    const T edx = dx_filter(dx, kd / (kd + T(1)));

    const T cutoff = min_cutoff + beta * std::abs(edx);
    const T kc = k * cutoff;
    return x_filter(x, kc / (kc + T(1)));
  }
};

//! Bring @p a within half a turn of @p reference, so a wrapping angle can be
//! filtered without lurching by a full turn at the discontinuity.
template <typename T>
[[nodiscard]] inline T unwrap_angle(T a, T reference) noexcept
{
  return reference + T(std::remainder(a - reference, T(two_pi)));
}

//! Scalar position-velocity Kalman filter, taking an explicit dt.
//!
//! A constant-velocity model with independent axes and diagonal noise
//! decouples exactly into one of these per axis, so the tracker composes N of
//! them instead of carrying a 2Nx2N matrix.
struct kalman_pv_filter
{
  struct process_noise
  {
    float q00, q01, q11;
  };

  //! Discretised continuous-white-noise-acceleration process noise:
  //! Q = sigma_a^2 * [[dt^3/3, dt^2/2], [dt^2/2, dt]].
  [[nodiscard]] static inline process_noise cwna(float sigma_a, float dt) noexcept
  {
    const float q = sigma_a * sigma_a;
    const float dt2 = dt * dt;
    return {q * dt2 * dt / 3.f, q * dt2 / 2.f, q * dt};
  }

  //! Ad-hoc diagonal process noise, dt-scaled: reproduces ByteTrack's
  //! per-frame standard deviations at the reference rate while still growing
  //! with elapsed time.
  [[nodiscard]] static inline process_noise
  diagonal(float std_p, float std_v, float time_ratio) noexcept
  {
    return {std_p * std_p * time_ratio, 0.f, std_v * std_v * time_ratio};
  }

  //! Position and velocity, in units and units/s.
  float p{}, v{};
  //! Symmetric 2x2 covariance: [[P00, P01], [P01, P11]].
  float P00{}, P01{}, P11{};

  inline void initiate(float p0, float var_p, float var_v) noexcept
  {
    p = p0;
    v = 0.f;
    P00 = var_p;
    P01 = 0.f;
    P11 = var_v;
  }

  inline void predict(float dt, const process_noise& q) noexcept
  {
    p += v * dt;
    P00 += dt * (2.f * P01 + dt * P11) + q.q00;
    P01 += dt * P11 + q.q01;
    P11 += q.q11;
  }

  inline void update(float z, float r) noexcept
  {
    const float S = P00 + r;
    if(!(S > 0.f)) [[unlikely]]
      return; // degenerate (or NaN) innovation covariance: skip the update
    const float K0 = P00 / S;
    const float K1 = P01 / S;
    const float y = z - p;
    p += K0 * y;
    v += K1 * y;
    P01 -= K0 * P01;
    P11 -= K1 * P01; // uses the already-updated P01: exact (I-KH)P
    P00 -= K0 * P00;
  }

  //! Squared Mahalanobis distance in measurement space, chi-square with 1 DOF;
  //! sum over axes for N.
  [[nodiscard]] inline float gating_distance2(float z, float r) const noexcept
  {
    const float S = P00 + r;
    const float y = z - p;
    return S > 0.f ? (y * y) / S : 0.f;
  }
};

}
#endif
