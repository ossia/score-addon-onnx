#pragma once
#include <Onnx/helpers/OneEuro.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Multi-instance pose tracker — assigns a persistent track_id to each detected
// person/hand/face across frames, ByteTrack-style:
//   * per-track constant-velocity Kalman (the ByteTrack 8-dim xyah filter),
//   * two-stage association (high-score IoU+OKS, then low-score IoU recovery),
//   * OKS pose-similarity cue (disambiguates crossings where box-IoU is
//     degenerate; works for box-less / bottom-up models too),
//   * OC-SORT observation-centric touches: velocity-direction consistency in the
//     cost, and velocity re-seeding when a track is re-found after a gap,
//   * One-Euro smoothing owned PER TRACK (no cross-identity bleed on crossings).
//
// Validated standalone in libreonnx/posetrack_example.cpp before integration.
// ossia-free; depends only on Eigen + OneEuro.hpp. Operates in whatever
// coordinate space the caller uses consistently (PoseDetector feeds normalized
// [0,1] image coordinates).
namespace Onnx
{
namespace Track
{

// COCO-17 OKS sigmas. For any other keypoint layout we fall back to a flat
// sigma (tracking is layout-agnostic; only the normalization constant changes).
inline float keypointSigma(int idx, int n)
{
  static const float coco17[17]
      = {.026f, .025f, .025f, .035f, .035f, .079f, .079f, .072f, .072f,
         .062f, .062f, .107f, .107f, .087f, .087f, .089f, .089f};
  if(n == 17 && idx >= 0 && idx < 17)
    return coco17[idx];
  return 0.05f;
}

struct Box
{
  float cx, cy, w, h;
};

struct Keypoint
{
  float x, y, z, score;
};

// One detection handed to the tracker for a frame.
struct Detection
{
  Box box;
  float score;
  std::vector<Keypoint> keypoints;  // may be empty -> OKS skipped for this det
  std::vector<float> embedding;     // L2-normalized ReID feature, or empty
};

// Association motion gate: how to reject an implausible single-frame jump.
enum class MotionGate : std::uint8_t
{
  None,        // no motion gate (baseline; IoU/OKS/ReID gating only)
  MaxSpeed,    // analytic: center move <= max_speed * box_size * (1 + frames_lost)
  Mahalanobis, // Kalman gating distance vs chi-square threshold (DeepSORT)
};

struct Config
{
  float track_high = 0.5f;  // >= : stage-1 (high-score) detections
  float track_low = 0.1f;   // [low,high): stage-2 recovery detections
  float new_track = 0.6f;   // birth a track only above this score
  float gate_iou = 0.10f;   // association gate (box)
  float gate_oks = 0.30f;   // association gate (pose)
  float w_iou = 0.5f;       // cost weights (w_iou + w_oks should ~= 1)
  float w_oks = 0.5f;
  float w_dir = 0.2f;       // OC-SORT velocity-direction consistency weight
  int min_hits = 3;         // tentative -> confirmed
  int max_age = 30;         // frames a lost track is kept alive (~ fps)
  bool use_oks = true;
  bool use_dir = true;      // OC-SORT OCM term
  bool smooth = true;
  float smooth_min_cutoff = 1.0f;
  float smooth_beta = 0.3f;
  // Appearance ReID (StrongSORT / Deep-OC-SORT style)
  bool use_reid = false;    // blend appearance cosine into the cost
  float w_emb = 0.25f;      // appearance weight
  float emb_gate = 0.25f;   // min cosine to allow an appearance-only re-acquire
  float emb_ema = 0.9f;     // per-track embedding EMA factor

  // --- Plausibility gates (composable; every default reproduces the baseline so
  //     each can be A/B-tested independently). See cost() and canBirth(). ---
  // Motion gate: reject an association that is an implausible jump from the
  // track's predicted state.
  MotionGate motion_gate = MotionGate::None;
  float max_speed = 2.0f;        // MaxSpeed budget, in units of max(box_w,box_h)
                                 // per frame, scaled by (1 + frames_lost).
  float maha_thresh = 9.4877f;   // Mahalanobis chi-square 0.95 gate, 4 DOF (xyah).
  bool gate_relax_on_reid = true;// let an appearance match bypass the motion/size
                                 // gate (deliberate re-acquire after occlusion).
  // Size gate: reject if det/track area ratio exceeds max_size_ratio.
  bool gate_size = false;
  float max_size_ratio = 2.0f;
  // Birth gates: reject spurious new tracks (e.g. a raised limb read as a person).
  bool birth_gate = false;
  float min_birth_area = 0.0f;     // area floor for a new track (box w*h units).
  float birth_contain_frac = 0.7f; // reject birth if >= this fraction of its box
                                   // lies inside an existing CONFIRMED track.
  // Strict confirmation (DeepSORT n_init): delete a tentative track the first
  // frame it fails to match, so a one-frame detection never persists.
  bool strict_confirm = false;
};

// ByteTrack 8-dim Kalman filter (state [x,y,a,h, vx,vy,va,vh], a = w/h).
class KalmanFilter
{
public:
  using State = Eigen::Matrix<float, 8, 1>;
  using Cov = Eigen::Matrix<float, 8, 8>;
  using Meas = Eigen::Vector4f;

  KalmanFilter()
  {
    _F.setIdentity();
    for(int i = 0; i < 4; ++i)
      _F(i, i + 4) = 1.f; // dt = 1
    _H.setZero();
    for(int i = 0; i < 4; ++i)
      _H(i, i) = 1.f;
  }

  void initiate(const Meas& m, State& mean, Cov& cov) const
  {
    mean.setZero();
    mean.head<4>() = m;
    const float h = m(3);
    State s;
    s << 2 * _spw * h, 2 * _spw * h, 1e-2f, 2 * _spw * h, 10 * _svw * h,
        10 * _svw * h, 1e-5f, 10 * _svw * h;
    cov = (s.array() * s.array()).matrix().asDiagonal();
  }

  void predict(State& mean, Cov& cov) const
  {
    const float h = std::max(1e-3f, mean(3)); // floor: a 0/neg/NaN height would
                                              // make R/Q singular -> inverse NaN
    State s;
    s << _spw * h, _spw * h, 1e-2f, _spw * h, _svw * h, _svw * h, 1e-5f,
        _svw * h;
    const Cov Q = (s.array() * s.array()).matrix().asDiagonal();
    mean = _F * mean;
    cov = _F * cov * _F.transpose() + Q;
  }

  void update(State& mean, Cov& cov, const Meas& m) const
  {
    const float h = std::max(1e-3f, mean(3)); // floor: a 0/neg/NaN height would
                                              // make R/Q singular -> inverse NaN
    Meas s;
    s << _spw * h, _spw * h, 1e-1f, _spw * h;
    const Eigen::Matrix4f R = (s.array() * s.array()).matrix().asDiagonal();
    const Meas pm = _H * mean;
    const Eigen::Matrix4f pc = _H * cov * _H.transpose() + R;
    const Eigen::Matrix<float, 8, 4> K
        = cov * _H.transpose() * pc.inverse();
    mean = mean + K * (m - pm);
    cov = (Cov::Identity() - K * _H) * cov;
  }

  // Squared Mahalanobis distance of measurement m from the predicted state, in
  // measurement space (4-DOF xyah). Pass the POST-predict mean/cov (as held by a
  // track during association). Covariance growth over missed frames makes this
  // gate naturally permit larger jumps after longer occlusions.
  float gatingDistance(const State& mean, const Cov& cov, const Meas& m) const
  {
    const float h = std::max(1e-3f, mean(3)); // floor: a 0/neg/NaN height would
                                              // make R/Q singular -> inverse NaN
    Meas s;
    s << _spw * h, _spw * h, 1e-1f, _spw * h;
    const Eigen::Matrix4f R = (s.array() * s.array()).matrix().asDiagonal();
    const Meas pm = _H * mean;
    const Eigen::Matrix4f pc = _H * cov * _H.transpose() + R;
    const Meas d = m - pm;
    return (d.transpose() * pc.inverse() * d).value(); // 1x1 -> scalar
  }

private:
  float _spw = 1.f / 20.f;  // std_weight_position
  float _svw = 1.f / 160.f; // std_weight_velocity
  Eigen::Matrix<float, 8, 8> _F;
  Eigen::Matrix<float, 4, 8> _H;
};

inline Eigen::Vector4f boxToXyah(const Box& b)
{
  return {b.cx, b.cy, (b.h > 0 ? b.w / b.h : 0.f), b.h};
}
inline Box xyahToBox(const Eigen::Vector4f& m)
{
  return {m(0), m(1), m(2) * m(3), m(3)};
}

inline float iou(const Box& a, const Box& b)
{
  const float ax1 = a.cx - a.w * .5f, ay1 = a.cy - a.h * .5f;
  const float ax2 = a.cx + a.w * .5f, ay2 = a.cy + a.h * .5f;
  const float bx1 = b.cx - b.w * .5f, by1 = b.cy - b.h * .5f;
  const float bx2 = b.cx + b.w * .5f, by2 = b.cy + b.h * .5f;
  const float ix = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
  const float iy = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
  const float inter = ix * iy;
  const float uni = a.w * a.h + b.w * b.h - inter;
  return uni > 0 ? inter / uni : 0.f;
}

// OKS between two keypoint sets given the object scale (box area).
inline float
oks(const std::vector<Keypoint>& a, const std::vector<Keypoint>& b,
    const Box& box)
{
  if(a.empty() || a.size() != b.size())
    return 0.f;
  const int n = (int)a.size();
  const float s2 = std::max(1e-9f, box.w * box.h);
  float sum = 0.f;
  int cnt = 0;
  for(int i = 0; i < n; ++i)
  {
    if(a[i].score <= 0.1f || b[i].score <= 0.1f)
      continue;
    const float dx = a[i].x - b[i].x, dy = a[i].y - b[i].y;
    const float k = keypointSigma(i, n);
    sum += std::exp(-(dx * dx + dy * dy) / (2.f * s2 * k * k));
    ++cnt;
  }
  return cnt ? sum / cnt : 0.f;
}

// Cosine of two L2-normalized vectors == dot product. 0 if either is empty or
// the dims differ (e.g. a track that never got an embedding).
inline float cosine(const std::vector<float>& a, const std::vector<float>& b)
{
  if(a.empty() || a.size() != b.size())
    return 0.f;
  float d = 0.f;
  for(size_t i = 0; i < a.size(); ++i)
    d += a[i] * b[i];
  return d;
}

struct Tracklet
{
  int id = -1;
  KalmanFilter::State mean;
  KalmanFilter::Cov cov;

  std::vector<Keypoint> kpts;        // last observed, motion-compensated
  std::vector<Keypoint> kpts_smooth; // One-Euro output (owned by this id)
  PoseSmoother smoother;             // 2 filters per keypoint (x,y)
  std::vector<float> embedding;      // EMA of L2-normalized ReID features

  float score = 0.f;
  int hits = 0;
  int age = 0;
  int time_since_update = 0;
  bool confirmed = false;

  // OC-SORT direction history
  float prev_cx = 0, prev_cy = 0;
  bool have_dir = false;

  Box box() const { return xyahToBox(mean.head<4>()); }
};

class PoseTracker
{
public:
  void configure(const Config& c) { _cfg = c; }
  const Config& config() const { return _cfg; }
  const std::vector<Tracklet>& tracks() const { return _tracks; }
  void reset()
  {
    _tracks.clear();
    _next_id = 1;
  }

  // Update with this frame's detections. Returns, per input detection index,
  // the assigned track id (-1 if the detection started a still-tentative track
  // or was dropped). After this call, tracks() holds the live tracklets with
  // smoothed keypoints.
  std::vector<int> update(const std::vector<Detection>& dets)
  {
    std::vector<int> out(dets.size(), -1);

    // 1) predict all tracks forward; motion-compensate their keypoints so OKS
    //    compares pose *shape*, not stale position (the key tuning lesson).
    for(auto& t : _tracks)
    {
      const float pcx = t.mean(0), pcy = t.mean(1);
      _kf.predict(t.mean, t.cov);
      const float dx = t.mean(0) - pcx, dy = t.mean(1) - pcy;
      for(auto& k : t.kpts)
      {
        k.x += dx;
        k.y += dy;
      }
      ++t.age;
      ++t.time_since_update;
    }

    // 2) split detections by score
    std::vector<int> high, low;
    for(int i = 0; i < (int)dets.size(); ++i)
      (dets[i].score >= _cfg.track_high ? high : low).push_back(i);

    std::vector<int> t_match(_tracks.size(), -1);
    std::vector<char> d_used(dets.size(), 0);

    // 3) stage 1: all tracks vs high-score detections (IoU + OKS + dir)
    std::vector<int> all_t(_tracks.size());
    for(int i = 0; i < (int)_tracks.size(); ++i)
      all_t[i] = i;
    associate(all_t, high, dets, t_match, d_used, /*full_cost=*/true);

    // 4) stage 2: unmatched tracks vs low-score detections (IoU only — recovers
    //    confidence dips / partial occlusion, the ByteTrack trick)
    std::vector<int> rem_t;
    for(int i = 0; i < (int)_tracks.size(); ++i)
      if(t_match[i] < 0)
        rem_t.push_back(i);
    associate(rem_t, low, dets, t_match, d_used, /*full_cost=*/false);

    // 5) update matched tracks
    for(int i = 0; i < (int)_tracks.size(); ++i)
    {
      if(t_match[i] < 0)
        continue;
      const int di = t_match[i];
      updateTrack(_tracks[i], dets[di]);
      if(_tracks[i].confirmed)
        out[di] = _tracks[i].id;
    }

    // 6) birth: unmatched high-score detections start tentative tracks
    for(int di : high)
    {
      if(d_used[di] || dets[di].score < _cfg.new_track)
        continue;
      if(!canBirth(dets[di]))
        continue;
      birth(dets[di]);
    }

    // 7) death: drop tracks lost longer than max_age. With strict confirmation,
    //    also drop a tentative track the first frame it fails to match (so a
    //    one-frame spurious detection never persists or is emitted).
    _tracks.erase(
        std::remove_if(
            _tracks.begin(), _tracks.end(),
            [&](const Tracklet& t) {
              if(t.time_since_update > _cfg.max_age)
                return true;
              if(_cfg.strict_confirm && !t.confirmed && t.time_since_update > 0)
                return true;
              return false;
            }),
        _tracks.end());

    return out;
  }

private:
  // Fraction of box a's area that lies inside box b (1 = fully contained).
  static float containedFrac(const Box& a, const Box& b)
  {
    const float ax1 = a.cx - a.w * .5f, ay1 = a.cy - a.h * .5f;
    const float ax2 = a.cx + a.w * .5f, ay2 = a.cy + a.h * .5f;
    const float bx1 = b.cx - b.w * .5f, by1 = b.cy - b.h * .5f;
    const float bx2 = b.cx + b.w * .5f, by2 = b.cy + b.h * .5f;
    const float ix = std::max(0.f, std::min(ax2, bx2) - std::max(ax1, bx1));
    const float iy = std::max(0.f, std::min(ay2, by2) - std::max(ay1, by1));
    const float aa = a.w * a.h;
    return aa > 0.f ? (ix * iy) / aa : 0.f;
  }

  // Birth plausibility: reject a new track that is too small, or whose box sits
  // mostly inside an existing confirmed track (the limb-on-torso signature).
  bool canBirth(const Detection& d) const
  {
    if(!_cfg.birth_gate)
      return true;
    if(_cfg.min_birth_area > 0.f && d.box.w * d.box.h < _cfg.min_birth_area)
      return false;
    if(_cfg.birth_contain_frac > 0.f)
      for(const auto& t : _tracks)
        if(t.confirmed
           && containedFrac(d.box, t.box()) >= _cfg.birth_contain_frac)
          return false;
    return true;
  }

  float cost(const Tracklet& t, const Detection& d, bool full) const
  {
    const Box tb = t.box();
    const float i = iou(tb, d.box);
    if(!full)
      return (i >= _cfg.gate_iou) ? (1.f - i) : kReject;

    float o = 0.f;
    const bool have_oks
        = _cfg.use_oks && !t.kpts.empty() && t.kpts.size() == d.keypoints.size();
    if(have_oks)
      o = oks(t.kpts, d.keypoints, tb);

    const bool have_emb = _cfg.use_reid && !t.embedding.empty()
                          && t.embedding.size() == d.embedding.size();
    const float emb_cos = have_emb ? cosine(t.embedding, d.embedding) : 0.f;

    // gate: need box overlap OR pose support OR appearance support (the last
    // lets a track moved far / briefly occluded re-acquire by looks).
    const bool box_ok = i >= _cfg.gate_iou;
    const bool pose_ok = have_oks && o >= _cfg.gate_oks;
    const bool emb_ok = have_emb && emb_cos >= _cfg.emb_gate;
    if(!box_ok && !pose_ok && !emb_ok)
      return kReject;

    // Plausibility gates. An appearance match can deliberately bypass them
    // (re-acquire after occlusion) when gate_relax_on_reid is set.
    const bool reid_relax = _cfg.gate_relax_on_reid && emb_ok;
    if(!reid_relax)
    {
      if(_cfg.motion_gate == MotionGate::MaxSpeed)
      {
        const float disp = std::hypot(d.box.cx - tb.cx, d.box.cy - tb.cy);
        const float budget = _cfg.max_speed * std::max(tb.w, tb.h)
                             * float(1 + t.time_since_update);
        if(disp > budget)
          return kReject;
      }
      else if(_cfg.motion_gate == MotionGate::Mahalanobis)
      {
        if(_kf.gatingDistance(t.mean, t.cov, boxToXyah(d.box)) > _cfg.maha_thresh)
          return kReject;
      }
      if(_cfg.gate_size)
      {
        const float at = tb.w * tb.h, ad = d.box.w * d.box.h;
        if(at > 0.f && ad > 0.f
           && (ad > at ? ad / at : at / ad) > _cfg.max_size_ratio)
          return kReject;
      }
    }

    float c = _cfg.w_iou * (1.f - i);
    if(have_oks)
      c += _cfg.w_oks * (1.f - o);
    else
      c += _cfg.w_oks * (1.f - i); // no pose -> lean on box
    if(have_emb)
      c += _cfg.w_emb * (1.f - emb_cos);

    // OC-SORT velocity-direction consistency (cheap, helps non-linear motion)
    if(_cfg.use_dir && t.have_dir)
    {
      const float vx = t.mean(0) - t.prev_cx, vy = t.mean(1) - t.prev_cy;
      const float tx = d.box.cx - t.prev_cx, ty = d.box.cy - t.prev_cy;
      const float nv = std::hypot(vx, vy), nt = std::hypot(tx, ty);
      if(nv > 1e-4f && nt > 1e-4f)
      {
        float cosang = (vx * tx + vy * ty) / (nv * nt);
        cosang = std::max(-1.f, std::min(1.f, cosang));
        const float ang = std::acos(cosang);        // [0,pi]
        c += _cfg.w_dir * (ang / 3.14159265f);       // [0,1]
      }
    }
    return c;
  }

  // Greedy bipartite matching over a track list and detection list.
  void associate(
      const std::vector<int>& tk, const std::vector<int>& dk,
      const std::vector<Detection>& dets, std::vector<int>& t_match,
      std::vector<char>& d_used, bool full_cost)
  {
    struct Pair
    {
      float c;
      int ti, di;
    };
    std::vector<Pair> pairs;
    for(int ti : tk)
    {
      if(t_match[ti] >= 0)
        continue;
      for(int di : dk)
      {
        if(d_used[di])
          continue;
        const float c = cost(_tracks[ti], dets[di], full_cost);
        if(c < kReject)
          pairs.push_back({c, ti, di});
      }
    }
    std::stable_sort(
        pairs.begin(), pairs.end(),
        [](const Pair& a, const Pair& b) { return a.c < b.c; });
    for(const auto& p : pairs)
    {
      if(t_match[p.ti] >= 0 || d_used[p.di])
        continue;
      t_match[p.ti] = p.di;
      d_used[p.di] = 1;
    }
  }

  void updateTrack(Tracklet& t, const Detection& d)
  {
    // OC-SORT velocity re-seed (ORU spirit): if the track was lost for a gap,
    // trust the new observation's implied velocity over the drifted prediction.
    const int gap = t.time_since_update;
    t.prev_cx = t.mean(0);
    t.prev_cy = t.mean(1);

    _kf.update(t.mean, t.cov, boxToXyah(d.box));

    if(gap > 1)
    {
      const float vx = (d.box.cx - t.prev_cx) / (float)gap;
      const float vy = (d.box.cy - t.prev_cy) / (float)gap;
      t.mean(4) = vx;
      t.mean(5) = vy;
    }
    t.have_dir = true;

    t.score = d.score;
    t.kpts = d.keypoints;

    // per-track One-Euro on the observed keypoints
    if(_cfg.smooth && !d.keypoints.empty())
    {
      t.smoother.configure(_cfg.smooth_min_cutoff, _cfg.smooth_beta);
      t.smoother.ensure(d.keypoints.size() * 2);
      t.kpts_smooth.resize(d.keypoints.size());
      for(size_t i = 0; i < d.keypoints.size(); ++i)
      {
        t.kpts_smooth[i].x = t.smoother.f[2 * i + 0].filter(d.keypoints[i].x, 1.f);
        t.kpts_smooth[i].y = t.smoother.f[2 * i + 1].filter(d.keypoints[i].y, 1.f);
        t.kpts_smooth[i].z = d.keypoints[i].z;
        t.kpts_smooth[i].score = d.keypoints[i].score;
      }
    }
    else
    {
      t.kpts_smooth = d.keypoints;
    }

    // EMA the per-track appearance embedding (kept L2-normalized).
    if(!d.embedding.empty())
    {
      if(t.embedding.size() != d.embedding.size())
        t.embedding = d.embedding; // first sight / changed model -> adopt
      else
      {
        const float a = _cfg.emb_ema;
        float norm = 0.f;
        for(size_t i = 0; i < t.embedding.size(); ++i)
        {
          t.embedding[i] = a * t.embedding[i] + (1.f - a) * d.embedding[i];
          norm += t.embedding[i] * t.embedding[i];
        }
        norm = std::sqrt(norm);
        if(norm > 1e-9f)
          for(auto& v : t.embedding)
            v /= norm;
      }
    }

    ++t.hits;
    t.time_since_update = 0;
    if(t.hits >= _cfg.min_hits)
      t.confirmed = true;
  }

  void birth(const Detection& d)
  {
    Tracklet t;
    t.id = _next_id++;
    _kf.initiate(boxToXyah(d.box), t.mean, t.cov);
    t.kpts = d.keypoints;
    t.kpts_smooth = d.keypoints;
    t.embedding = d.embedding;
    t.score = d.score;
    t.hits = 1;
    t.age = 1;
    t.time_since_update = 0;
    t.confirmed = (_cfg.min_hits <= 1);
    _tracks.push_back(std::move(t));
  }

  static constexpr float kReject = std::numeric_limits<float>::max();

  Config _cfg;
  KalmanFilter _kf;
  std::vector<Tracklet> _tracks;
  int _next_id = 1;
};

} // namespace Track
} // namespace Onnx
