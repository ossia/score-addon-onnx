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
  std::vector<Keypoint> keypoints; // may be empty -> OKS skipped for this det
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
    const float h = mean(3);
    State s;
    s << _spw * h, _spw * h, 1e-2f, _spw * h, _svw * h, _svw * h, 1e-5f,
        _svw * h;
    const Cov Q = (s.array() * s.array()).matrix().asDiagonal();
    mean = _F * mean;
    cov = _F * cov * _F.transpose() + Q;
  }

  void update(State& mean, Cov& cov, const Meas& m) const
  {
    const float h = mean(3);
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

struct Tracklet
{
  int id = -1;
  KalmanFilter::State mean;
  KalmanFilter::Cov cov;

  std::vector<Keypoint> kpts;        // last observed, motion-compensated
  std::vector<Keypoint> kpts_smooth; // One-Euro output (owned by this id)
  PoseSmoother smoother;             // 2 filters per keypoint (x,y)

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
      birth(dets[di]);
    }

    // 7) death: drop tracks lost longer than max_age
    _tracks.erase(
        std::remove_if(
            _tracks.begin(), _tracks.end(),
            [&](const Tracklet& t)
            { return t.time_since_update > _cfg.max_age; }),
        _tracks.end());

    return out;
  }

private:
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

    // gate: need box overlap OR pose support
    if(i < _cfg.gate_iou && (!have_oks || o < _cfg.gate_oks))
      return kReject;

    float c = _cfg.w_iou * (1.f - i);
    if(have_oks)
      c += _cfg.w_oks * (1.f - o);
    else
      c += _cfg.w_oks * (1.f - i); // no pose -> lean on box

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
