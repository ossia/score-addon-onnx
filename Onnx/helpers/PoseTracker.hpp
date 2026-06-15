#pragma once
#include <Onnx/helpers/OneEuro.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
  float emb_ema = 0.9f;     // per-track embedding EMA factor (max incorporation)
  // Confidence-adaptive EMA (Deep OC-SORT "Dynamic Appearance"): below this
  // detection-quality threshold the new embedding is IGNORED (occlusion/blur
  // proxy), so a low-confidence frame can't contaminate the identity template
  // (and therefore the gallery, which stores the EMA'd template). Between thresh
  // and 1 the update rate ramps from 0 to (1 - emb_ema).
  float emb_quality_thresh = 0.5f;
  // Long-term appearance gallery: when a confirmed track dies (left the frame),
  // remember its id+embedding for gallery_ttl frames; a newly-born track whose
  // appearance matches (cosine >= gallery_gate) RE-ACQUIRES that id instead of
  // getting a fresh one. This is what lets a subject walk off and come back with
  // the same id. Off unless ReID is on.
  bool use_gallery = false;
  int gallery_ttl = 1800;   // ~30-60s at typical fps; entries expire after this
  float gallery_gate = 0.5f; // min cosine to re-acquire a gallery id
  // Multi-shot gallery: instead of one averaged (EMA) template per departed
  // identity, keep a small reservoir of DISTINCT clean appearance snapshots
  // (slot 0 = peak-confidence view, the rest sampled every snapshot_interval
  // frames). A query is matched by the BEST-matching snapshot in the set (plus a
  // vote count), so a returning person matches whichever stored pose/viewpoint
  // is closest to their current one — far more robust than a single mean.
  int snapshot_interval = 45; // frames between captured snapshots (~1.5s @ 30fps)
  int max_snapshots = 8;      // reservoir cap per identity

  // Emit per-decision tracing to stderr (which id was picked and why). Off by
  // default; the node turns it on from the SCORE_ONNX_TRACK_DEBUG env var.
  bool debug = false;
  // Ambiguity guard: the best gallery match must beat the SECOND-best by at
  // least this cosine margin before its id is adopted. Prevents ID "hijacking"
  // between near-identical people (teammates in the same kit, dancers) where the
  // top two gallery entries are similarly close. 0 disables the margin check.
  float reacquire_margin = 0.1f;

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
  // Multi-shot appearance reservoir frozen into the gallery on death: slot 0 is
  // the peak-confidence view, the rest are interval samples (round-robin).
  std::vector<std::vector<float>> snapshots;
  float snap_peak_q = -1.f; // quality of the peak snapshot (slot 0)
  int snap_last_age = 0;    // track age at the last interval sample
  int snap_rr = 1;          // round-robin write cursor over slots [1, max)

  float score = 0.f;
  int hits = 0;
  int age = 0;
  int time_since_update = 0;
  bool confirmed = false;
  // True while this track holds a FRESHLY-MINTED id (the birth-time gallery
  // re-acquire failed or didn't run). Such a track keeps re-checking the gallery
  // each frame with its EMA-stabilized embedding, so a person who was given a
  // wrong new id at re-entry gets corrected back to their original id once the
  // appearance stabilizes. Cleared once an id is gallery-confirmed.
  bool id_provisional = true;

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
    _gallery.clear();
    _next_id = 1;
  }

  // Update with this frame's detections. Returns, per input detection index,
  // the assigned track id (-1 if the detection started a still-tentative track
  // or was dropped). After this call, tracks() holds the live tracklets with
  // smoothed keypoints.
  const std::vector<int>& update(const std::vector<Detection>& dets)
  {
    // All per-frame scratch is held in reused members (cleared, never reallocated
    // once warm) — the hot path must not allocate. `out` is returned by const&;
    // callers consume it before the next update() overwrites it.
    auto& out = _assign;
    out.assign(dets.size(), -1);

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
    auto& high = _high;
    auto& low = _low;
    high.clear();
    low.clear();
    for(int i = 0; i < (int)dets.size(); ++i)
      (dets[i].score >= _cfg.track_high ? high : low).push_back(i);

    auto& t_match = _t_match;
    auto& d_used = _d_used;
    t_match.assign(_tracks.size(), -1);
    d_used.assign(dets.size(), 0);

    // 3) stage 1: all tracks vs high-score detections (IoU + OKS + dir)
    auto& all_t = _all_t;
    all_t.resize(_tracks.size());
    for(int i = 0; i < (int)_tracks.size(); ++i)
      all_t[i] = i;
    associate(all_t, high, dets, t_match, d_used, /*full_cost=*/true);

    // 4) stage 2: unmatched tracks vs low-score detections (IoU only — recovers
    //    confidence dips / partial occlusion, the ByteTrack trick)
    auto& rem_t = _rem_t;
    rem_t.clear();
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
      // Mid-track re-acquire: a provisional-id track (one given a fresh id
      // because the birth-time gallery match failed on a poor re-entry frame)
      // re-checks the gallery every frame with its now EMA-stabilized embedding.
      // Once it matches its original entry it adopts that id — correcting a
      // wrong id instead of being stuck with it for the rest of the session.
      maybeReacquireFromGallery(_tracks[i]);
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
    //    one-frame spurious detection never persists or is emitted). A confirmed
    //    track that dies with an appearance embedding is remembered in the
    //    gallery so the subject re-acquires its id on return.
    _tracks.erase(
        std::remove_if(
            _tracks.begin(), _tracks.end(),
            [&](const Tracklet& t) {
              const bool dead = t.time_since_update > _cfg.max_age
                                || (_cfg.strict_confirm && !t.confirmed
                                    && t.time_since_update > 0);
              if(dead)
              {
                const bool galleried = _cfg.use_gallery && t.confirmed
                                       && !t.snapshots.empty();
                if(galleried)
                  _gallery.push_back({t.id, t.snapshots, 0});
                if(_cfg.debug)
                  std::fprintf(
                      stderr,
                      "[track] DEATH id=%d (lost %d frames) -> %s (snaps=%d, "
                      "gallery=%d)\n",
                      t.id, t.time_since_update,
                      galleried ? "gallery" : "discarded",
                      (int)t.snapshots.size(), (int)_gallery.size());
              }
              return dead;
            }),
        _tracks.end());

    // Age the gallery; expire stale entries.
    if(_cfg.use_gallery)
    {
      for(auto& g : _gallery)
        ++g.age;
      _gallery.erase(
          std::remove_if(
              _gallery.begin(), _gallery.end(),
              [&](const GalleryEntry& g) { return g.age > _cfg.gallery_ttl; }),
          _gallery.end());
    }

    if(_cfg.debug)
    {
      std::fprintf(
          stderr, "[track] frame: %d dets -> ids [", (int)dets.size());
      for(size_t i = 0; i < out.size(); ++i)
        std::fprintf(stderr, "%s%d", i ? "," : "", out[i]);
      std::fprintf(
          stderr, "] | %d live, %d gallery\n", (int)_tracks.size(),
          (int)_gallery.size());
    }

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
        const float ddx = d.box.cx - tb.cx, ddy = d.box.cy - tb.cy;
        const float disp = std::sqrt(ddx * ddx + ddy * ddy);
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
      const float nv = std::sqrt(vx * vx + vy * vy);
      const float nt = std::sqrt(tx * tx + ty * ty);
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
    auto& pairs = _pairs; // reused member (cost matrix); never reallocated warm
    pairs.clear();
    pairs.reserve(tk.size() * dk.size());
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
        // Confidence-adaptive EMA factor: a is the fraction of the OLD template
        // kept. Low detection quality -> a -> 1 (ignore the new, possibly
        // occluded/blurred embedding); high quality -> a -> emb_ema. Mirrors
        // Deep OC-SORT's Dynamic Appearance, with the pose mean-confidence as the
        // quality proxy. When a ~= 1 the template (and thus the gallery) is left
        // untouched this frame, keeping the identity clean.
        const float sigma = _cfg.emb_quality_thresh;
        const float trust = (sigma < 1.f)
                                ? std::clamp(
                                      (d.score - sigma) / (1.f - sigma), 0.f, 1.f)
                                : 0.f;
        const float a = _cfg.emb_ema + (1.f - _cfg.emb_ema) * (1.f - trust);
        if(a < 0.999f)
        {
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

      // Multi-shot reservoir: capture only CLEAN (high-quality) RAW embeddings
      // (not the EMA), so each snapshot is a crisp distinct view rather than a
      // blurred average. Slot 0 holds the peak-confidence view; the rest are
      // interval samples written round-robin to stay temporally diverse.
      if(_cfg.use_gallery && d.score >= _cfg.emb_quality_thresh)
      {
        if(d.score > t.snap_peak_q)
        {
          t.snap_peak_q = d.score;
          if(t.snapshots.empty())
            t.snapshots.push_back(d.embedding);
          else
            t.snapshots[0] = d.embedding;
        }
        if(_cfg.max_snapshots > 1
           && t.age - t.snap_last_age >= _cfg.snapshot_interval)
        {
          t.snap_last_age = t.age;
          if((int)t.snapshots.size() < _cfg.max_snapshots)
            t.snapshots.push_back(d.embedding);
          else
          {
            if(t.snap_rr < 1 || t.snap_rr >= (int)t.snapshots.size())
              t.snap_rr = 1;
            t.snapshots[t.snap_rr++] = d.embedding;
          }
        }
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
    // Long-term re-acquire: if this new detection's appearance matches a
    // remembered (departed) track, reuse that id instead of minting a new one.
    int reacquired = -1;
    if(_cfg.use_gallery)
    {
      const int g = findGalleryMatch(d.embedding, "birth");
      if(g >= 0)
      {
        reacquired = _gallery[g].id;
        // Inherit the identity's whole appearance reservoir so its memory (and
        // multi-view robustness) persists across the exit/return, and pin slot 0
        // so the original peak view is not overwritten by this re-entry frame.
        t.snapshots = std::move(_gallery[g].snapshots);
        t.snap_peak_q = 1.f;
        _gallery.erase(_gallery.begin() + g);
      }
    }
    t.id = (reacquired >= 0) ? reacquired : _next_id++;
    if(_cfg.debug)
      std::fprintf(
          stderr, "[track] BIRTH id=%d (%s) score=%.2f reid=%s\n", t.id,
          reacquired >= 0 ? "RE-ACQUIRED" : "new", d.score,
          d.embedding.empty() ? "no" : "yes");
    _kf.initiate(boxToXyah(d.box), t.mean, t.cov);
    t.kpts = d.keypoints;
    t.kpts_smooth = d.keypoints;
    t.embedding = d.embedding;
    t.score = d.score;
    t.hits = 1;
    t.age = 1;
    t.time_since_update = 0;
    // A re-acquired (gallery-matched) track is a known subject — confirm it
    // immediately so its id reappears at once instead of after min_hits frames.
    t.confirmed = (reacquired >= 0) || (_cfg.min_hits <= 1);
    // A gallery-matched id is final; a fresh id stays provisional so it keeps
    // re-checking the gallery as its embedding stabilizes.
    t.id_provisional = (reacquired < 0);
    _tracks.push_back(std::move(t));
  }

  // Best gallery entry for `emb` that (a) passes the cosine gate, (b) beats the
  // SECOND-best by reacquire_margin (ambiguity guard vs near-identical people),
  // and (c) whose id is not already held by a live track (one id per frame).
  // Returns the gallery index, or -1. Shared by birth and mid-track adoption.
  int findGalleryMatch(const std::vector<float>& emb, const char* ctx = "") const
  {
    if(emb.empty() || _gallery.empty())
      return -1;
    // Per entry: sim = BEST-matching snapshot (DeepSORT min-distance, robust to
    // pose change), votes = how many snapshots match (consensus). Rank entries
    // by votes then sim; track the runner-up sim for the ambiguity margin.
    int best_g = -1, best_votes = -1;
    float best_sim = -2.f, second_sim = -2.f;
    for(int g = 0; g < (int)_gallery.size(); ++g)
    {
      float sim = -2.f;
      int votes = 0;
      for(const auto& s : _gallery[g].snapshots)
      {
        const float c = cosine(s, emb);
        if(c > sim)
          sim = c;
        if(c >= _cfg.gallery_gate)
          ++votes;
      }
      const bool better
          = (votes > best_votes) || (votes == best_votes && sim > best_sim);
      if(better)
      {
        if(best_g >= 0 && best_sim > second_sim)
          second_sim = best_sim;
        best_g = g;
        best_votes = votes;
        best_sim = sim;
      }
      else if(sim > second_sim)
      {
        second_sim = sim;
      }
    }
    const int cand_id = best_g >= 0 ? _gallery[best_g].id : -1;
    const int cand_snaps
        = best_g >= 0 ? (int)_gallery[best_g].snapshots.size() : 0;
    if(best_g < 0 || best_sim < _cfg.gallery_gate)
    {
      if(_cfg.debug)
        std::fprintf(
            stderr,
            "[track] gallery %s: REJECT (below gate) best_id=%d sim=%.3f "
            "votes=%d gate=%.2f (gallery=%d)\n",
            ctx, cand_id, best_sim, best_votes, _cfg.gallery_gate,
            (int)_gallery.size());
      return -1;
    }
    // Ambiguity guard: the best identity must beat the runner-up by a margin,
    // else we'd risk hijacking a similar-looking person's id.
    if(_cfg.reacquire_margin > 0.f && second_sim > -1.f
       && (best_sim - second_sim) < _cfg.reacquire_margin)
    {
      if(_cfg.debug)
        std::fprintf(
            stderr,
            "[track] gallery %s: REJECT (ambiguous) best_id=%d sim=%.3f vs "
            "2nd=%.3f margin=%.2f\n",
            ctx, cand_id, best_sim, second_sim, _cfg.reacquire_margin);
      return -1;
    }
    // One id per frame: never adopt an id a live track already holds.
    const int id = _gallery[best_g].id;
    for(const auto& t : _tracks)
      if(t.id == id)
      {
        if(_cfg.debug)
          std::fprintf(
              stderr,
              "[track] gallery %s: REJECT (id %d already live) sim=%.3f\n", ctx,
              id, best_sim);
        return -1;
      }
    if(_cfg.debug)
      std::fprintf(
          stderr,
          "[track] gallery %s: MATCH id=%d sim=%.3f votes=%d/%d 2nd=%.3f\n", ctx,
          cand_id, best_sim, best_votes, cand_snaps, second_sim);
    return best_g;
  }

  // If a provisional-id track's EMA appearance now matches a gallery entry,
  // adopt that (original) id and lock it. Requires a couple of observations so
  // the embedding is past its noisy birth value.
  void maybeReacquireFromGallery(Tracklet& t)
  {
    if(!_cfg.use_gallery || !t.id_provisional || t.hits < 2)
      return;
    const int g = findGalleryMatch(t.embedding, "midtrack");
    if(g >= 0)
    {
      const int old = t.id;
      t.id = _gallery[g].id;
      // Inherit the original identity's appearance reservoir (as birth() does),
      // pinning slot 0 so the stored peak view survives. Without this the
      // correction would discard the long-term memory and re-gallery only this
      // track's younger, poorer snapshots — degrading re-ID on every correction.
      t.snapshots = std::move(_gallery[g].snapshots);
      t.snap_peak_q = 1.f;
      _gallery.erase(_gallery.begin() + g);
      t.id_provisional = false;
      if(_cfg.debug)
        std::fprintf(
            stderr, "[track] RE-ACQUIRE mid-track: id %d -> %d\n", old, t.id);
    }
  }

  static constexpr float kReject = std::numeric_limits<float>::max();

  Config _cfg;
  KalmanFilter _kf;
  std::vector<Tracklet> _tracks;
  int _next_id = 1;

  // Per-frame association scratch, hoisted out of update()/associate() so the
  // hot path never heap-allocates once these have grown to their working size.
  struct Pair
  {
    float c;
    int ti, di;
  };
  std::vector<int> _assign;  // returned by update() (one entry per detection)
  std::vector<int> _high, _low, _t_match, _all_t, _rem_t;
  std::vector<char> _d_used;
  std::vector<Pair> _pairs;  // cost matrix candidates

  // Long-term ReID gallery: embeddings of confirmed tracks that left the frame,
  // kept so a returning subject re-acquires its old id (see Config::use_gallery).
  struct GalleryEntry
  {
    int id;
    std::vector<std::vector<float>> snapshots; // multi-shot appearance set
    int age = 0;
  };
  std::vector<GalleryEntry> _gallery;
};

} // namespace Track
} // namespace Onnx
