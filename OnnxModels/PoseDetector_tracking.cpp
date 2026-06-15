#include "PoseDetector_internal.hpp"

namespace OnnxModels
{

// Death gate: on a no-detection frame, age the multi-instance tracker.
// update({}) predicts every track forward (Kalman) and advances
// time_since_update, so the death sweep (max_age) fires and a track surviving a
// brief dropout keeps its id on re-acquisition instead of being matched as a
// fresh detection -> new id. Without this the tracker is frozen, not aged.
void PoseDetector::ageTracker()
{
  m_track_in.clear();
  m_tracker.update(m_track_in);
}

// Rebuild m_instances from the still-live, recently-seen tracker tracks (their
// Kalman-coasted box + One-Euro keypoints) and draw/publish them, so an
// instance doesn't visually blink during a 1-2 frame dropout. Returns false if
// no track is fresh enough to show (caller then falls back to passthrough).
bool PoseDetector::reEmitCoasted(PoseWorkflow draw)
{
  // Only show a coasted ghost for a few frames; a subject that genuinely left
  // shouldn't linger (the track itself still survives to max_age for ID reuse).
  const int coast_limit
      = std::clamp(static_cast<int>(inputs.detector_cadence.value), 1, 5);
  m_instances.clear();
  for(const auto& tk : m_tracker.tracks())
  {
    if(!tk.confirmed || tk.time_since_update <= 0
       || tk.time_since_update > coast_limit || tk.kpts_smooth.empty())
      continue;
    DetectedPose p;
    p.keypoints.reserve(tk.kpts_smooth.size());
    for(const auto& k : tk.kpts_smooth)
      p.keypoints.push_back({k.x, k.y, k.z, k.score});
    const auto bb = tk.box();
    if(bb.w > 0.f && bb.h > 0.f)
      p.box = {bb.cx - bb.w * 0.5f, bb.cy - bb.h * 0.5f, bb.w, bb.h};
    p.mean_confidence = tk.score;
    p.track_id = tk.id;
    m_instances.push_back(std::move(p));
  }
  if(m_instances.empty())
    return false;
  // do_track=false: we already aged the tracker; just remap/draw/publish.
  emitInstances(draw, /*do_track=*/false);
  return true;
}

// Tracking-path no-detection handler: age the tracker, re-emit coasted poses if
// any are fresh, else passthrough.
void PoseDetector::coastOrPassthrough(
    PoseWorkflow draw, const Onnx::ImageView& src)
{
  if(inputs.track_ids.value)
  {
    ageTracker();
    if(reEmitCoasted(draw))
      return;
  }
  passthrough(src);
}

void PoseDetector::applySmoothing(DetectedPose& pose)
{
  m_lost_frames = 0; // we have a pose this frame
  if(!inputs.smoothing.value || pose.keypoints.empty())
    return;
  const float amt
      = std::clamp(static_cast<float>(inputs.smoothing_amount.value), 0.f, 1.f);

  // One-Euro is a jitter-vs-lag trade governed by TWO params (see the original
  // paper + MediaPipe's LandmarksSmoothingCalculator):
  //   * min_cutoff — the smoothing when the point is (nearly) still. Low = very
  //     smooth at rest. This is what kills jiggle.
  //   * beta       — how fast the cutoff opens up with speed. This is what keeps
  //     motion responsive (low lag). For NORMALIZED [0,1] coords the per-frame
  //     velocity is tiny (~0.01-0.1), so beta must be O(1-10), not 0.02 — that
  //     was why max smoothing felt "too smooth/laggy": no speed adaptivity.
  // Map the single Amount slider onto both: more amount => lower rest-cutoff and
  // (slightly) higher beta so heavy smoothing still tracks fast motion.
  const float min_cutoff = 5.0f * std::pow(0.02f / 5.0f, amt); // 5 -> 0.02 Hz
  const float beta = 1.0f + 4.0f * amt;                        // 1 -> 5

  // Filter the raw normalized coordinates. NOTE: an earlier "scale-aware"
  // variant divided each coord by the per-frame keypoint-bbox diagonal. Because
  // the One-Euro STATE is kept in those units while the bbox size changes every
  // frame, a frame with a small/sparse bbox multiplied the lagged state by a
  // tiny scale and collapsed *every* point toward (0,0) — the "all points jump
  // to the top-left" glitch. Scale normalization needs a STABLE per-subject
  // scale; revisit once per-track IDs (ByteTrack) provide one.
  m_smoother.configure(min_cutoff, beta);
  m_smoother.ensure(pose.keypoints.size() * 3);
  const bool have_prev
      = m_prev_smoothed_kps.size() == pose.keypoints.size();
  for(size_t i = 0; i < pose.keypoints.size(); ++i)
  {
    auto& k = pose.keypoints[i];
    // A single NaN/Inf from an untrusted model must not enter the One-Euro
    // state: x_prev would stay NaN forever (the multi-instance path sanitizes
    // in emitInstances; this is the single-instance equivalent). Leave the
    // raw value and this joint's filter state untouched.
    if(!finitef(k.x) || !finitef(k.y) || !finitef(k.z))
      continue;
    k.x = m_smoother.f[i * 3 + 0].filter(k.x, 1.0f);
    k.y = m_smoother.f[i * 3 + 1].filter(k.y, 1.0f);
    k.z = m_smoother.f[i * 3 + 2].filter(k.z, 1.0f);

    // Confidence-weighted hold: a LOW-confidence joint (e.g. the face/hand
    // points of a wholebody-133 model on a distant person) is unreliable and
    // its SimCC peak hops around, producing the "ultra fast jumping" the One-
    // Euro can't tame (its velocity term opens the cutoff on a teleport). Blend
    // toward the previous smoothed position by (1 - confidence), so an unsure
    // joint sticks and a confident one tracks freely.
    if(have_prev)
    {
      const float w = std::clamp(k.confidence, 0.f, 1.f); // trust = confidence
      const auto& pk = m_prev_smoothed_kps[i];
      if(finitef(pk.x) && finitef(pk.y))
      {
        k.x = w * k.x + (1.f - w) * pk.x;
        k.y = w * k.y + (1.f - w) * pk.y;
        k.z = w * k.z + (1.f - w) * pk.z;
      }
    }
  }
  m_prev_smoothed_kps = pose.keypoints;
}

// ROI rect derived from the previous frame's landmarks (MediaPipe tracking
// loop): bbox of the confident keypoints, expanded, with a rotation taken from
// a domain-appropriate keypoint pair so the crop follows the subject.
Onnx::ROI::Rect PoseDetector::roiRectFromKeypoints(
    PoseWorkflow draw, const std::vector<PoseKeypoint>& kps, int W, int H,
    int model_w, int model_h)
{
  Onnx::ROI::Rect r{}; // w==0 means "invalid" (caller falls back to detecting)
  if(kps.empty())
    return r;

  auto kp = [&](int i) -> Onnx::Vec2 {
    return (i >= 0 && i < (int)kps.size()) ? Onnx::Vec2{kps[i].x * W, kps[i].y * H}
                                           : Onnx::Vec2{0, 0};
  };
  auto conf = [&](int i) {
    return (i >= 0 && i < (int)kps.size()) ? kps[i].confidence : 0.f;
  };
  auto mid = [&](int a, int b) -> Onnx::Vec2 {
    return {(kp(a).x + kp(b).x) * 0.5f, (kp(a).y + kp(b).y) * 0.5f};
  };

  // bbox of confident keypoints (image px) + count
  int nconf = 0;
  float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
  for(const auto& k : kps)
  {
    if(k.confidence < 0.2f)
      continue;
    ++nconf;
    const float x = k.x * W, y = k.y * H;
    minx = std::min(minx, x); maxx = std::max(maxx, x);
    miny = std::min(miny, y); maxy = std::max(maxy, y);
  }
  if(nconf < 3 || maxx <= minx)
    return r; // too sparse/degenerate -> invalid -> re-detect
  const float bbox_cx = 0.5f * (minx + maxx);
  const float bbox_cy = 0.5f * (miny + maxy);
  const float bw = maxx - minx, bh = maxy - miny;

  switch(draw)
  {
    case PoseWorkflow::BlazePose:
    {
      // The hip/shoulder anchors drive the crop center, size and rotation. If
      // they are not themselves confident, the rect is built on garbage joint
      // positions (kp() of a low-confidence joint), which drags the crop toward
      // the origin and shrinks it — the runaway inward collapse over a few ROI
      // cycles the user saw. Bail to a re-detect instead of trusting them.
      if(conf(23) < 0.3f || conf(24) < 0.3f || conf(11) < 0.3f
         || conf(12) < 0.3f)
        return r; // invalid -> caller re-detects with the BlazePose detector

      // Match the DETECTOR's alignment rect: hip-centered, size = 2 x radius to
      // the farthest body point, rotated so hips->shoulders points "up". Using
      // the keypoint bbox center/size instead re-frames the crop and makes the
      // landmark model drift (and can collapse to image center).
      const Onnx::Vec2 hip = mid(23, 24), sh = mid(11, 12);
      float R = 0.f;
      for(int i = 0; i < (int)kps.size(); ++i)
        if(conf(i) >= 0.2f)
          R = std::max(
              R, float(std::hypot(kp(i).x - hip.x, kp(i).y - hip.y)));
      const float size = 2.0f * R * 1.15f;
      const float target = float(M_PI) / 2.0f;
      r.cx = hip.x;
      r.cy = hip.y;
      r.w = r.h = size;
      r.angle = target - std::atan2(-(sh.y - hip.y), sh.x - hip.x);
      break;
    }
    case PoseWorkflow::MediaPipeHands:
    {
      const Onnx::Vec2 wrist = kp(0), mcp = kp(9);
      const float size = std::max(bw, bh) * 2.0f;
      r.cx = bbox_cx; r.cy = bbox_cy; r.w = r.h = size;
      r.angle = float(M_PI) / 2.0f
                - std::atan2(-(mcp.y - wrist.y), mcp.x - wrist.x);
      break;
    }
    case PoseWorkflow::FaceMesh:
    case PoseWorkflow::MobileFaceNet:
    {
      const int e0 = (draw == PoseWorkflow::FaceMesh) ? 33 : 36;
      const int e1 = (draw == PoseWorkflow::FaceMesh) ? 263 : 45;
      const float size = std::max(bw, bh) * 1.5f;
      r.cx = bbox_cx; r.cy = bbox_cy; r.w = r.h = size;
      r.angle = -std::atan2(
          -(kp(e1).y - kp(e0).y), kp(e1).x - kp(e0).x);
      break;
    }
    default:
    {
      // top-down (RTMPose/ViTPose/YOLO): axis-aligned bbox, aspect-fixed.
      // Recompute the bbox from only WELL-confident points: wholebody (133)
      // emits many low-confidence face/hand points that, if included, blow up
      // the bbox and send the crop off the subject. Require >=0.35 here.
      float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
      int n = 0;
      for(const auto& k : kps)
      {
        if(k.confidence < 0.35f)
          continue;
        ++n;
        const float x = k.x * W, y = k.y * H;
        bx0 = std::min(bx0, x); bx1 = std::max(bx1, x);
        by0 = std::min(by0, y); by1 = std::max(by1, y);
      }
      if(n < 3 || bx1 <= bx0)
        return Onnx::ROI::Rect{}; // invalid -> re-detect
      float sw = (bx1 - bx0) * 1.25f, sh = (by1 - by0) * 1.25f;
      const float a = float(model_w) / float(model_h);
      if(sw > sh * a) sh = sw / a; else sw = sh * a;
      r.cx = 0.5f * (bx0 + bx1); r.cy = 0.5f * (by0 + by1);
      r.w = sw; r.h = sh; r.angle = 0.f;
      break;
    }
  }
  return r;
}

Onnx::ROI::Rect PoseDetector::smoothRoi(Onnx::ROI::Rect r)
{
  if(!inputs.track_roi.value && !inputs.smoothing.value)
    return r;
  // ROI is smoothed a bit more aggressively than keypoints (it should be very
  // stable for a still subject); reuse the smoothing amount, biased lower.
  const float amt
      = std::clamp(static_cast<float>(inputs.smoothing_amount.value), 0.f, 1.f);
  const float min_cutoff = 15.0f * std::pow(0.005f / 15.0f, amt);
  m_roi_smoother.configure(min_cutoff, 0.01f);
  float v[5] = {r.cx, r.cy, r.w, r.h, r.angle};
  m_roi_smoother.smooth(v, 1.0f);
  return Onnx::ROI::Rect{v[0], v[1], v[2], v[3], v[4]};
}

// Multi-instance two-stage path (Track IDs on): detect top-K (or reuse per-track
// ROIs on detector-skip frames), landmark each into m_instances, then emit.
void PoseDetector::runMultiInstance(
    const Onnx::ModelRole& role, PoseWorkflow draw, const Onnx::ImageView& src)
{
  const int W = inputs.image.texture.width, H = inputs.image.texture.height;
  const int mw = role.input_w > 0 ? role.input_w : 256;
  const int mh = role.input_h > 0 ? role.input_h : 256;
  const int max_inst = std::clamp(static_cast<int>(inputs.max_instances.value), 1, 16);

  // Per-track ROI feedback (detector-skip) is only steady for the
  // MediaPipe-rotated landmark kinds; top-down (SimCC/heatmap) re-detects.
  const bool roi_trackable
      = role.kind == Onnx::ModelKind::BlazePoseLandmark
        || role.kind == Onnx::ModelKind::HandLandmark
        || role.kind == Onnx::ModelKind::FaceMeshLandmark;
  const int detect_cadence
      = std::max(1, static_cast<int>(inputs.detector_cadence.value));

  // --- Gather this frame's ROIs (image px), then batch-landmark them. ---
  m_rois.clear();

  const bool use_track_rois = inputs.track_roi.value && roi_trackable
                              && !m_tracker.tracks().empty()
                              && m_frames_since_detect < detect_cadence;
  if(use_track_rois)
  {
    ++m_frames_since_detect;
    for(const auto& tk : m_tracker.tracks())
    {
      if(tk.time_since_update != 0 || tk.kpts_smooth.empty())
        continue;
      m_kp_scratch.clear();
      m_kp_scratch.reserve(tk.kpts_smooth.size());
      for(const auto& k : tk.kpts_smooth)
        m_kp_scratch.push_back({k.x, k.y, k.z, k.score});
      const Onnx::ROI::Rect r
          = roiRectFromKeypoints(draw, m_kp_scratch, W, H, mw, mh);
      if(rectValid(r, W, H))
        m_rois.push_back(r);
    }
    if(m_rois.empty())
      m_frames_since_detect = detect_cadence; // force re-detect next frame
  }

  if(m_rois.empty())
  {
    m_frames_since_detect = 0;
    m_dets = runDetector(m_detector_role, src, role.domain);
    // App-side NMS before the tracker: the end2end / PINTO / pre-decoded
    // detector branches assume the graph already deduplicated, but some exports
    // emit overlapping boxes for one subject. Two boxes -> two ROIs -> two
    // competing tracks -> per-frame ID churn. NMS here is idempotent for the
    // branches that already ran it inside runDetector.
    m_dets = Onnx::Detection::nms(std::move(m_dets), 0.45f, 0.8f);
    if(m_dets.empty())
    {
      coastOrPassthrough(draw, src);
      return;
    }
    if(static_cast<int>(m_dets.size()) > max_inst)
    {
      std::partial_sort(
          m_dets.begin(), m_dets.begin() + max_inst, m_dets.end(),
          [](const auto& a, const auto& b) { return a.score > b.score; });
      m_dets.resize(max_inst);
    }
    for(const auto& d : m_dets)
    {
      const Onnx::ROI::Rect r = detectionRect(role, d, W, H);
      if(rectValid(r, W, H))
        m_rois.push_back(r);
    }
  }

  if(m_rois.empty())
  {
    coastOrPassthrough(draw, src);
    return;
  }

  runLandmarkBatch(role, draw, src, m_rois); // batched if the model allows

  if(m_instances.empty())
  {
    coastOrPassthrough(draw, src);
    return;
  }
  emitInstances(draw, /*do_track=*/true);
}

// Track m_instances, assign ids + per-id smoothed keypoints + colors, draw all,
// and publish every output port (poses/detection/geometry/poses_geometry/count).
void PoseDetector::emitInstances(PoseWorkflow draw, bool do_track)
{
  m_lost_frames = 0;

  // Every instance carries its bbox in metadata (detector box for box-only
  // detections, else the confident-keypoint bbox).
  for(auto& pose : m_instances)
    fillBoxFromKeypoints(pose);

  if(do_track)
  {
    // The tracker owns per-id One-Euro smoothing (no cross-identity bleed).
    Onnx::Track::Config cfg = m_tracker.config();
    cfg.smooth = inputs.smoothing.value;
    if(inputs.smoothing.value)
    {
      const float amt = std::clamp(
          static_cast<float>(inputs.smoothing_amount.value), 0.f, 1.f);
      cfg.smooth_min_cutoff = 5.0f * std::pow(0.02f / 5.0f, amt);
      cfg.smooth_beta = 1.0f + 4.0f * amt;
    }
    cfg.use_reid = inputs.reid.value && reid_ctx && m_reid_spec.valid;
    cfg.w_emb = std::clamp(static_cast<float>(inputs.reid_weight.value), 0.f, 1.f);
    // Coast window (how long a lost track is kept) and the long-term gallery
    // (how long a departed person's appearance is remembered for re-acquisition)
    // are user-controllable. The gallery only runs with Re-ID on and a non-zero
    // memory.
    cfg.max_age = std::max(1, static_cast<int>(inputs.track_memory.value));
    cfg.gallery_ttl = std::max(0, static_cast<int>(inputs.reid_memory.value));
    cfg.use_gallery = cfg.use_reid && cfg.gallery_ttl > 0;
    cfg.reacquire_margin
        = std::clamp(static_cast<float>(inputs.reid_margin.value), 0.f, 1.f);
    // Per-decision tracing to stderr, enabled by SCORE_ONNX_TRACK_DEBUG=1
    // (checked once). Shows which ids are picked at birth/death/re-acquire.
    static const bool track_debug
        = std::getenv("SCORE_ONNX_TRACK_DEBUG") != nullptr;
    cfg.debug = track_debug;

    // Plausibility gates (anti-jitter). Each is independent so methods can be
    // A/B-compared; defaults reproduce the pre-gate baseline.
    cfg.motion_gate = inputs.motion_gate.value;
    cfg.max_speed = std::max(0.01f, static_cast<float>(inputs.max_speed.value));
    cfg.birth_gate = inputs.birth_gate.value;
    cfg.strict_confirm = inputs.strict_confirm.value;
    m_tracker.configure(cfg);

    // One-line config trace (on change) so it's obvious WHY re-id is/ isn't
    // active and what the lost-track window actually is. Explains a reid=no at
    // birth (toggle on but no reid_ctx / invalid spec) and a too-short max_age.
    if(cfg.debug)
    {
      char buf[256];
      std::snprintf(
          buf, sizeof(buf),
          "[track] cfg: reid_toggle=%d reid_ctx=%d reid_spec_valid=%d -> "
          "use_reid=%d | max_age=%d gallery_ttl=%d gate=%.2f margin=%.2f",
          (int)inputs.reid.value, reid_ctx ? 1 : 0, m_reid_spec.valid ? 1 : 0,
          cfg.use_reid ? 1 : 0, cfg.max_age, cfg.gallery_ttl, cfg.gallery_gate,
          cfg.reacquire_margin);
      if(std::string(buf) != m_last_cfg_log)
      {
        m_last_cfg_log = buf;
        std::fprintf(stderr, "%s\n", buf);
      }
    }

    m_track_in.clear();
    m_track_in.reserve(m_instances.size());
    for(const auto& pose : m_instances)
    {
      Onnx::Track::Detection td;
      // Tracker box (center form) from the pose bbox; keypoints (if any) feed
      // the OKS cue, otherwise the tracker leans on IoU + Re-ID. A non-finite
      // box (a NaN model coordinate reaching here) would poison the Kalman state
      // permanently, so neutralize it: degenerate box + zero score (won't birth
      // or match). Index alignment with m_instances must be preserved, so we
      // sanitize rather than skip.
      const auto& pb = pose.box;
      const bool box_ok = finitef(pb.x) && finitef(pb.y) && finitef(pb.w)
                          && finitef(pb.h) && pb.w > 0.f && pb.h > 0.f;
      if(box_ok)
        td.box = {pb.x + pb.w * 0.5f, pb.y + pb.h * 0.5f, pb.w, pb.h};
      else
        td.box = {0.5f, 0.5f, 1e-3f, 1e-3f};
      td.score = box_ok ? pose.mean_confidence : 0.f;
      td.keypoints.reserve(pose.keypoints.size());
      for(const auto& k : pose.keypoints)
      {
        if(finitef(k.x) && finitef(k.y))
          td.keypoints.push_back({k.x, k.y, k.z, k.confidence});
        else // neutralize a NaN keypoint without breaking OKS's equal-size need
          td.keypoints.push_back({td.box.cx, td.box.cy, 0.f, 0.f});
      }
      m_track_in.push_back(std::move(td));
    }

    if(cfg.use_reid)
      embedInstances(); // crop + Re-ID -> m_track_in[i].embedding

    const auto& ids = m_tracker.update(m_track_in);
    for(size_t i = 0; i < m_instances.size(); ++i)
    {
      const int id = (i < ids.size()) ? ids[i] : -1;
      m_instances[i].track_id = id;
      if(id < 0 || !inputs.smoothing.value)
        continue;
      for(const auto& tk : m_tracker.tracks())
      {
        if(tk.id != id)
          continue;
        if(tk.kpts_smooth.size() == m_instances[i].keypoints.size())
          for(size_t k = 0; k < tk.kpts_smooth.size(); ++k)
          {
            m_instances[i].keypoints[k].x = tk.kpts_smooth[k].x;
            m_instances[i].keypoints[k].y = tk.kpts_smooth[k].y;
          }
        // Smooth the bbox too: the tracker's Kalman box state is temporally
        // filtered, so emit it instead of the raw per-frame detection box.
        const auto bb = tk.box();
        if(bb.w > 0.f && bb.h > 0.f)
          m_instances[i].box
              = {bb.cx - bb.w * 0.5f, bb.cy - bb.h * 0.5f, bb.w, bb.h};
        break;
      }
    }
  }

  // Remap every instance to the chosen target skeleton, AFTER tracking/
  // smoothing (which ran on the native layout). The overlay and every output
  // port below then share the remapped layout.
  setRemapState(
      draw, m_instances.empty()
                ? 0
                : static_cast<int>(m_instances.front().keypoints.size()));
  if(m_remap_active)
    for(auto& pose : m_instances)
      remapPose(pose);

  // primary = highest-confidence instance (back-compat single-pose ports)
  int primary = -1;
  float best = -1.f;
  for(size_t i = 0; i < m_instances.size(); ++i)
    if(m_instances[i].mean_confidence > best)
    {
      best = m_instances[i].mean_confidence;
      primary = static_cast<int>(i);
    }

  drawAllSkeletons(draw);

  outputs.poses.value = m_instances;
  if(primary >= 0)
  {
    outputs.detection.value = m_instances[primary];
    generateGeometryOutput(m_instances[primary], draw); // fills outputs.geometry
  }
  else
  {
    outputs.detection.value.reset();
    outputs.geometry.value.clear();
  }

  // Fixed-stride multi geometry: max_inst slots of
  //   [track_id, class_id, box_x, box_y, box_w, box_h, (x,y,z,conf)*K],
  // zero-padded. Constant layout regardless of Data Format (GPU-friendly).
  // K is 0 for box-only detections (header carries the box).
  constexpr int HEADER = 6;
  const int max_inst = std::clamp(
      static_cast<int>(inputs.max_instances.value),
      1, 16);
  auto& pg = outputs.poses_geometry.value;
  pg.clear();
  if(!m_instances.empty())
  {
    // Stride from the MAX keypoint count across instances so a mixed-K frame
    // (e.g. body + hand) never truncates the richer instances; shorter ones
    // zero-pad. (Today all instances share K, but don't bake that in.)
    int nkpt = 0;
    for(const auto& p : m_instances)
      nkpt = std::max(nkpt, static_cast<int>(p.keypoints.size()));
    const int stride = HEADER + nkpt * 4;
    pg.assign(static_cast<size_t>(max_inst) * stride, 0.f);
    int slot = 0;
    for(const auto& pose : m_instances)
    {
      if(slot >= max_inst)
        break;
      float* s = pg.data() + static_cast<size_t>(slot) * stride;
      s[0] = static_cast<float>(pose.track_id);
      s[1] = static_cast<float>(pose.class_id);
      s[2] = pose.box.x;
      s[3] = pose.box.y;
      s[4] = pose.box.w;
      s[5] = pose.box.h;
      const int K = std::min(nkpt, static_cast<int>(pose.keypoints.size()));
      for(int k = 0; k < K; ++k)
      {
        s[HEADER + k * 4 + 0] = pose.keypoints[k].x;
        s[HEADER + k * 4 + 1] = pose.keypoints[k].y;
        s[HEADER + k * 4 + 2] = pose.keypoints[k].z;
        s[HEADER + k * 4 + 3] = pose.keypoints[k].confidence;
      }
      ++slot;
    }
  }
  outputs.count.value = static_cast<int>(m_instances.size());
}

// Crop each instance's person box, run the Re-ID model (batched if it allows),
// L2-normalize the feature vector, and store it in m_track_in[i].embedding.
void PoseDetector::embedInstances()
{
  if(!reid_ctx || !m_reid_spec.valid || m_track_in.empty())
    return;
  auto& rctx = *reid_ctx;
  const auto& spec = rctx.readModelSpec();
  if(spec.inputs.empty() || spec.outputs.empty())
    return;

  const int W = inputs.image.texture.width, H = inputs.image.texture.height;
  Onnx::ImageView src{
      reinterpret_cast<const uint8_t*>(inputs.image.texture.bytes), W, H, 4,
      W * 4};
  const int mw = m_reid_spec.in_w, mh = m_reid_spec.in_h;
  const int D = m_reid_spec.embed_dim;
  const int oidx = m_reid_spec.out_index;
  const int N = static_cast<int>(m_track_in.size());

  // Resolve preprocessing (channel order + mean/std), Auto by input size.
  ReidPreprocess pp = inputs.reid_preprocess.value;
  if(pp == ReidPreprocess::Auto)
    pp = (mw == 112 && mh == 112) ? ReidPreprocess::ArcFaceRGB
         : (mw <= 128 && mh <= 128) ? ReidPreprocess::RawBGR
                                    : ReidPreprocess::ImageNetRGB;
  bool bgr = false;
  std::array<float, 3> mean{0, 0, 0}, stdv{255, 255, 255};
  switch(pp)
  {
    case ReidPreprocess::ImageNetRGB:
      mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
      stdv = {0.229f * 255, 0.224f * 255, 0.225f * 255};
      break;
    case ReidPreprocess::RawBGR:
      bgr = true; mean = {0, 0, 0}; stdv = {1, 1, 1};
      break;
    case ReidPreprocess::RawRGB:
      mean = {0, 0, 0}; stdv = {1, 1, 1};
      break;
    case ReidPreprocess::ZeroOneRGB:
      mean = {0, 0, 0}; stdv = {255, 255, 255};
      break;
    case ReidPreprocess::ArcFaceRGB:
      mean = {127.5f, 127.5f, 127.5f}; stdv = {128, 128, 128};
      break;
    default:
      break;
  }

  // Layout + normalization for the reid model input (NCHW RGB/BGR or NHWC RGB).
  const NormSpec rns = m_reid_spec.nhwc
      ? normAB(Onnx::TensorLayout::NhwcRgb, 1.f, 0.f)
      : normMeanStd(
          bgr ? Onnx::TensorLayout::NchwBgr : Onnx::TensorLayout::NchwRgb, mean,
          stdv);

  // Affine mapping track i's (padded) box -> model-crop pixels.
  auto affineFor = [&](int i) -> Onnx::Affine {
    const auto& b = m_track_in[i].box; // normalized center form
    const Onnx::Rect box_px{
        (b.cx - b.w * 0.5f) * W, (b.cy - b.h * 0.5f) * H, b.w * W, b.h * H};
    const Onnx::ROI::Rect r = Onnx::ROI::topdownRect(box_px, mw, mh, 1.1f);
    return Onnx::ROI::rectToAffine(r, mw, mh);
  };

  auto storeRow = [&](int i, const float* row) {
    auto& e = m_track_in[i].embedding;
    e.resize(D);
    float norm = 0.f;
    for(int k = 0; k < D; ++k) { e[k] = row[k]; norm += row[k] * row[k]; }
    norm = std::sqrt(norm);
    if(norm > 1e-9f) for(auto& v : e) v /= norm;
  };

  const bool can_batch = m_reid_spec.batchable && N >= 2;
  if(can_batch)
  {
    const int CHW = 3 * mw * mh;
    m_reid_batch.resize(static_cast<size_t>(N) * CHW, boost::container::default_init);
    for(int i = 0; i < N; ++i)
      Onnx::sampleAffineToTensor(
          rns.layout, src, affineFor(i), mw, mh, rns.mean.data(),
          rns.invstd.data(), m_reid_batch.data() + static_cast<size_t>(i) * CHW);
    std::vector<int64_t> in_shape = spec.inputs[0].shape;
    if(in_shape.size() == 4) in_shape[0] = N;
    else in_shape = {N, 3, mh, mw};
    Ort::Value in0 = Onnx::vec_to_tensor<float>(m_reid_batch, in_shape);
    Ort::Value outs[4]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}};
    const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
    Ort::Value ins[1] = {std::move(in0)};
    rctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
    if(oidx < static_cast<int>(n_out) && outs[oidx].IsTensor()
       && outs[oidx].GetTensorTypeAndShapeInfo().GetElementType()
              == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
      const float* data = outs[oidx].GetTensorData<float>();
      for(int i = 0; i < N; ++i)
        storeRow(i, data + static_cast<size_t>(i) * D);
    }
  }
  else
  {
    for(int i = 0; i < N; ++i)
    {
      Onnx::FloatTensor ft = fusedAffineTensor(
          spec.inputs[0], src, affineFor(i), mw, mh, rns, m_reid_tmp);
      Ort::Value outs[4]{
          Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
          Ort::Value{nullptr}};
      const size_t n_out = std::min<size_t>(4, spec.output_names_char.size());
      Ort::Value ins[1] = {std::move(ft.value)};
      rctx.infer(spec, ins, std::span<Ort::Value>(outs, n_out));
      std::swap(m_reid_tmp, ft.storage);
      if(oidx < static_cast<int>(n_out) && outs[oidx].IsTensor()
         && outs[oidx].GetTensorTypeAndShapeInfo().GetElementType()
                == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        storeRow(i, outs[oidx].GetTensorData<float>());
    }
  }
}

} // namespace OnnxModels
