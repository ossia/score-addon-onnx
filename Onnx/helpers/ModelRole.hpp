#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

namespace Onnx
{
// What a model does in a (possibly) two-stage pose pipeline.
enum class ModelStage : uint8_t
{
  Unknown,
  Detector,    // Stage 1: full-frame bbox (+ alignment keypoints)
  Landmark,    // Stage 2: runs on a cropped ROI, outputs keypoints
  SingleStage, // Detect + keypoints in one pass (YOLO-pose / RTMO)
};

enum class ModelDomain : uint8_t
{
  Unknown,
  Body,
  Hand,
  Face,
};

// Concrete model family, derived from input/output signature.
enum class ModelKind : uint8_t
{
  Unknown,

  // Stage-1 detectors
  BlazePoseDetector, // NHWC/NCHW, anchors, 4 alignment kpts (coords 12)
  PalmDetector,      // NHWC, anchors, 7 kpts (coords 18)
  BlazeFaceDetector, // NHWC, anchors, 6 kpts (coords 16)
  PersonDetector,    // YOLOX / RTMDet end2end (dets[1,N,5] + labels[1,N])

  // Stage-2 landmark / pose
  BlazePoseLandmark, // NHWC 256, output 124/155/195
  HandLandmark,      // NHWC, output 63 (=21*3)
  FaceMeshLandmark,  // NHWC 192, output 1404/1434
  MobileFaceNet,     // NCHW 112, output 136 (=68*2)
  SimccPose,         // NCHW, simcc_x/simcc_y
  HeatmapPose,       // NCHW, single 4-D heatmap [1,K,Hh,Wh]

  // Single-stage
  YoloPose, // 640, single output width 5+K*3
  RtmoPose, // 640, dets[1,N,5] + keypoints[1,N,K,3] (NMS-free)
};

struct ModelRole
{
  ModelKind kind = ModelKind::Unknown;
  ModelStage stage = ModelStage::Unknown;
  ModelDomain domain = ModelDomain::Unknown;
  bool nhwc = false;
  int input_w = 0;
  int input_h = 0;
  int num_keypoints = 0; // landmark/pose keypoint count when known

  bool valid() const noexcept { return kind != ModelKind::Unknown; }
};

// Lightweight, dependency-free view of a model's I/O. Both PoseDetector (from
// Onnx::ModelSpec) and the validation harness (from Ort directly) build one of
// these, so classify() stays free of ossia/Qt.
struct ModelIO
{
  struct Port
  {
    std::string name;
    std::vector<int64_t> shape;
  };
  std::vector<Port> inputs;
  std::vector<Port> outputs;
};

namespace detail
{
inline int64_t totalSize(const std::vector<int64_t>& shape)
{
  int64_t s = 1;
  for(auto d : shape)
    if(d > 0)
      s *= d;
  return s;
}
inline bool nameContains(const std::string& s, const char* sub)
{
  std::string a = s, b = sub;
  std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return a.find(b) != std::string::npos;
}
} // namespace detail

// Classify an ONNX model purely from its declared input/output shapes & names.
// See TWO_STAGE_ARCHITECTURE.md §5 for the decision table.
inline ModelRole classify(const ModelIO& spec)
{
  ModelRole r;
  if(spec.inputs.empty())
    return r;

  // --- Input layout & resolution ---
  const auto& in = spec.inputs[0].shape;
  if(in.size() == 4)
  {
    if(in[3] == 3 || in[3] == 4)
    {
      r.nhwc = true;
      r.input_h = static_cast<int>(in[1]);
      r.input_w = static_cast<int>(in[2]);
    }
    else // assume NCHW [N,C,H,W]
    {
      r.nhwc = false;
      r.input_h = static_cast<int>(in[2]);
      r.input_w = static_cast<int>(in[3]);
    }
  }
  const int W = r.input_w, H = r.input_h;

  // --- Output summary ---
  struct OutInfo
  {
    const ModelIO::Port* port;
    int64_t total;
    int64_t last; // last dim (>0)
    int rank;
  };
  std::vector<OutInfo> outs;
  outs.reserve(spec.outputs.size());
  bool has_simcc = false;
  bool has_heatmap = false;
  for(const auto& o : spec.outputs)
  {
    OutInfo oi{&o, detail::totalSize(o.shape), 1, static_cast<int>(o.shape.size())};
    if(!o.shape.empty())
      oi.last = o.shape.back() > 0 ? o.shape.back() : 1;
    outs.push_back(oi);

    if(detail::nameContains(o.name, "simcc"))
      has_simcc = true;
    if(o.shape.size() == 4 && o.shape[1] > 1 && o.shape[1] <= 200
       && o.shape[2] > 1 && o.shape[2] <= 256 && o.shape[3] > 1
       && o.shape[3] <= 256)
      has_heatmap = true;
  }
  const int num_outputs = static_cast<int>(outs.size());

  auto anyTotal = [&](int64_t v) {
    for(auto& o : outs)
      if(o.total == v)
        return true;
    return false;
  };

  auto setKind = [&](ModelKind k, ModelStage s, ModelDomain d, int nk) {
    r.kind = k;
    r.stage = s;
    r.domain = d;
    r.num_keypoints = nk;
    return r;
  };

  // --- A) SSD-anchor detectors (layout-agnostic) ---------------------------
  {
    bool has_scores = false;
    int coords = 0;
    for(auto& o : outs)
    {
      if(o.rank == 3 && o.last == 1)
        has_scores = true;
      else if(o.rank == 3 && (o.last == 12 || o.last == 16 || o.last == 18))
        coords = static_cast<int>(o.last);
    }
    if(has_scores && coords != 0)
    {
      switch(coords)
      {
        case 12:
          return setKind(
              ModelKind::BlazePoseDetector, ModelStage::Detector,
              ModelDomain::Body, 0);
        case 16:
          return setKind(
              ModelKind::BlazeFaceDetector, ModelStage::Detector,
              ModelDomain::Face, 0);
        case 18:
          return setKind(
              ModelKind::PalmDetector, ModelStage::Detector, ModelDomain::Hand,
              0);
      }
    }
  }

  // --- A1) RTMO single-stage: rank-4 "keypoints" [1,N,K,3] + rank-3 dets -----
  // (shapes are usually dynamic, so match on name + rank.)
  {
    bool has_kpts4 = false, has_dets3 = false;
    int nk = 17;
    for(auto& o : outs)
    {
      if(o.rank == 4
         && (detail::nameContains(o.port->name, "keypoint") || o.last == 3))
      {
        has_kpts4 = true;
        if(o.port->shape.size() == 4 && o.port->shape[2] > 0)
          nk = static_cast<int>(o.port->shape[2]);
      }
      else if(o.rank == 3)
        has_dets3 = true;
    }
    if(has_kpts4 && has_dets3)
      return setKind(
          ModelKind::RtmoPose, ModelStage::SingleStage, ModelDomain::Body, nk);
  }

  // --- A2) End2end person/object detector: dets [1,N,5] + labels [1,N] ------
  {
    bool has_dets = false, has_labels = false;
    for(auto& o : outs)
    {
      if(o.rank == 3 && o.last == 5)
        has_dets = true;
      else if(o.rank == 2)
        has_labels = true;
    }
    if(has_dets && has_labels)
      return setKind(
          ModelKind::PersonDetector, ModelStage::Detector, ModelDomain::Unknown,
          0);
  }

  // --- B) Landmark regression (magic output sizes) -------------------------
  if(anyTotal(1404) || anyTotal(1434))
    return setKind(
        ModelKind::FaceMeshLandmark, ModelStage::Landmark, ModelDomain::Face,
        anyTotal(1434) ? 478 : 468);
  if(anyTotal(63))
    return setKind(
        ModelKind::HandLandmark, ModelStage::Landmark, ModelDomain::Hand, 21);
  // BlazePose landmark variants: full-body 195 (=39*5), upper-body 155 (=31*5)
  // or 124 (=31*4). Keypoints = total / (5 or 4).
  for(int64_t v : {int64_t(195), int64_t(155), int64_t(124)})
    if(anyTotal(v))
    {
      const int stride = (v % 5 == 0) ? 5 : 4;
      return setKind(
          ModelKind::BlazePoseLandmark, ModelStage::Landmark,
          ModelDomain::Body, static_cast<int>(v / stride));
    }

  // --- C) SimCC pose (RTMPose / DWPose / RTMW) -----------------------------
  auto trySimcc = [&]() -> bool {
    if(num_outputs < 2)
      return false;
    const auto& s0 = outs[0].port->shape;
    const auto& s1 = outs[1].port->shape;
    if(s0.size() < 3 || s1.size() < 3)
      return false;
    if(s0[1] != s1[1] || s0[1] <= 0)
      return false;
    if(s0.back() < 32 || s1.back() < 32)
      return false;
    return true;
  };
  if(has_simcc || trySimcc())
  {
    int nk = 17;
    for(auto& o : outs)
      if(o.rank >= 3 && o.port->shape[1] > 0)
      {
        nk = static_cast<int>(o.port->shape[1]);
        break;
      }
    ModelDomain dom = (nk == 21) ? ModelDomain::Hand : ModelDomain::Body;
    return setKind(ModelKind::SimccPose, ModelStage::Landmark, dom, nk);
  }

  // --- D) Heatmap pose (ViTPose / HRNet) -----------------------------------
  if(has_heatmap)
  {
    int nk = 17;
    for(auto& o : outs)
      if(o.rank == 4 && o.port->shape[1] > 0)
      {
        nk = static_cast<int>(o.port->shape[1]);
        break;
      }
    ModelDomain dom = (nk == 21) ? ModelDomain::Hand : ModelDomain::Body;
    return setKind(ModelKind::HeatmapPose, ModelStage::Landmark, dom, nk);
  }

  // --- E) MobileFaceNet (68 landmarks) -------------------------------------
  if(anyTotal(136))
    return setKind(
        ModelKind::MobileFaceNet, ModelStage::Landmark, ModelDomain::Face, 68);
  for(auto& o : outs)
    if(o.rank >= 2 && o.port->shape.size() >= 2 && o.port->shape[1] == 68)
      return setKind(
          ModelKind::MobileFaceNet, ModelStage::Landmark, ModelDomain::Face,
          68);

  // --- F) Single-stage YOLO-pose -------------------------------------------
  // Two row/feature formats:
  //   v8/v11: feature dim 5 + K*3 (cxcywh, conf, kpts), anchor-free grid.
  //   yolo26/v5/v7: feature dim 6 + K*3 (xyxy, conf, class, kpts).
  if(num_outputs == 1 && outs[0].rank == 3)
  {
    const auto& s = outs[0].port->shape;
    for(int d : {static_cast<int>(s[1]), static_cast<int>(s[2])})
    {
      int nk = 0;
      if(d >= 8 && (d - 5) % 3 == 0)
        nk = (d - 5) / 3;
      else if(d >= 8 && (d - 6) % 3 == 0)
        nk = (d - 6) / 3;
      if(nk >= 5 && nk <= 60)
        return setKind(
            ModelKind::YoloPose, ModelStage::SingleStage, ModelDomain::Body,
            nk);
    }
  }

  // --- G) Fallbacks by input resolution ------------------------------------
  if(r.nhwc)
  {
    if(W == 128 && H == 128)
      return setKind(
          ModelKind::BlazeFaceDetector, ModelStage::Detector,
          ModelDomain::Face, 0);
    if(W == 192 && H == 192)
      return setKind(
          ModelKind::FaceMeshLandmark, ModelStage::Landmark, ModelDomain::Face,
          468);
    if(W == 224 && H == 224)
      return setKind(
          ModelKind::HandLandmark, ModelStage::Landmark, ModelDomain::Hand,
          21);
    if(W == 256 && H == 256)
      return setKind(
          ModelKind::BlazePoseLandmark, ModelStage::Landmark,
          ModelDomain::Body, 33);
  }
  return r; // Unknown
}

} // namespace Onnx
