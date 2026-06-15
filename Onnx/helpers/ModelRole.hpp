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
  Animal,
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
  MultiClassDetector, // PINTO [N,7] batchno,classid,score,xyxy (Gold-YOLO etc.)
  YoloxDetector,      // raw YOLOX/COCO grid [1,A,5+C] (person/animal classes)
  RetinaFaceDetector, // priorbox loc[1,N,4]+conf[1,N,2]+landms[1,N,10], 5 kpts
  FaceBoxesDetector,  // priorbox loc[1,N,4]+conf[1,N,2] (or PINTO pre-decoded)

  // Stage-2 landmark / pose
  BlazePoseLandmark, // NHWC 256, output 124/155/195
  HandLandmark,      // NHWC, output 63 (=21*3)
  FaceMeshLandmark,  // NHWC 192, output 1404/1434
  MobileFaceNet,     // NCHW 112, output 136 (=68*2)
  SimccPose,         // NCHW, simcc_x/simcc_y
  HeatmapPose,       // NCHW, single 4-D heatmap [1,K,Hh,Wh]
  XyScoreLandmark,   // single [1,K,3] x,y,score output (Peppa-Pig, RTMPose-post)

  // Single-stage / whole-frame
  YoloPose,    // 640, single output width 5+K*3
  RtmoPose,    // 640, dets[1,N,5] + keypoints[1,N,K,3] (NMS-free)
  MoveNetPose, // single [1,1,K,3] (y,x,score) normalized, full-frame

  // Appearance ReID (loaded in its own port; one image in, one feature vector out)
  ReidEmbedder,
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
  int num_inputs = 1;    // model input count (PINTO mmpose-post adds [1,2] bbox)

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
  // Every supported family takes a rank-4 image tensor (NCHW/NHWC). Anything
  // else (audio [1,1,N], mel [1,F,T], voxel rank-5...) must classify Unknown
  // instead of pattern-matching into a pose family by coincidence.
  if(spec.inputs[0].shape.size() != 4)
    return r;
  r.num_inputs = static_cast<int>(spec.inputs.size());

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
  int heatmap_k = 0; // channel count of qualifying heatmap outputs
  for(const auto& o : spec.outputs)
  {
    OutInfo oi{&o, detail::totalSize(o.shape), 1, static_cast<int>(o.shape.size())};
    if(!o.shape.empty())
      oi.last = o.shape.back() > 0 ? o.shape.back() : 1;
    outs.push_back(oi);

    if(detail::nameContains(o.name, "simcc"))
      has_simcc = true;
    // A pose heatmap is [1,K,h,w] with a plausible keypoint count (>=5 kills
    // dbface's box/landmark planes, colorization's ab channels, SRResNet RGB),
    // BOTH spatial dims in [32,256] (kills 22x22 detector grids and NHWC
    // [1,h,w,2] text maps where the trailing channel lands in the w slot), and
    // a stride >= 4 vs the input when it is known: pose heads are input/4 —
    // an input/2 map is an encoder feature / text-region map (craft 256->128).
    // Multiple qualifying outputs must agree on K (a real pose model exports
    // one heatmap, or an aux copy with the same K — differing channel counts
    // are a detector head: heatmap + boxes + landmarks). A sibling rank-4
    // output with 1..4 channels at the same spatial size is a detector's
    // score/box plane (dbface) and disqualifies the whole model.
    if(o.shape.size() == 4 && o.shape[2] >= 32 && o.shape[2] <= 256
       && o.shape[3] >= 32 && o.shape[3] <= 256)
    {
      const auto k = o.shape[1];
      if(k >= 5 && k <= 200 && !(H > 0 && o.shape[2] * 3 > H))
      {
        if(heatmap_k == 0)
        {
          has_heatmap = true;
          heatmap_k = static_cast<int>(k);
        }
        else if(heatmap_k > 0 && heatmap_k != static_cast<int>(k))
        {
          has_heatmap = false;
          heatmap_k = -1; // poisoned: inconsistent K -> not a pose model
        }
      }
      else if(k >= 1 && k <= 4)
      {
        has_heatmap = false;
        heatmap_k = -1; // poisoned: score/box plane alongside -> detector head
      }
    }
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

  // --- A0) PriorBox face detectors (RetinaFace / FaceBoxes) ------------------
  // RetinaFace: exactly three rank-3 outputs whose last dims are {4,2,10}
  // (loc, conf, 5-pt landms). FaceBoxes: exactly two with {4,2} — or the PINTO
  // pre-decoded export, recognized by its boxes_x1y1x2y2 output name.
  {
    int last4 = 0, last2 = 0, last10 = 0, rank3 = 0;
    bool predecoded = false;
    for(auto& o : outs)
    {
      if(o.rank != 3)
        continue;
      ++rank3;
      // pre-decoded boxes must be an actual [1,N,4] tensor: the PINTO [N,7]
      // post detectors ALSO have "x1y1x2y2" in their output name.
      if(o.last == 4 && detail::nameContains(o.port->name, "x1y1x2y2"))
        predecoded = true;
      if(o.last == 4) ++last4;
      else if(o.last == 2) ++last2;
      else if(o.last == 10) ++last10;
    }
    if(num_outputs == 3 && rank3 == 3 && last4 == 1 && last2 == 1 && last10 == 1)
      return setKind(
          ModelKind::RetinaFaceDetector, ModelStage::Detector, ModelDomain::Face,
          5);
    if((num_outputs == 2 && rank3 == 2 && last4 == 1 && last2 == 1) || predecoded)
      return setKind(
          ModelKind::FaceBoxesDetector, ModelStage::Detector, ModelDomain::Face,
          0);
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

  // --- A1b) PINTO multi-class end2end detector: one [N,7] output ------------
  // (batchno, classid, score, x1,y1,x2,y2 — col order varies, read by name).
  // Used as a body/person detector (class 0). Covers Gold-YOLO / YOLOX-Body-*
  // / YOLOv9-Wholebody / DEIM-Wholebody families.
  if(num_outputs == 1 && outs[0].rank == 2 && outs[0].last == 7)
    return setKind(
        ModelKind::MultiClassDetector, ModelStage::Detector,
        ModelDomain::Body, 0);

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

  // --- A3) Direct xy-score landmark: a single [1,K,3] output ------------------
  // Peppa-Pig face landmarks ([1,98,3], normalized) and the PINTO mmpose
  // "with_post" exports ([1,21,3] crop pixels + a [1,2] bbox input). The
  // single-output requirement keeps MediaPipe hand landmark exports shaped
  // [1,21,3] (x,y,z + flag/handedness siblings) on the HandLandmark path below.
  if(num_outputs == 1 && outs[0].rank == 3)
  {
    const auto& s = outs[0].port->shape;
    if(s[1] >= 5 && s[1] <= 200 && s[2] == 3)
    {
      const int nk = static_cast<int>(s[1]);
      ModelDomain dom = ModelDomain::Body;
      if(nk == 21)
        dom = ModelDomain::Hand;
      else if(nk == 68 || nk == 98 || nk == 106)
        dom = ModelDomain::Face;
      return setKind(ModelKind::XyScoreLandmark, ModelStage::Landmark, dom, nk);
    }
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
    // SimCC heads are exactly [1,K,Sx] / [1,K,Sy] — rank 3. A pair of rank-4
    // [1,K,h,w] outputs with matching K is a heatmap model (fashionai), which
    // the heatmap rule below decodes correctly.
    if(s0.size() != 3 || s1.size() != 3)
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
    // 21 -> hand; a SQUARE 17-kpt SimCC (256x256) is RTMPose-Animal (AP10K),
    // whereas human body RTMPose is 256x192. A SQUARE SimCC with a face-landmark
    // count (RTMPose-Face is 106 = LaPa; also 68/98) is a face model. Otherwise
    // body (17/26/133).
    ModelDomain dom = ModelDomain::Body;
    if(nk == 21)
      dom = ModelDomain::Hand;
    else if(nk == 17 && W > 0 && W == H)
      dom = ModelDomain::Animal;
    else if(W > 0 && W == H && (nk == 106 || nk == 98 || nk == 68))
      dom = ModelDomain::Face;
    return setKind(ModelKind::SimccPose, ModelStage::Landmark, dom, nk);
  }

  // --- D) Heatmap pose (ViTPose / HRNet) -----------------------------------
  if(has_heatmap)
  {
    const int nk = heatmap_k > 0 ? heatmap_k : 17;
    // 21 -> hand; a face-landmark count (68 = FAN/300W, 98 = WFLW, 106 = LaPa)
    // is a face-alignment net (2DFAN); SQUARE input with an animal keypoint
    // count is AP-10K (17) or Animal-Pose (20) — human-body HRNets are
    // 256x192, never square.
    ModelDomain dom = ModelDomain::Body;
    if(nk == 21)
      dom = ModelDomain::Hand;
    else if(nk == 68 || nk == 98 || nk == 106)
      dom = ModelDomain::Face;
    else if((nk == 17 || nk == 20) && W > 0 && W == H)
      dom = ModelDomain::Animal;
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
  // YOLO-family inputs are >= 320 square-ish; a small input with a [1,T,F]
  // output is a ViT/CRNN sequence head (BLIP-2 vision 224 -> [1,33,2560],
  // easyocr 64x128 -> [1,31,6719]) — never a detector.
  const bool yolo_sized = W >= 256 && H >= 256;
  if(num_outputs == 1 && outs[0].rank == 3 && yolo_sized)
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
    // Not a pose head: a big-grid [1,A,5+C] is a raw YOLOX object detector
    // (e.g. COCO 85 = 4+1+80). Used as a person/animal detector.
    const int A = static_cast<int>(s[1]), F = static_cast<int>(s[2]);
    const int grid = std::max(A, F), feat = std::min(A, F);
    if(grid >= 1000 && feat >= 6 && feat <= 100)
      return setKind(
          ModelKind::YoloxDetector, ModelStage::Detector, ModelDomain::Unknown,
          0);
  }

  // --- F2) MoveNet: a single [1,1,K,3] (y,x,score) normalized output, decoded
  // full-frame. Checked before the resolution fallbacks: its 192/256 NHWC input
  // would otherwise mis-tag as FaceMesh/BlazePose landmark.
  if(num_outputs == 1 && outs[0].rank == 4)
  {
    const auto& s = outs[0].port->shape;
    if(s[0] == 1 && s[1] == 1 && s[2] >= 5 && s[2] <= 200 && s[3] == 3)
      return setKind(
          ModelKind::MoveNetPose, ModelStage::Landmark, ModelDomain::Body,
          static_cast<int>(s[2]));
  }

  // --- G) Fallbacks by input resolution ------------------------------------
  // The resolution fallbacks only make sense if some output actually carries a
  // landmark payload (hand 63 .. facemesh 1434, generously [60, 4096]). A lone
  // [1,256,256,1] mask (Selfie-Segmentation) or three Euler scalars
  // (head-pose nets) at MediaPipe-ish input sizes must stay Unknown.
  bool landmark_payload = false;
  for(auto& o : outs)
    if(o.total >= 60 && o.total <= 4096)
      landmark_payload = true;
  if(r.nhwc && landmark_payload)
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

// --- Generic ReID / appearance-embedding model --------------------------------
// One image input -> one feature vector. Geometry (layout, size, embed dim,
// batchability) is read from the ONNX shapes here; channel order + normalization
// are NOT inferable from the graph and are chosen by a separate preprocessing
// control (see PoseDetector's "Re-ID Preprocess"). Pair-comparison exports
// (two image inputs, or a "similarit*" output) are rejected.
struct ReidSpec
{
  bool valid = false;
  bool nhwc = false;
  int in_w = 0, in_h = 0;
  int embed_dim = 0;      // flattened non-batch size of the feature output
  int out_index = 0;      // which output is the feature vector
  bool batchable = false; // input batch dim dynamic (or unspecified)
};

inline ReidSpec classifyReid(const ModelIO& spec)
{
  ReidSpec rs;
  if(spec.inputs.empty() || spec.outputs.empty())
    return rs;

  // Exactly one image-like input (reject base+target pair-comparison variants).
  int image_inputs = 0, img_idx = -1;
  for(int i = 0; i < static_cast<int>(spec.inputs.size()); ++i)
  {
    const auto& s = spec.inputs[i].shape;
    if(s.size() == 4 && (s[1] == 3 || s[3] == 3))
    {
      ++image_inputs;
      if(img_idx < 0)
        img_idx = i;
    }
  }
  if(image_inputs != 1)
    return rs;

  const auto& in = spec.inputs[img_idx].shape;
  if(in[3] == 3)
  {
    rs.nhwc = true;
    rs.in_h = static_cast<int>(in[1]);
    rs.in_w = static_cast<int>(in[2]);
  }
  else
  {
    rs.nhwc = false;
    rs.in_h = static_cast<int>(in[2]);
    rs.in_w = static_cast<int>(in[3]);
  }
  rs.batchable = (in[0] <= 0);

  // Pick the feature output: largest non-batch flatten in a plausible embed
  // range. Any "similarit*" output marks a pair-comparison model -> reject.
  int best_idx = -1;
  int64_t best_dim = 0;
  for(int i = 0; i < static_cast<int>(spec.outputs.size()); ++i)
  {
    const auto& o = spec.outputs[i];
    if(detail::nameContains(o.name, "similar"))
      return rs;
    int64_t d = 1;
    for(size_t k = 1; k < o.shape.size(); ++k)
      if(o.shape[k] > 0)
        d *= o.shape[k];
    if(o.shape.size() >= 2 && d >= 32 && d <= 8192 && d > best_dim)
    {
      best_dim = d;
      best_idx = i;
    }
  }
  if(best_idx < 0)
    return rs;

  rs.out_index = best_idx;
  rs.embed_dim = static_cast<int>(best_dim);
  // Many ReID exports (centroids-reid, OSNet, FastReID) declare a DYNAMIC input
  // size [batch,3,?,?]. That's still usable — fall back to the de-facto person-
  // ReID crop 256x128 (HxW) rather than rejecting the model (which silently
  // disables appearance tracking). Fixed-size exports keep their declared dims.
  if(rs.in_h <= 0)
    rs.in_h = 256;
  if(rs.in_w <= 0)
    rs.in_w = 128;
  rs.valid = (rs.in_w > 0 && rs.in_h > 0);
  return rs;
}

} // namespace Onnx
