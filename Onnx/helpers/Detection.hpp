#pragma once
#include <Onnx/helpers/OnnxBase.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <QPointF>
#include <QRectF>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

// Generic MediaPipe-style SSD-anchor detector decode.
//
// The three MediaPipe detectors (pose_detection, palm_detection, BlazeFace)
// are the SAME decoder with different constants — this consolidates them.
// See TWO_STAGE_ARCHITECTURE.md §2a for the parameter table.
namespace Onnx::Detection
{

struct Detection
{
  float xc{}, yc{}, w{}, h{};       // center-form, normalized [0,1] in model square
  float score{};                    // [0,1]
  int class_id{-1};                 // detector class id; -1 = none/unknown
                                    // (SSD/single-class leave it; multi-class set it)
  std::vector<QPointF> keypoints;   // alignment keypoints, normalized [0,1]

  QRectF box() const noexcept
  {
    return QRectF(xc - w * 0.5f, yc - h * 0.5f, w, h);
  }
};

struct SsdParams
{
  int input_size = 128;
  std::vector<int> strides; // e.g. {8,16,32,32,32}; duplicates = merged layers
  int anchors_per_cell = 2; // aspect_ratios(1) + interpolated_scale(1)
  float anchor_offset = 0.5f;
};

// Canonical SsdAnchorsCalculator anchor generation (fixed_anchor_size=true).
// Same-stride layers are merged so their anchors are emitted consecutively per
// cell — matching the model's output ordering. Returns anchor centers in [0,1].
inline std::vector<QPointF> generateAnchors(const SsdParams& p)
{
  std::vector<QPointF> anchors;
  const int n = static_cast<int>(p.strides.size());
  int i = 0;
  while(i < n)
  {
    const int stride = p.strides[i];
    int repeats = 0, j = i;
    while(j < n && p.strides[j] == stride)
    {
      ++repeats;
      ++j;
    }
    const int fm = (p.input_size + stride - 1) / stride; // ceil
    const int per_cell = p.anchors_per_cell * repeats;
    for(int y = 0; y < fm; ++y)
      for(int x = 0; x < fm; ++x)
      {
        const float cx = (x + p.anchor_offset) / fm;
        const float cy = (y + p.anchor_offset) / fm;
        for(int a = 0; a < per_cell; ++a)
          anchors.push_back(QPointF(cx, cy));
      }
    i = j;
  }
  return anchors;
}

// Decode raw model outputs (any number of score/box tensors, concatenated in
// output order) into detections. Boxes are encoded as
// [dx, dy, dw, dh, kp0x, kp0y, ...] relative to anchor centers, scaled by
// input_size (the MediaPipe convention with fixed unit anchors).
inline std::vector<Detection> decode(
    std::span<Ort::Value> outputs, const SsdParams& params, float score_thr)
{
  struct Buf
  {
    const float* p;
    int n; // anchors
    int c; // channels (last dim)
  };
  std::vector<Buf> scoreBufs, boxBufs;
  for(auto& o : outputs)
  {
    if(!o.IsTensor())
      continue;
    auto shape = o.GetTensorTypeAndShapeInfo().GetShape();
    if(shape.size() != 3)
      continue;
    const int n = static_cast<int>(shape[1]);
    const int c = static_cast<int>(shape[2]);
    const float* data = o.GetTensorData<float>();
    if(c == 1)
      scoreBufs.push_back({data, n, c});
    else
      boxBufs.push_back({data, n, c});
  }
  if(scoreBufs.empty() || boxBufs.empty())
    return {};

  const auto anchors = generateAnchors(params);
  const int total = static_cast<int>(anchors.size());
  const int coords = boxBufs[0].c;
  const int num_kp = std::max(0, (coords - 4) / 2);
  const float inv = 1.0f / params.input_size;

  auto scoreAt = [&](int idx) -> float {
    for(auto& b : scoreBufs)
    {
      if(idx < b.n)
        return b.p[idx];
      idx -= b.n;
    }
    return -1e30f;
  };
  auto boxAt = [&](int idx) -> const float* {
    for(auto& b : boxBufs)
    {
      if(idx < b.n)
        return b.p + static_cast<size_t>(idx) * b.c;
      idx -= b.n;
    }
    return nullptr;
  };

  std::vector<Detection> dets;
  for(int i = 0; i < total; ++i)
  {
    const float raw = std::clamp(scoreAt(i), -100.0f, 100.0f);
    const float s = sigmoid(raw);
    if(s < score_thr)
      continue;
    const float* box = boxAt(i);
    if(!box)
      continue;
    const auto& a = anchors[i];

    Detection d;
    d.score = s;
    d.xc = a.x() + box[0] * inv;
    d.yc = a.y() + box[1] * inv;
    d.w = box[2] * inv;
    d.h = box[3] * inv;
    d.keypoints.reserve(num_kp);
    for(int k = 0; k < num_kp; ++k)
      d.keypoints.push_back(QPointF(
          a.x() + box[4 + 2 * k] * inv, a.y() + box[4 + 2 * k + 1] * inv));
    dets.push_back(std::move(d));
  }
  return dets;
}

// Greedy IoU non-maximum suppression.
inline std::vector<Detection>
nms(std::vector<Detection> dets, float iou_threshold = 0.3f)
{
  if(dets.empty())
    return {};
  std::sort(dets.begin(), dets.end(), [](const auto& a, const auto& b) {
    return a.score > b.score;
  });

  auto iou = [](const Detection& a, const Detection& b) -> float {
    const float x1 = std::max(a.xc - a.w * 0.5f, b.xc - b.w * 0.5f);
    const float y1 = std::max(a.yc - a.h * 0.5f, b.yc - b.h * 0.5f);
    const float x2 = std::min(a.xc + a.w * 0.5f, b.xc + b.w * 0.5f);
    const float y2 = std::min(a.yc + a.h * 0.5f, b.yc + b.h * 0.5f);
    const float iw = std::max(0.0f, x2 - x1);
    const float ih = std::max(0.0f, y2 - y1);
    const float inter = iw * ih;
    const float uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
  };

  std::vector<Detection> out;
  std::vector<bool> dead(dets.size(), false);
  for(size_t i = 0; i < dets.size(); ++i)
  {
    if(dead[i])
      continue;
    out.push_back(dets[i]);
    for(size_t j = i + 1; j < dets.size(); ++j)
      if(!dead[j] && iou(dets[i], dets[j]) > iou_threshold)
        dead[j] = true;
  }
  return out;
}

// --- Presets -------------------------------------------------------------
// pose_detection: 224 -> {8,16,32,32,32} = 2254 anchors;
//                 128 -> {8,16,16,16}   = 896 anchors. 4 alignment kpts.
inline SsdParams blazePoseParams(int input_size = 224)
{
  if(input_size <= 128)
    return SsdParams{.input_size = 128, .strides = {8, 16, 16, 16}};
  return SsdParams{.input_size = input_size, .strides = {8, 16, 32, 32, 32}};
}
// palm_detection (full): 192, strides {8,16,16,16} -> 2016 anchors, 7 kpts
inline SsdParams palmParams(int input_size = 192)
{
  return SsdParams{.input_size = input_size, .strides = {8, 16, 16, 16}};
}
// BlazeFace: 896 anchors on grids 16 & 8, so the strides scale with the input:
//   front 128 -> {8,16,16,16}   (grids 16, 8)
//   back  256 -> {16,32,32,32}  (grids 16, 8)
// Hard-coding {8,16,16,16} at 256 would generate 3584 anchors (grids 32,16) and
// misalign the decode -> boxes drift to the top of the frame.
inline SsdParams blazeFaceParams(int input_size = 128)
{
  const int s0 = std::max(1, input_size / 16);
  const int s1 = std::max(1, input_size / 8);
  return SsdParams{.input_size = input_size, .strides = {s0, s1, s1, s1}};
}

// --- End2end (mmdeploy YOLOX / RTMDet) detector decode -------------------
// Outputs: dets [1,N,5] (x1,y1,x2,y2,score) in input-pixel space + labels
// [1,N]. Boxes are returned center-form, normalized by input_size (so the
// caller's letterbox removal maps them to the image). keep_label<0 keeps all.
inline std::vector<Detection> decodeEnd2End(
    std::span<Ort::Value> outputs, int input_size, int keep_label,
    float score_thr)
{
  const float* dets = nullptr;
  const float* labels_f = nullptr;
  const int64_t* labels_i = nullptr;
  int n = 0;
  for(auto& o : outputs)
  {
    if(!o.IsTensor())
      continue;
    auto info = o.GetTensorTypeAndShapeInfo();
    auto sh = info.GetShape();
    if(sh.size() == 3 && sh[2] == 5)
    {
      dets = o.GetTensorData<float>();
      n = static_cast<int>(sh[1]);
    }
    else if(sh.size() == 2)
    {
      if(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        labels_i = o.GetTensorData<int64_t>();
      else
        labels_f = o.GetTensorData<float>();
    }
  }
  if(!dets)
    return {};

  const float inv = 1.0f / input_size;
  std::vector<Detection> out;
  for(int i = 0; i < n; ++i)
  {
    const float* d = dets + i * 5;
    const float score = d[4];
    if(score < score_thr)
      continue;
    const int lbl
        = labels_i ? static_cast<int>(labels_i[i])
                   : (labels_f ? static_cast<int>(labels_f[i]) : 0);
    if(keep_label >= 0 && lbl != keep_label)
      continue;
    const float x1 = d[0] * inv, y1 = d[1] * inv, x2 = d[2] * inv,
                y2 = d[3] * inv;
    Detection det;
    det.score = score;
    det.class_id = lbl;
    det.xc = (x1 + x2) * 0.5f;
    det.yc = (y1 + y2) * 0.5f;
    det.w = x2 - x1;
    det.h = y2 - y1;
    out.push_back(std::move(det));
  }
  return out;
}

// PINTO multi-class end2end detector: a single [N,7] output. Columns are
// batchno, classid, and (score,x1,y1,x2,y2) in an order that varies by export
// — pass score_before_box=true for "..._score_x1y1x2y2", false for
// "..._x1y1x2y2_score". Boxes are in input-pixel space; returned normalized by
// (input_w,input_h), center-form. keep_class<0 keeps all classes.
inline std::vector<Detection> decodeMultiClass(
    std::span<Ort::Value> outputs, int input_w, int input_h, int keep_class,
    float score_thr, bool score_before_box)
{
  const float* d = nullptr;
  int n = 0;
  for(auto& o : outputs)
  {
    if(!o.IsTensor())
      continue;
    auto sh = o.GetTensorTypeAndShapeInfo().GetShape();
    if(sh.size() == 2 && sh[1] == 7)
    {
      d = o.GetTensorData<float>();
      n = static_cast<int>(sh[0]);
    }
  }
  if(!d)
    return {};

  const int c_score = score_before_box ? 2 : 6;
  const int c_x1 = score_before_box ? 3 : 2;
  const float iw = 1.0f / input_w, ih = 1.0f / input_h;

  std::vector<Detection> out;
  for(int i = 0; i < n; ++i)
  {
    const float* r = d + i * 7;
    const float score = r[c_score];
    if(score < score_thr)
      continue;
    if(keep_class >= 0 && static_cast<int>(r[1] + 0.5f) != keep_class)
      continue;
    const float x1 = r[c_x1] * iw, y1 = r[c_x1 + 1] * ih,
                x2 = r[c_x1 + 2] * iw, y2 = r[c_x1 + 3] * ih;
    Detection det;
    det.score = score;
    det.class_id = static_cast<int>(r[1] + 0.5f);
    det.xc = (x1 + x2) * 0.5f;
    det.yc = (y1 + y2) * 0.5f;
    det.w = x2 - x1;
    det.h = y2 - y1;
    out.push_back(std::move(det));
  }
  return out;
}

// Raw YOLOX grid decode (COCO): single [1,A,5+C] output (cx,cy,w,h,obj,classes)
// with NO NMS baked. Anchors are the concatenated strides 8/16/32 grids. Boxes
// are grid/exp-decoded to input-pixel space, returned normalized (in_w,in_h),
// center-form. Keeps detections whose argmax class is in [cls_lo,cls_hi]
// (person=0; COCO animals 14..23). obj/cls are assumed already sigmoided
// (Megvii/rtmlib convention). Followed by NMS.
inline std::vector<Detection> decodeYoloxGrid(
    std::span<Ort::Value> outputs, int in_w, int in_h, int cls_lo, int cls_hi,
    float score_thr, float nms_thr)
{
  const float* d = nullptr;
  int A = 0, F = 0;
  int64_t total = 0;
  for(auto& o : outputs)
  {
    if(!o.IsTensor())
      continue;
    auto info = o.GetTensorTypeAndShapeInfo();
    auto sh = info.GetShape();
    if(sh.size() == 3)
    {
      A = static_cast<int>(sh[1]);
      F = static_cast<int>(sh[2]);
      total = static_cast<int64_t>(info.GetElementCount());
      d = o.GetTensorData<float>();
    }
  }
  if(!d || F < 6)
    return {};
  // Clamp the anchor count to what the buffer actually holds (a dynamic/negative
  // A, or a declared A*F larger than the real element count, would over-read).
  if(A < 0 || static_cast<int64_t>(A) * F > total)
    A = static_cast<int>(total / F);
  const int C = F - 5;

  std::vector<Detection> dets;
  const int strides[3] = {8, 16, 32};
  int idx = 0;
  for(int s : strides)
  {
    const int gh = in_h / s, gw = in_w / s;
    for(int gy = 0; gy < gh; ++gy)
      for(int gx = 0; gx < gw; ++gx)
      {
        if(idx >= A)
          break;
        const float* r = d + static_cast<size_t>(idx) * F;
        ++idx;
        const float obj = r[4];
        int bc = 0;
        float bcs = r[5];
        for(int c = 1; c < C; ++c)
          if(r[5 + c] > bcs)
          {
            bcs = r[5 + c];
            bc = c;
          }
        const float score = obj * bcs;
        if(score < score_thr || bc < cls_lo || bc > cls_hi)
          continue;
        const float cx = (r[0] + gx) * s, cy = (r[1] + gy) * s;
        const float w = std::exp(r[2]) * s, h = std::exp(r[3]) * s;
        Detection det;
        det.score = score;
        det.class_id = bc;
        det.xc = cx / in_w;
        det.yc = cy / in_h;
        det.w = w / in_w;
        det.h = h / in_h;
        dets.push_back(std::move(det));
      }
  }
  return nms(std::move(dets), nms_thr);
}

} // namespace Onnx::Detection
