// Pose Detector detection stage on real models and images. Each test SKIPs
// when its model or image is missing.
#include <tests/AllocCounter.hpp>
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/PoseDetector_internal.hpp>

#include <QFile>
#include <QImage>
#include <QPainter>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace
{
const std::string presets = TestPaths::models() + "/pose-detector";
const std::string package = TestPaths::posePackage();
const std::string wild2 = TestPaths::wild() + "/wild2";
const std::string body_jpg = TestPaths::images() + "/body.jpg";
const std::string hand_jpg
    = TestPaths::ailia() + "/hand_recognition/blazehand/person_hand.jpg";

struct Harness
{
  OnnxModels::PoseDetector node;
  QImage image;
  // The ports hold views: keep the names and bytes alive.
  std::string det_name, det_bytes, lm_name, lm_bytes;

  Harness(const std::string& img, const std::string& det, const std::string& lm = {})
      : image{QImage(QString::fromStdString(img)).convertToFormat(QImage::Format_RGBA8888)}
      , det_name{det}
      , det_bytes{TestPaths::slurp(det)}
      , lm_name{lm}
      , lm_bytes{lm.empty() ? std::string{} : TestPaths::slurp(lm)}
  {
    node.inputs.det_model.file.bytes = det_bytes;
    node.inputs.det_model.file.filename = det_name;
    node.inputs.model.file.bytes = lm_bytes;
    node.inputs.model.file.filename = lm_name;
    inlineWorker(node);
  }

  // The Body Model port (a view too). The worker reads the file: a name that
  // is not a file on disk is written to a temporary one.
  std::string body_name, body_bytes;
  void setBodyModel(std::string name, std::string bytes)
  {
    if(!name.empty() && !std::filesystem::exists(name))
    {
      static const auto dir = [] {
        auto d = std::filesystem::temp_directory_path()
                 / ("score-onnx-pose-" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(d);
        return d;
      }();
      const auto path = (dir / name).string();
      std::ofstream(path, std::ios::binary).write(bytes.data(), bytes.size());
      name = path;
    }
    body_name = std::move(name);
    body_bytes = std::move(bytes);
    node.inputs.body_model.file.bytes = body_bytes;
    node.inputs.body_model.file.filename = body_name;
  }

  void run()
  {
    auto& t = node.inputs.image.texture;
    t.bytes = image.bits();
    t.width = image.width();
    t.height = image.height();
    t.changed = true;
    node();
  }
};
}

TEST_CASE("Gold-YOLO head detector gets [0,1] input", "[onnx][pose][model]")
{
  const auto gold
      = TestPaths::libreonnx() + "/423_6DRepNet360/gold_yolo_n_head_post_0277_0.5071_1x3x480x640.onnx";
  const auto bhh = presets + "/detectors/det-bhh-yolox-n-480x640.onnx";
  if(!TestPaths::haveAll({gold, bhh, body_jpg}))
    SKIP("models or image not found");

  // Raw 0-255 saturates it: every row scores 1.0, with inverted boxes.
  Harness h{body_jpg, gold};
  h.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
  h.run();
  const auto& heads = h.node.outputs.poses.value;
  REQUIRE(!heads.empty());
  for(const auto& p : heads)
  {
    CHECK(p.mean_confidence < 0.99f);
    CHECK(p.box.w > 0.f);
    CHECK(p.box.h > 0.f);
    CHECK(p.box.w < 0.2f); // a head, not the frame
  }

  // The YOLOX body/head/hand export still takes raw 0-255.
  Harness b{body_jpg, bhh};
  b.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
  b.node.inputs.detection_class.value = 0;
  b.run();
  REQUIRE(!b.node.outputs.poses.value.empty());
  CHECK(b.node.outputs.poses.value.front().mean_confidence > 0.7f);
}

TEST_CASE("YOLOX COCO detectors: both input ranges find the players", "[onnx][pose][model]")
{
  // ailia's yolox_*.opt take raw 0-255, the PINTO exports [0,1].
  for(const auto& det :
      {wild2 + "/pose2__yolox__yolox_tiny.opt.onnx",
       wild2 + "/pose2__yolox__yolox_s.opt.onnx",
       package + "/detectors/det-coco-yolox-416.onnx",
       presets + "/detectors/det-coco-yolox-nano-480x640.onnx"})
  {
    DYNAMIC_SECTION(det)
    {
      if(!TestPaths::haveAll({det, body_jpg}))
        SKIP("model or image not found");
      Harness h{body_jpg, det};
      h.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
      h.node.inputs.detection_class.value = 0; // person
      for(int frame = 0; frame < 2; frame++) // the range, once probed, sticks
      {
        h.run();
        const auto& people = h.node.outputs.poses.value;
        REQUIRE(people.size() >= 2);
        CHECK(people.front().mean_confidence > 0.5f);
      }
    }
  }
}

TEST_CASE("BlazePalm 256 uses its five-layer anchors", "[onnx][pose][model]")
{
  const auto palm = wild2 + "/pose2__blazepalm__blazepalm.onnx";
  if(!TestPaths::haveAll({palm, hand_jpg}))
    SKIP("model or image not found");

  // The palm fills the lower middle of the frame: the stride-32 anchors must
  // not decode it near the top edge.
  Harness h{hand_jpg, palm};
  h.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  const auto& b = h.node.outputs.detection.value->box;
  const float xc = b.x + b.w / 2, yc = b.y + b.h / 2;
  CHECK(xc > 0.3f);
  CHECK(xc < 0.6f);
  CHECK(yc > 0.55f);
  CHECK(yc < 0.9f);
}

TEST_CASE("Hand landmarks crop the hand class of a body-head-hand detector", "[onnx][pose][model]")
{
  const auto bhh = package + "/detectors/det-body-yolox-bhh-320.onnx";
  const auto lm = package + "/landmarks/lm-hand-rtmpose-21.onnx";
  if(!TestPaths::haveAll({bhh, lm, body_jpg}))
    SKIP("models or image not found");

  auto spread = [](const OnnxModels::DetectedPose& p) {
    float x0 = 1, x1 = 0, y0 = 1, y1 = 0;
    for(const auto& k : p.keypoints)
    {
      x0 = std::min(x0, k.x);
      x1 = std::max(x1, k.x);
      y0 = std::min(y0, k.y);
      y1 = std::max(y1, k.y);
    }
    return std::max(x1 - x0, y1 - y0);
  };

  // The hand model gets a hand crop, not class 0 (a body); a hand is a small
  // part of body.jpg.
  Harness h{body_jpg, bhh, lm};
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  CHECK(spread(*h.node.outputs.detection.value) < 0.15f);

  // An explicit Detection Class wins: 0 crops a body again.
  Harness body{body_jpg, bhh, lm};
  body.node.inputs.detection_class.value = 0;
  body.run();
  REQUIRE(body.node.outputs.detection.value.has_value());
  CHECK(spread(*body.node.outputs.detection.value) > 0.2f);
}

// YOLO-pose layouts other than the 640 px v8 one (56x8400) and yolo26's
// 300x57, and K other than 17.
namespace
{
std::vector<OnnxModels::Yolo::YOLO_pose::pose_type>
decodeYolo(std::vector<float>& data, std::vector<int64_t> shape, int& K)
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value v[1]{Ort::Value::CreateTensor<float>(
      mem, data.data(), data.size(), shape.data(), shape.size())};
  std::vector<OnnxModels::Yolo::YOLO_pose::pose_type> out;
  K = OnnxModels::Yolo::YOLO_pose{}.processOutput({}, v, out, 100, 0.5f);
  return out;
}

// One confident pose at anchor/detection `j`, every keypoint i at (i, 2i).
std::vector<float> onePose(int count, int features, int j, bool row_major, bool corners)
{
  std::vector<float> d((std::size_t)count * features, 0.f);
  auto set = [&](int f, float v) {
    (row_major ? d[(std::size_t)j * features + f] : d[(std::size_t)f * count + j]) = v;
  };
  set(0, 100.f);
  set(1, 100.f);
  set(2, corners ? 150.f : 50.f);
  set(3, corners ? 180.f : 80.f);
  set(4, 0.9f);
  const int kp0 = corners ? 6 : 5;
  for(int i = 0; kp0 + i * 3 + 2 < features; i++)
  {
    set(kp0 + i * 3, (float)i);
    set(kp0 + i * 3 + 1, 2.f * i);
    set(kp0 + i * 3 + 2, 0.95f);
  }
  return d;
}
}

TEST_CASE("YOLO-pose decodes any resolution and keypoint count", "[onnx][pose]")
{
  int K = 0;
  SECTION("v8 at 320 px: [1,56,2100]")
  {
    auto d = onePose(2100, 56, 7, false, false);
    auto p = decodeYolo(d, {1, 56, 2100}, K);
    CHECK(K == 17);
    REQUIRE(p.size() == 1);
    CHECK(p[0].keypoints.size() == 17);
    CHECK(p[0].geometry.x == 75.f); // centre 100, width 50
  }
  SECTION("v8 hand head: [1,68,8400], 21 keypoints")
  {
    auto d = onePose(8400, 68, 123, false, false);
    auto p = decodeYolo(d, {1, 68, 8400}, K);
    CHECK(K == 21);
    REQUIRE(p.size() == 1);
    REQUIRE(p[0].keypoints.size() == 21);
    CHECK(p[0].keypoints[20].x == 20.f);
    CHECK(p[0].keypoints[20].y == 40.f);
  }
  SECTION("v8 transposed: [1,8400,56]")
  {
    auto d = onePose(8400, 56, 5, true, false);
    auto p = decodeYolo(d, {1, 8400, 56}, K);
    CHECK(K == 17);
    REQUIRE(p.size() == 1);
    CHECK(p[0].keypoints[3].y == 6.f);
  }
  SECTION("yolo26: [1,300,57], corner boxes")
  {
    auto d = onePose(300, 57, 2, true, true);
    auto p = decodeYolo(d, {1, 300, 57}, K);
    CHECK(K == 17);
    REQUIRE(p.size() == 1);
    CHECK(p[0].geometry.w == 50.f); // x1 100 -> x2 150
    CHECK(p[0].keypoints.size() == 17);
  }
}


// The shipped single-stage presets (v8 and yolo26 layouts) find the players,
// each with its 17 COCO keypoints.
TEST_CASE("YOLO-pose presets find the players", "[onnx][pose][model]")
{
  for(const auto& m : {package + "/single_stage/ss-yolov8n-pose.onnx",
                       package + "/single_stage/ss-yolov8s-pose.onnx",
                       package + "/single_stage/ss-yolo26n-pose.onnx",
                       package + "/single_stage/ss-yolo26s-pose.onnx"})
  {
    DYNAMIC_SECTION(m)
    {
      if(!TestPaths::haveAll({m, body_jpg}))
        SKIP("model or image not found");
      Harness h{body_jpg, "", m};
      h.node.inputs.track_ids.value = true;
      h.node.inputs.max_instances.value = 8;
      h.run();
      const auto& poses = h.node.outputs.poses.value;
      CHECK(poses.size() >= 2);
      for(const auto& p : poses)
        CHECK(p.keypoints.size() == 17);
    }
  }
}

// With all output dims dynamic (FaceBoxesProd when ORT's shape inference is
// off, e.g. OpenVINO), the model is still classified by probing.
TEST_CASE("FaceBoxes: dynamic output dims are probed", "[onnx][pose][model]")
{
  const auto model = package + "/detectors/det-face-faceboxes.onnx";
  if(!TestPaths::haveAll({model}))
    SKIP("model not found");
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto bytes = TestPaths::slurp(model);
  Onnx::OnnxRunContext ctx{bytes, model};
  Onnx::ModelIO io;
  for(const auto& p : ctx.readModelSpec().inputs)
    io.inputs.push_back({p.name, p.shape});
  for(const auto& p : ctx.readModelSpec().outputs)
    io.outputs.push_back({p.name, std::vector<int64_t>(p.shape.size(), -1)});
  CHECK(Onnx::classify(io).kind == Onnx::ModelKind::Unknown);
  REQUIRE(OnnxModels::probeOutputShapes(ctx, io));
  CHECK(Onnx::classify(io).kind == Onnx::ModelKind::FaceBoxesDetector);
}


// A dynamic-input multi-class detector fed 128x128 reports boxes in its
// internal 160x128 space; x must not come out 1.25x too wide.
TEST_CASE("Dynamic-input multi-class detector: boxes in the image", "[onnx][pose][model]")
{
  const std::string det
      = TestPaths::wild() + "/pinto/459__yolov9_n_wholebody25_post_0100_1x3x128x160.onnx";
  if(!TestPaths::haveAll({det, body_jpg}))
    SKIP("model or image not found");
  Harness h{body_jpg, det};
  h.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
  h.node.inputs.detection_class.value = 0; // body
  h.run();
  const auto& bodies = h.node.outputs.poses.value;
  REQUIRE(bodies.size() >= 2);
  for(const auto& b : bodies)
  {
    CHECK(b.box.x >= -0.02f);
    CHECK(b.box.x + b.box.w <= 1.02f);
  }
}

// A crop that spills over the frame edge must read a CONSTANT border (the
// black copyMakeBorder pad such models are trained on), not the replicated
// edge. nullptr keeps the clamp.
TEST_CASE("Affine sampler: constant border outside the frame", "[pose][imageops]")
{
  // 4x4 RGBA, each column its own colour so a clamp is distinguishable.
  std::vector<uint8_t> px(4 * 4 * 4);
  for(int y = 0; y < 4; y++)
    for(int x = 0; x < 4; x++)
    {
      uint8_t* p = px.data() + (y * 4 + x) * 4;
      p[0] = uint8_t(40 + 50 * x);
      p[1] = uint8_t(10 + 20 * x + y);
      p[2] = uint8_t(200 - 30 * x);
      p[3] = 255;
    }
  const Onnx::ImageView src{px.data(), 4, 4, 4, 16};
  const float mean[3] = {123.675f, 116.28f, 103.53f};
  const float invstd[3] = {1.f / 58.395f, 1.f / 57.12f, 1.f / 57.375f};
  const int plane = 16;
  auto at = [&](const std::vector<float>& t, int c, int x, int y) {
    return t[c * plane + y * 4 + x];
  };

  // Output x -> source x - 2: output columns 0,1 are left of the image (sx
  // -2 and -1: every tap outside), 2,3 read source columns 0,1 exactly.
  Onnx::Affine a;
  a.m2 = -2.f;
  std::vector<float> border(3 * plane), clamp(3 * plane);
  const uint8_t black[3] = {0, 0, 0};
  Onnx::sampleAffineToTensor(
      Onnx::TensorLayout::NchwRgb, src, a, 4, 4, mean, invstd, border.data(),
      Onnx::prof::WarpCrop, black);
  Onnx::sampleAffineToTensor(
      Onnx::TensorLayout::NchwRgb, src, a, 4, 4, mean, invstd, clamp.data());

  for(int y = 0; y < 4; y++)
    for(int c = 0; c < 3; c++)
    {
      for(int x = 0; x < 2; x++)
      {
        CHECK(at(border, c, x, y) == (0.f - mean[c]) * invstd[c]);
        // The clamp replicates source column 0.
        CHECK(at(clamp, c, x, y) == at(clamp, c, 2, y));
      }
      for(int x = 2; x < 4; x++)
      {
        CHECK(at(border, c, x, y) == at(clamp, c, x, y));
        const float v = px[(y * 4 + (x - 2)) * 4 + c];
        CHECK(at(border, c, x, y) == (v - mean[c]) * invstd[c]);
      }
    }

  // A sample half a pixel outside blends the edge pixel with the border 50/50,
  // like cv2.warpAffine(BORDER_CONSTANT); the far side clamps nothing.
  a.m2 = -0.5f;
  Onnx::sampleAffineToTensor(
      Onnx::TensorLayout::NchwRgb, src, a, 4, 4, mean, invstd, border.data(),
      Onnx::prof::WarpCrop, black);
  for(int c = 0; c < 3; c++)
  {
    const float v = px[(1 * 4 + 0) * 4 + c]; // row 1, column 0
    CHECK(std::abs(at(border, c, 0, 1) - (0.5f * v - mean[c]) * invstd[c]) < 1e-4f);
  }

  // Fully inside: identical in both modes (the per-row fast path).
  a = {};
  Onnx::sampleAffineToTensor(
      Onnx::TensorLayout::NchwRgb, src, a, 4, 4, mean, invstd, border.data(),
      Onnx::prof::WarpCrop, black);
  Onnx::sampleAffineToTensor(
      Onnx::TensorLayout::NchwRgb, src, a, 4, 4, mean, invstd, clamp.data());
  CHECK(border == clamp);
}

// Angles are smoothed on the unwrapped signal: a joint hovering at +-pi must
// stay at +-pi, not average to 0 (the wrong way round).
TEST_CASE("smoothParams unwraps angle entries", "[pose][smoothing]")
{
  Onnx::PoseSmoother angles, linear;
  angles.configure(1.0f, 0.f);
  linear.configure(1.0f, 0.f);
  const std::uint8_t mask[2] = {1, 0};
  for(int i = 0; i < 30; i++)
  {
    const float raw = (i % 2) ? -3.1f : 3.1f;
    float v[2] = {raw, raw};
    Onnx::smoothParams(angles, v, mask, Onnx::kNominalFrameDt);
    CHECK(std::abs(v[0]) > 3.0f);   // near +-pi
    CHECK(std::abs(v[0]) <= 3.14159266f);
    if(i > 10)
      CHECK(std::abs(v[1]) < 2.0f); // the plain filter averages toward 0
  }

  // A NaN leaves its value and filter alone.
  float v[2] = {std::numeric_limits<float>::quiet_NaN(), 1.f};
  Onnx::smoothParams(angles, v, mask, Onnx::kNominalFrameDt);
  CHECK(v[0] != v[0]);
  float w[2] = {3.1f, 0.f};
  Onnx::smoothParams(angles, w, mask, Onnx::kNominalFrameDt);
  CHECK(std::abs(w[0]) > 3.0f);
}

// The tracker carries world / translation / params per id: smoothed on
// updates, and still held on a coasted (no-detection) frame.
TEST_CASE("Tracker carries the 3D payload per id", "[pose][track]")
{
  Onnx::Track::PoseTracker tr;
  Onnx::Track::Config cfg;
  cfg.min_hits = 1;
  cfg.smooth = true;
  tr.configure(cfg);

  std::vector<Onnx::Track::Detection> dets(1);
  auto& d = dets[0];
  d.box = {0.5f, 0.5f, 0.2f, 0.4f};
  d.score = 0.9f;
  d.keypoints = {{0.5f, 0.4f, 0.f, 0.9f}, {0.5f, 0.6f, 0.f, 0.9f}};
  d.world = {{0.f, -0.5f, 0.f, 0.9f}, {0.f, 0.5f, 0.f, 0.9f}};
  d.translation = {0.f, 0.f, 3.f};
  d.params = {0.1f, 0.2f, 0.3f};

  tr.update(dets); // births the track (ids are reported from the next match)
  REQUIRE(tr.tracks().size() == 1);
  CHECK(tr.tracks()[0].translation_smooth == d.translation); // first sample
  CHECK(tr.tracks()[0].world_smooth.size() == 2);
  CHECK(tr.tracks()[0].params_smooth == d.params);

  // A 1 m jump in depth is smoothed, not taken at once.
  d.translation = {0.f, 0.f, 4.f};
  d.world[0].y = -0.4f;
  REQUIRE(tr.update(dets).front() >= 0);
  const float z = tr.tracks()[0].translation_smooth[2];
  CHECK(z > 3.f);
  CHECK(z < 4.f);
  CHECK(tr.tracks()[0].world_smooth[0].y > -0.5f);
  CHECK(tr.tracks()[0].world_smooth[0].y < -0.4f);

  // Coasted frame: the payload is still there to re-emit.
  tr.update({});
  REQUIRE(tr.tracks().size() == 1);
  CHECK(tr.tracks()[0].time_since_update == 1);
  CHECK(tr.tracks()[0].translation_smooth[2] == z);
  CHECK(tr.tracks()[0].params_smooth.size() == 3);

  // Steady state: carrying + smoothing the payload reuses the track's
  // buffers, no allocation per frame.
  for(int i = 0; i < 3; i++)
    tr.update(dets);
  AllocCounter::begin(1);
  for(int i = 0; i < 5; i++)
  {
    d.translation[2] = 4.f + 0.01f * i;
    tr.update(dets);
  }
  CHECK(AllocCounter::end() == 0);

  // Smoothing off: the raw payload passes through.
  cfg.smooth = false;
  tr.configure(cfg);
  tr.update(dets);
  CHECK(tr.tracks()[0].translation_smooth == d.translation);
}

TEST_CASE("OnnxRunContext exposes the graph's custom metadata", "[onnx][pose]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  const std::string path = SCORE_ONNX_TEST_DATA_DIR "/metadata/metadata_identity.onnx";
  const auto bytes = TestPaths::slurp(path);
  REQUIRE(!bytes.empty());
  Onnx::OnnxRunContext ctx{bytes, path};
  const auto& md = ctx.metadata();
  CHECK(md.size() == 3);
  CHECK(ctx.metadataValue("cliff_focal") == "true");
  CHECK(ctx.metadataValue("image_size") == "224");
  CHECK(ctx.metadataValue("empty", "x").empty());
  CHECK(ctx.metadataValue("absent", "fallback") == "fallback");
  CHECK(&ctx.metadata() == &md); // read once, cached

  // A model without metadata: an empty map, no throw.
  const std::string plain = SCORE_ONNX_TEST_DATA_DIR "/classify/dyn8.onnx";
  const auto pbytes = TestPaths::slurp(plain);
  REQUIRE(!pbytes.empty());
  Onnx::OnnxRunContext pctx{pbytes, plain};
  CHECK(pctx.metadata().empty());
}

// BlazePose's world output follows the Skeleton remap like the keypoints
// (not the native 33 joints), coasts with its track id, and
// CameraXYZArray falls back to WorldXYZArray without a translation.
TEST_CASE("BlazePose world: remapped, tracked, and in the 3D formats", "[onnx][pose][model]")
{
  const auto det = presets + "/detectors/det-body-blazepose-224.onnx";
  const auto lm = presets + "/landmarks/lm-body-blazepose-full.onnx";
  if(!TestPaths::haveAll({det, lm, body_jpg}))
    SKIP("models or image not found");

  using F = OnnxModels::KeypointOutputFormat;
  using T = Onnx::Skel::TargetSkeleton;

  // Single path, first frame (smoothing passes the first sample through).
  Harness native{body_jpg, det, lm};
  native.node.inputs.data_format.value = F::WorldXYZArray;
  native.node.inputs.min_confidence.value = 0.f;
  native.run();
  REQUIRE(native.node.outputs.detection.value.has_value());
  const auto nat = *native.node.outputs.detection.value;
  REQUIRE(nat.world.size() == 33);
  REQUIRE(nat.keypoints.size() == 33);
  CHECK(native.node.outputs.geometry.value.size() == 33 * 3);

  Harness coco{body_jpg, det, lm};
  coco.node.inputs.skeleton_type.value = T::Coco17;
  coco.node.inputs.data_format.value = F::CameraXYZArray;
  coco.node.inputs.min_confidence.value = 0.f;
  coco.run();
  REQUIRE(coco.node.outputs.detection.value.has_value());
  const auto& rc = *coco.node.outputs.detection.value;
  REQUIRE(rc.keypoints.size() == 17);
  REQUIRE(rc.world.size() == 17);
  // COCO 0 = BlazePose 0 (nose), COCO 5 = BlazePose 11 (left shoulder).
  CHECK(rc.world[0].x == nat.world[0].x);
  CHECK(rc.world[5].y == nat.world[11].y);
  CHECK(rc.world[5].z == nat.world[11].z);
  CHECK(rc.translation.empty());
  // No translation: CameraXYZArray == WorldXYZArray (17 remapped joints).
  const auto& g = coco.node.outputs.geometry.value;
  REQUIRE(g.size() == 17 * 3);
  CHECK(g[5 * 3 + 1] == nat.world[11].y);

  // No body params: BodyParams emits nothing.
  coco.node.inputs.data_format.value = F::BodyParams;
  coco.run();
  CHECK(coco.node.outputs.geometry.value.empty());

  // Multi-instance: tracked poses keep their (remapped) world; a black frame
  // (detector miss) re-emits the coasted instance WITH its world.
  Harness multi{body_jpg, det, lm};
  multi.node.inputs.track_ids.value = true;
  multi.node.inputs.skeleton_type.value = T::Coco17;
  multi.node.inputs.data_format.value = F::WorldXYZArray;
  multi.node.inputs.min_confidence.value = 0.f;
  for(int i = 0; i < 4; i++)
    multi.run();
  const auto& poses = multi.node.outputs.poses.value;
  REQUIRE(!poses.empty());
  for(const auto& p : poses)
  {
    CHECK(p.keypoints.size() == 17);
    CHECK(p.world.size() == 17);
  }
  QImage black(multi.image.size(), QImage::Format_RGBA8888);
  black.fill(Qt::black);
  multi.image = black;
  multi.run();
  const auto& coasted = multi.node.outputs.poses.value;
  REQUIRE(!coasted.empty());
  for(const auto& p : coasted)
  {
    CHECK(p.track_id >= 0);
    CHECK(p.world.size() == 17);
  }
}

// ---------------------------------------------------------------------------
// InstantHMR. The references in tests/data/instanthmr/*.json come from
// upstream inference.py preprocessing + decode on the same model, CPU EP,
// FIXED person boxes.
//
// How the reference box reaches the node: the node's whole-frame InstantHMR
// mode (no Detection Model) takes the frame itself as the person box, exactly
// as upstream crops a detector box. body_crop.json is upstream run on body.jpg
// cropped to `crop` with box = that whole crop, so the test crops body.jpg the
// same way (QImage::copy) and the node must reproduce crop, CLIFF, inference
// and decode end to end, no detector and no test hook. The crop was chosen so
// upstream's floor/ceil square lands on whole pixels. The CLIFF builder itself
// is checked on the non-trivial boxes (body_box / f0_box) in its own case.
namespace
{
const std::string hmr_model = TestPaths::instantHmr();
const std::string person_det = presets + "/detectors/det-person-yolox-tiny-416.onnx";

struct HmrRef
{
  QJsonObject o;
  std::vector<float> arr(const char* key) const
  {
    std::vector<float> v;
    for(const auto& x : o.value(key).toArray())
      v.push_back(static_cast<float>(x.toDouble()));
    return v;
  }
};
HmrRef loadRef(const char* name)
{
  QFile f(QString(SCORE_ONNX_TEST_DATA_DIR "/instanthmr/") + name + ".json");
  REQUIRE(f.open(QIODevice::ReadOnly));
  return {QJsonDocument::fromJson(f.readAll()).object()};
}
}

TEST_CASE("InstantHMR: CLIFF vector and square match upstream", "[pose][instanthmr]")
{
  for(const char* name : {"body_box", "f0_box", "body_crop"})
  {
    DYNAMIC_SECTION(name)
    {
      const auto ref = loadRef(name);
      const auto box = ref.arr("box"); // x1,y1,x2,y2
      const auto cliff = ref.arr("cliff");
      const auto sq = ref.arr("square"); // x1, y1, side
      const int W = ref.o.value("width").toInt(), H = ref.o.value("height").toInt();
      REQUIRE(box.size() == 4);
      const Onnx::ROI::Rect r = Onnx::ROI::topdownRect(
          Onnx::Rect{box[0], box[1], box[2] - box[0], box[3] - box[1]}, 224, 224,
          OnnxModels::kHmrCropExpand);
      // Upstream's crop square.
      CHECK(std::abs(r.w - sq[2]) < 1e-3f);
      CHECK(std::abs(r.h - sq[2]) < 1e-3f);
      CHECK(std::abs((r.cx - r.w / 2) - sq[0]) < 1e-3f);
      CHECK(std::abs((r.cy - r.h / 2) - sq[1]) < 1e-3f);
      float c[3];
      OnnxModels::instantHmrCliff(r, W, H, false, c);
      for(int i = 0; i < 3; i++)
        CHECK(std::abs(c[i] - cliff[i]) < 1e-5f);
      // The per-crop affine gives the rect (and so the CLIFF vector) back.
      const auto back = OnnxModels::rectFromAffine(
          Onnx::ROI::rectToAffine(r, 224, 224), 224, 224);
      CHECK(std::abs(back.cx - r.cx) < 1e-3f);
      CHECK(std::abs(back.cy - r.cy) < 1e-3f);
      CHECK(std::abs(back.w - r.w) < 1e-3f);
    }
  }
  // Whole-frame mode: the frame is the box.
  const auto wf = OnnxModels::instantHmrFrameRect(500, 560, 224, 224);
  float c[3];
  OnnxModels::instantHmrCliff(wf, 500, 560, false, c);
  CHECK(c[0] == 0.f);
  CHECK(c[1] == 0.f);
  CHECK(std::abs(c[2] - 1.f) < 1e-6f);
  // Angular form (metadata cliff_focal=true): f = 1.05 * diag.
  OnnxModels::instantHmrCliff(
      Onnx::ROI::Rect{400.f, 300.f, 240.f, 240.f, 0.f}, 800, 600, true, c);
  const float f = 1.05f * 1000.f;
  CHECK(std::abs(c[0] - std::atan(0.f / f)) < 1e-6f);
  CHECK(std::abs(c[2] - 200.f / f) < 1e-6f);
}

TEST_CASE("InstantHMR: node reproduces the upstream reference", "[onnx][pose][instanthmr][model]")
{
  if(!TestPaths::haveAll({hmr_model, body_jpg}))
    SKIP("model or image not found");
  const auto ref = loadRef("body_crop");
  const auto crop = ref.arr("crop");
  const auto j2 = ref.arr("joints_2d"), j3 = ref.arr("joints_3d");
  const auto cam = ref.arr("cam_trans"), mhr = ref.arr("mhr_params"),
             shp = ref.arr("shape_params");
  REQUIRE(j2.size() == 140);
  REQUIRE(j3.size() == 210);

  Harness h{body_jpg, "", hmr_model};
  h.image = h.image.copy(
      (int)crop[0], (int)crop[1], (int)(crop[2] - crop[0]), (int)(crop[3] - crop[1]));
  const int W = h.image.width(), Hh = h.image.height();
  REQUIRE(W == ref.o.value("width").toInt());
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.smoothing.value = false;
  h.node.inputs.data_format.value = OnnxModels::KeypointOutputFormat::CameraXYZArray;
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  const auto& p = *h.node.outputs.detection.value;
  REQUIRE(p.keypoints.size() == 70);
  REQUIRE(p.world.size() == 70);
  REQUIRE(p.translation.size() == 3);
  REQUIRE(p.body_params.size() == 204 + 45);
  CHECK(p.mean_confidence == 1.f); // whole frame: no detector score

  // 2D, frame px.
  double e2 = 0, e2max = 0;
  for(int k = 0; k < 70; k++)
  {
    const double e = std::hypot(p.keypoints[k].x * W - j2[2 * k], p.keypoints[k].y * Hh - j2[2 * k + 1]);
    e2 += e / 70;
    e2max = std::max(e2max, e);
  }
  // 3D: world is pelvis-relative, translation carries the pelvis; world +
  // translation == joints_3d + cam_trans.
  float pel[3];
  for(int c = 0; c < 3; c++)
    pel[c] = 0.5f * (j3[9 * 3 + c] + j3[10 * 3 + c]);
  double e3 = 0, e3max = 0, ecam = 0;
  for(int k = 0; k < 70; k++)
  {
    const auto& w = p.world[k];
    const double e = std::sqrt(
        std::pow(w.x - (j3[3 * k] - pel[0]), 2) + std::pow(w.y - (j3[3 * k + 1] - pel[1]), 2)
        + std::pow(w.z - (j3[3 * k + 2] - pel[2]), 2));
    e3 += e / 70;
    e3max = std::max(e3max, e);
  }
  for(int c = 0; c < 3; c++)
    ecam = std::max<double>(ecam, std::abs(p.translation[c] - (cam[c] + pel[c])));
  double epar = 0, epar_mean = 0;
  for(int i = 0; i < 249; i++)
  {
    const float r = i < 204 ? mhr[i] : shp[i - 204];
    const double e = std::abs(p.body_params[i] - r);
    epar = std::max(epar, e);
    epar_mean += e / 249;
  }
  INFO("vs upstream: 2D mean " << e2 << " px, 3D mean " << e3 * 100 << " cm");
  CHECK(e2max < 3.0);
  CHECK(e3max < 0.01);
  CHECK(ecam < 0.02);
  CHECK(epar_mean < 0.002); // rad / scale units
  CHECK(epar < 0.02);

  // CameraXYZArray = world + translation, every joint at Min Confidence 0.
  const auto& g = h.node.outputs.geometry.value;
  REQUIRE(g.size() == 70 * 3);
  CHECK(std::abs(g[0] - (j3[0] + cam[0])) < 0.01f);
  CHECK(std::abs(g[2] - (j3[2] + cam[2])) < 0.02f);

  // Skeleton = BODY_25: remapped keypoints AND world, feet filled (MHR70 has
  // real heels and toes), neck a real joint.
  h.node.inputs.skeleton_type.value = Onnx::Skel::TargetSkeleton::OpenPoseBody25;
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  const auto& b = *h.node.outputs.detection.value;
  REQUIRE(b.keypoints.size() == 25);
  REQUIRE(b.world.size() == 25);
  for(int k = 19; k < 25; k++)
    CHECK(b.keypoints[k].confidence > 0.f);
  CHECK(std::abs(b.keypoints[1].x * W - j2[2 * 69]) < 3.f); // Neck = MHR 69
  CHECK(std::abs(b.keypoints[21].y * Hh - j2[2 * 17 + 1]) < 3.f); // LHeel = 17
  CHECK(h.node.outputs.geometry.value.size() == 25 * 3);
}

namespace
{
// Pinhole projection of world + translation with upstream's camera: principal
// point at the frame centre, f = the frame diagonal (px). Mean distance to the
// 2D keypoints, px.
double reprojection(const OnnxModels::DetectedPose& p, int W, int H)
{
  const double f = std::hypot(W, H);
  double e = 0;
  for(size_t k = 0; k < p.keypoints.size(); k++)
  {
    const double X = p.world[k].x + p.translation[0], Y = p.world[k].y + p.translation[1],
                 Z = p.world[k].z + p.translation[2];
    e += std::hypot(f * X / Z + W / 2. - p.keypoints[k].x * W,
                    f * Y / Z + H / 2. - p.keypoints[k].y * H);
  }
  return e / p.keypoints.size();
}
}

TEST_CASE("InstantHMR: two-stage with a person detector", "[onnx][pose][instanthmr][model]")
{
  if(!TestPaths::haveAll({hmr_model, person_det, body_jpg}))
    SKIP("models or image not found");
  using F = OnnxModels::KeypointOutputFormat;
  Harness h{body_jpg, person_det, hmr_model};
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.data_format.value = F::CameraXYZArray;
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  const auto& p = *h.node.outputs.detection.value;
  REQUIRE(p.keypoints.size() == 70);
  REQUIRE(p.world.size() == 70);
  REQUIRE(p.translation.size() == 3);
  CHECK(p.translation[2] > 1.f);
  CHECK(p.translation[2] < 10.f);
  CHECK(p.body_params.size() == 249);
  // Joint confidence = the detector score (MHR70 has none of its own).
  CHECK(p.mean_confidence > 0.3f);
  CHECK(p.mean_confidence <= 1.f);
  CHECK(p.keypoints[0].confidence == p.mean_confidence);
  CHECK(h.node.outputs.geometry.value.size() == 70 * 3);

  // The pose lies on the most prominent player (reference box of body_box).
  const auto ref = loadRef("body_box");
  const auto j2 = ref.arr("joints_2d");
  double near = 0;
  for(int k = 0; k < 17; k++)
    near += std::hypot(p.keypoints[k].x * h.image.width() - j2[2 * k],
                       p.keypoints[k].y * h.image.height() - j2[2 * k + 1]) / 17;
  CHECK(near < 25.0);

  // Reprojection: the 3D placement agrees with the 2D head through upstream's
  // camera.
  const double rep = reprojection(p, h.image.width(), h.image.height());
  CHECK(rep < 15.0);

  h.node.inputs.data_format.value = F::BodyParams;
  h.run();
  {
    // The record leads with the rig origin (translation - rig_offset), where
    // a mesh evaluated from the body params sits, then the 249 params.
    const auto& g = h.node.outputs.geometry.value;
    REQUIRE(g.size() == 3 + 249);
    REQUIRE(h.node.outputs.detection.value.has_value());
    const auto& q = *h.node.outputs.detection.value;
    REQUIRE(q.translation.size() == 3);
    REQUIRE(q.rig_offset.size() == 3);
    for(int i = 0; i < 3; i++)
      CHECK(g[i] == q.translation[i] - q.rig_offset[i]);
    // The pelvis sits ~1 m above the rig origin (MHR joint 0, near the floor).
    CHECK(std::abs(g[1] - q.translation[1]) > 0.5f);
    for(int i = 0; i < 249; i++)
      CHECK(g[3 + i] == q.body_params[i]);
  }
  h.node.inputs.data_format.value = F::LineArray;
  h.run();
  CHECK(h.node.outputs.geometry.value.size() > 60 * 6); // body + finger bones
}

TEST_CASE("InstantHMR: batched multi-instance == per-crop inference", "[onnx][pose][instanthmr][model]")
{
  if(!TestPaths::haveAll({hmr_model, person_det, body_jpg}))
    SKIP("models or image not found");

  // The detector boxes, straight from Box Detection (first frame, unsmoothed).
  Harness boxes{body_jpg, person_det};
  boxes.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
  boxes.node.inputs.detection_class.value = 0;
  boxes.node.inputs.smoothing.value = false;
  boxes.node.inputs.min_confidence.value = 0.f;
  boxes.run();
  const auto dets = boxes.node.outputs.poses.value;
  REQUIRE(dets.size() >= 2);

  Harness h{body_jpg, person_det, hmr_model};
  h.node.inputs.track_ids.value = true;
  h.node.inputs.max_instances.value = 5;
  h.node.inputs.smoothing.value = false;
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.data_format.value = OnnxModels::KeypointOutputFormat::BodyParams;
  for(int i = 0; i < 4; i++) // confirm the tracks (min_hits); unsmoothed
    h.run();
  const auto poses = h.node.outputs.poses.value;
  REQUIRE(poses.size() >= 2);
  const int W = h.image.width(), H = h.image.height();

  // Per-crop reference, one batch-1 inference per detector box through the
  // node's own crop / CLIFF helpers.
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto bytes = TestPaths::slurp(hmr_model);
  Onnx::OnnxRunContext ctx{bytes, hmr_model};
  const auto& spec = ctx.readModelSpec();
  Onnx::ModelIO io;
  for(const auto& q : spec.outputs)
    io.outputs.push_back({q.name, q.shape});
  const auto hi = Onnx::resolveHmrOutputs(io);
  REQUIRE(hi.valid());
  const Onnx::ImageView src{h.image.bits(), W, H, 4, W * 4};
  boost::container::vector<float> buf;
  int matched = 0;
  for(const auto& d : dets)
  {
    const Onnx::ROI::Rect r = Onnx::ROI::topdownRect(
        Onnx::Rect{d.box.x * W, d.box.y * H, d.box.w * W, d.box.h * H}, 224, 224,
        OnnxModels::kHmrCropExpand);
    const auto M = Onnx::ROI::rectToAffine(r, 224, 224);
    const uint8_t black[3] = {0, 0, 0};
    auto t = OnnxModels::fusedAffineTensor(
        spec.inputs[0], src, OnnxModels::instantHmrSampleAffine(M), 224, 224,
        OnnxModels::normMeanStd(
            Onnx::TensorLayout::NchwRgb, {123.675f, 116.28f, 103.53f},
            {58.395f, 57.12f, 57.375f}),
        buf, Onnx::prof::WarpCrop, black);
    std::array<float, 3> cliff;
    OnnxModels::instantHmrCliff(r, W, H, false, cliff.data());
    std::array<int64_t, 2> cs{1, 3};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value ins[2]{
        std::move(t.value),
        Ort::Value::CreateTensor<float>(mem, cliff.data(), 3, cs.data(), 2)};
    Ort::Value outs[5]{
        Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
        Ort::Value{nullptr}, Ort::Value{nullptr}};
    ctx.infer(spec, ins, outs);
    std::swap(buf, t.storage);
    const float* jj = outs[hi.joints_2d].GetTensorData<float>();
    const float* j3 = outs[hi.joints_3d].GetTensorData<float>();
    const float* ct = outs[hi.cam_trans].GetTensorData<float>();
    const float* mp = outs[hi.mhr_params].GetTensorData<float>();
    // Reference nose, normalized.
    const auto nose = Onnx::ROI::applyAffine(M, (jj[0] + 1) * 112.f, (jj[1] + 1) * 112.f);

    // The batched instance on this person: the nearest nose.
    const OnnxModels::DetectedPose* best = nullptr;
    float bd = 1e9f;
    for(const auto& p : poses)
    {
      const float dd = std::hypot(p.keypoints[0].x - nose.x / W, p.keypoints[0].y - nose.y / H);
      if(dd < bd)
      {
        bd = dd;
        best = &p;
      }
    }
    if(!best || bd > 0.02f)
      continue; // a detection the tracker did not keep (max instances / gate)
    ++matched;
    const auto& p = *best;
    CHECK(p.track_id >= 0);
    REQUIRE(p.keypoints.size() == 70);
    REQUIRE(p.translation.size() == 3);
    REQUIRE(p.body_params.size() == 249);
    float e2 = 0, e3 = 0;
    for(int k = 0; k < 70; k++)
    {
      const auto q = Onnx::ROI::applyAffine(M, (jj[2 * k] + 1) * 112.f, (jj[2 * k + 1] + 1) * 112.f);
      e2 = std::max(e2, std::abs(p.keypoints[k].x - q.x / W));
      e2 = std::max(e2, std::abs(p.keypoints[k].y - q.y / H));
    }
    const float pz = 0.5f * (j3[9 * 3 + 2] + j3[10 * 3 + 2]);
    e3 = std::max(e3, std::abs(p.translation[2] - (ct[2] + pz)));
    float ep = 0;
    for(int i = 0; i < 204; i++)
      ep = std::max(ep, std::abs(p.body_params[i] - mp[i]));
    CHECK(e2 < 1e-3f);
    CHECK(e3 < 1e-3f);
    CHECK(ep < 1e-3f);
  }
  CHECK(matched >= 2);

  // Steady state: the batched InstantHMR frames reuse every large buffer (the
  // [N,3,224,224] input, the CLIFF rows, the scratch) — no big allocation.
  for(int i = 0; i < 3; i++)
    h.run();
  // Threshold: ORT's own per-run temporaries (the detector's ~55 KB conv
  // scratch, dozens per frame) stay below it; the node's buffers it guards are
  // 0.6 MB (one crop) and up.
  AllocCounter::begin(256 * 1024);
  for(int i = 0; i < 5; i++)
    h.run();
  const int allocs = AllocCounter::end();
  CHECK(allocs == 0);
  CHECK(h.node.outputs.poses.value.size() == poses.size());
}

// ---------------------------------------------------------------------------
// InstantHMR + the MHR body mesh (Body Model port, PoseDetector_mesh.cpp).
// Models from $ONNX_TEST_MHR/bin (tools/mhr_export.py with --kp-regressor and
// --landmarks); SKIP when absent.
namespace
{
namespace Mhr = Onnx::Mhr;
using KF = OnnxModels::KeypointOutputFormat;

// Signed volume of a triangle soup (9 floats per face), m^3: positive for a
// closed mesh wound counter-clockwise seen from outside, in any right-handed
// frame and wherever the mesh sits.
double soupVolume(const float* t, size_t faces)
{
  double v = 0;
  for(size_t f = 0; f < faces; f++, t += 9)
    v += t[0] * (double(t[4]) * t[8] - double(t[5]) * t[7])
         - t[1] * (double(t[3]) * t[8] - double(t[5]) * t[6])
         + t[2] * (double(t[3]) * t[7] - double(t[4]) * t[6]);
  return v / 6;
}

// The node's camera: f = frame diagonal, principal point at the centre.
std::array<double, 2> project(const float* X, int W, int H)
{
  const double f = std::hypot(W, H);
  return {f * X[0] / X[2] + W / 2., f * X[1] / X[2] + H / 2.};
}

// The rig origin in camera space, the node's placement of the mesh.
std::array<float, 3> rigOrigin(const OnnxModels::DetectedPose& p)
{
  return {p.translation[0] - p.rig_offset[0], p.translation[1] - p.rig_offset[1],
          p.translation[2] - p.rig_offset[2]};
}

// A reference evaluation of pose p's mesh (camera space, OpenCV axes).
struct MeshRef
{
  std::vector<float> verts, kp;
  bool eval(const Mhr::Model& m, Mhr::Workspace& ws, const OnnxModels::DetectedPose& p)
  {
    verts.assign(m.numVertices() * 3, 0.f);
    kp.assign(Mhr::kNumKeypoints * 3, 0.f);
    const std::span<const float> bp(p.body_params);
    if(!Mhr::evaluate(
           m, bp.first(204), bp.subspan(204, 45), {}, ws,
           {verts, {}, kp, Mhr::KeypointMethod::Exact}))
      return false;
    const auto o = rigOrigin(p);
    for(size_t i = 0; i < verts.size(); i++)
      verts[i] += o[i % 3];
    for(size_t i = 0; i < kp.size(); i++)
      kp[i] += o[i % 3];
    return true;
  }
};
}

TEST_CASE("InstantHMR: body mesh on the reference crop", "[onnx][pose][instanthmr][mhr][model]")
{
  const auto bin = TestPaths::mhrBin(3);
  if(!TestPaths::haveAll({hmr_model, body_jpg, bin}))
    SKIP("models or image not found");
  const auto ref = loadRef("body_crop");
  const auto crop = ref.arr("crop");
  const auto cam = ref.arr("cam_trans");

  std::string err;
  auto model = Mhr::Model::loadFile(bin, &err);
  REQUIRE(model);
  REQUIRE(model->hasExactKeypoints());
  Mhr::Workspace ws(*model);
  const size_t V = model->numVertices(), F = model->numFaces();

  Harness h{body_jpg, "", hmr_model};
  h.image = h.image.copy(
      (int)crop[0], (int)crop[1], (int)(crop[2] - crop[0]), (int)(crop[3] - crop[1]));
  const int W = h.image.width(), H = h.image.height();
  h.setBodyModel(bin, TestPaths::slurp(bin));
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.smoothing.value = false;
  h.node.inputs.data_format.value = KF::MeshVertices;
  h.node.inputs.mesh_space.value = OnnxModels::MeshSpace::Camera;
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  const auto p = *h.node.outputs.detection.value;
  REQUIRE(p.body_params.size() == 249);
  REQUIRE(p.translation.size() == 3);
  REQUIRE(p.rig_offset.size() == 3);

  // translation - rig_offset is InstantHMR's raw cam_trans (unsmoothed).
  const auto o = rigOrigin(p);
  for(int c = 0; c < 3; c++)
    CHECK(std::abs(o[c] - cam[c]) < 0.02f);

  // (a) MeshVertices == Mhr::evaluate on the node's own params, placed at
  // translation - rig_offset.
  MeshRef mr;
  REQUIRE(mr.eval(*model, ws, p));
  const auto& g = h.node.outputs.geometry.value;
  REQUIRE(g.size() == V * 3);
  float emax = 0;
  for(size_t i = 0; i < g.size(); i++)
    emax = std::max(emax, std::abs(g[i] - mr.verts[i]));
  CHECK(emax < 1e-5f);

  // Plausible body in camera space: height, in front of the camera.
  float ymin = 1e9f, ymax = -1e9f, zmin = 1e9f;
  for(size_t i = 0; i < V; i++)
  {
    ymin = std::min(ymin, g[3 * i + 1]);
    ymax = std::max(ymax, g[3 * i + 1]);
    zmin = std::min(zmin, g[3 * i + 2]);
  }
  CHECK(ymax - ymin > 1.4f);
  CHECK(ymax - ymin < 2.1f);
  CHECK(zmin > 0.5f);

  // ... and it lands on the person: the mesh's MHR70 keypoints, projected
  // through the node's camera, vs the node's 2D keypoints (the 2D head).
  double e2 = 0;
  for(int k = 0; k < 70; k++)
  {
    const auto uv = project(&mr.kp[3 * k], W, H);
    e2 += std::hypot(uv[0] - p.keypoints[k].x * W, uv[1] - p.keypoints[k].y * H) / 70;
  }
  CHECK(e2 < 20.0);

  // OpenGL space = (x, -y, -z) of the camera space.
  h.node.inputs.mesh_space.value = OnnxModels::MeshSpace::OpenGL;
  h.run();
  REQUIRE(h.node.outputs.geometry.value.size() == V * 3);
  {
    const auto& gl = h.node.outputs.geometry.value;
    float e = 0;
    for(size_t i = 0; i < V * 3; i++)
      e = std::max(e, std::abs(gl[i] - (i % 3 ? -1.f : 1.f) * mr.verts[i]));
    CHECK(e < 1e-5f);
  }

  // (b) MeshTriangles: F * 9 floats, the faces' corners, outward (positive
  // signed volume) in both spaces.
  for(auto space : {OnnxModels::MeshSpace::OpenGL, OnnxModels::MeshSpace::Camera})
  {
    h.node.inputs.mesh_space.value = space;
    h.node.inputs.data_format.value = KF::MeshTriangles;
    h.run();
    const auto& t = h.node.outputs.geometry.value;
    REQUIRE(t.size() == F * 9);
    const double vol = soupVolume(t.data(), F);
    INFO("signed volume " << vol);
    CHECK(vol > 0.03);
    CHECK(vol < 0.2);
    if(space == OnnxModels::MeshSpace::Camera)
    {
      const auto faces = model->faces();
      CHECK(t[0] == mr.verts[faces[0] * 3]);
      CHECK(t[9 * F - 1] == mr.verts[faces[3 * F - 1] * 3 + 2]);
    }

  }

  // Draw Mesh changes the pixels inside the person (and only with a mesh).
  h.node.inputs.draw_mesh.value = false;
  h.run();
  const auto* px = reinterpret_cast<const uint8_t*>(h.node.outputs.image.texture.bytes);
  const std::vector<uint8_t> plain(px, px + size_t(W) * H * 4);
  h.node.inputs.draw_mesh.value = true;
  h.run();
  px = reinterpret_cast<const uint8_t*>(h.node.outputs.image.texture.bytes);
  size_t changed = 0;
  for(size_t i = 0; i < plain.size(); i += 4)
    changed += px[i] != plain[i] || px[i + 1] != plain[i + 1] || px[i + 2] != plain[i + 2];
  CHECK(changed > size_t(W) * H / 20);
  CHECK(changed < size_t(W) * H * 3 / 4);

  // (e) Mesh Keypoints: world = the mesh's MHR70 relative to the same anchor
  // (world + translation = camera space), 2D = their projection.
  h.node.inputs.mesh_keypoints.value = true;
  h.node.inputs.data_format.value = KF::CameraXYZArray;
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  const auto& q = *h.node.outputs.detection.value;
  REQUIRE(q.world.size() == 70);
  REQUIRE(q.keypoints.size() == 70);
  float ew = 0, ek = 0;
  for(int k = 0; k < 70; k++)
  {
    ew = std::max(ew, std::abs(q.world[k].x + q.translation[0] - mr.kp[3 * k]));
    ew = std::max(ew, std::abs(q.world[k].y + q.translation[1] - mr.kp[3 * k + 1]));
    ew = std::max(ew, std::abs(q.world[k].z + q.translation[2] - mr.kp[3 * k + 2]));
    const auto uv = project(&mr.kp[3 * k], W, H);
    ek = std::max(ek, float(std::hypot(uv[0] - q.keypoints[k].x * W, uv[1] - q.keypoints[k].y * H)));
  }
  CHECK(ew < 1e-5f);
  CHECK(ek < 1e-2f);
  // CameraXYZArray follows (every joint kept at Min Confidence 0).
  const auto& cx = h.node.outputs.geometry.value;
  REQUIRE(cx.size() == 70 * 3);
  CHECK(std::abs(cx[3 * 5 + 1] - mr.kp[3 * 5 + 1]) < 1e-5f);

  // Skeleton remap after the mesh keypoints: COCO-17 5 = MHR70 5.
  h.node.inputs.skeleton_type.value = Onnx::Skel::TargetSkeleton::Coco17;
  h.run();
  REQUIRE(h.node.outputs.detection.value->world.size() == 17);
  CHECK(std::abs(h.node.outputs.detection.value->world[5].z + q.translation[2] - mr.kp[3 * 5 + 2]) < 1e-5f);
}

// (d) A garbage / truncated body model: logged, no mesh, the node keeps
// running; a later valid file (new name) turns the mesh on.
TEST_CASE("InstantHMR: an invalid body model disables the mesh", "[onnx][pose][instanthmr][mhr][model]")
{
  const auto bin = TestPaths::mhrBin(6);
  if(!TestPaths::haveAll({hmr_model, body_jpg}))
    SKIP("models or image not found");
  Harness h{body_jpg, "", hmr_model};
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.data_format.value = KF::MeshTriangles;
  h.node.inputs.draw_mesh.value = true;
  h.node.inputs.mesh_keypoints.value = true;
  h.setBodyModel("garbage.mhrbin", std::string(4096, 'x'));
  for(int i = 0; i < 2; i++)
  {
    h.run();
    REQUIRE(h.node.outputs.detection.value.has_value());
    CHECK(h.node.outputs.detection.value->keypoints.size() == 70);
    CHECK(h.node.outputs.geometry.value.empty());
  }
  // Truncated real file.
  if(std::filesystem::exists(bin))
  {
    auto raw = TestPaths::slurp(bin);
    raw.resize(raw.size() / 2);
    h.setBodyModel("truncated.mhrbin", raw);
    h.run();
    REQUIRE(h.node.outputs.detection.value.has_value());
    CHECK(h.node.outputs.geometry.value.empty());

    h.setBodyModel(bin, TestPaths::slurp(bin));
    h.run();
    auto model = Mhr::Model::loadFile(bin);
    REQUIRE(model);
    CHECK(h.node.outputs.geometry.value.size() == size_t(model->numFaces()) * 9);
  }
  // No file: no mesh.
  h.setBodyModel("", "");
  h.run();
  REQUIRE(h.node.outputs.detection.value.has_value());
  CHECK(h.node.outputs.geometry.value.empty());

  // A model without body params (BlazePose) and a body model: no mesh either.
  const auto det = presets + "/detectors/det-body-blazepose-224.onnx";
  const auto lm = presets + "/landmarks/lm-body-blazepose-full.onnx";
  if(TestPaths::haveAll({det, lm, bin}))
  {
    Harness b{body_jpg, det, lm};
    b.setBodyModel(bin, TestPaths::slurp(bin));
    b.node.inputs.data_format.value = KF::MeshTriangles;
    b.node.inputs.draw_mesh.value = true;
    b.run();
    REQUIRE(b.node.outputs.detection.value.has_value());
    CHECK(b.node.outputs.geometry.value.empty());
  }
}

// (c) Track IDs, several people: one mesh per instance at a fixed stride in
// Poses Geometry, each the node's own params through Mhr::evaluate, and the
// per-id workspaces / payload buffers reused: no big allocation once warm.
TEST_CASE("InstantHMR: tracked meshes, one per person", "[onnx][pose][instanthmr][mhr][model]")
{
  const auto bin1 = TestPaths::mhrBin(1);
  if(!TestPaths::haveAll({hmr_model, person_det, body_jpg, bin1}))
    SKIP("models or image not found");

  auto model = Mhr::Model::loadFile(bin1);
  REQUIRE(model);
  Mhr::Workspace ws(*model);
  const size_t F = model->numFaces();

  Harness h{body_jpg, person_det, hmr_model};
  h.setBodyModel(bin1, TestPaths::slurp(bin1));
  h.node.inputs.track_ids.value = true;
  h.node.inputs.max_instances.value = 5;
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.data_format.value = KF::MeshTriangles;
  h.node.inputs.mesh_space.value = OnnxModels::MeshSpace::Camera;
  for(int i = 0; i < 4; i++) // confirm the tracks
    h.run();
  const auto& poses = h.node.outputs.poses.value;
  const int n = h.node.outputs.count.value;
  REQUIRE(n >= 3);
  REQUIRE(poses.size() == size_t(n));
  const auto& pg = h.node.outputs.poses_geometry.value;
  const size_t stride = F * 9;
  REQUIRE(pg.size() == 5 * stride);
  MeshRef mr;
  const auto faces = model->faces();
  for(int i = 0; i < n; i++)
  {
    INFO("instance " << i << " id " << poses[i].track_id);
    CHECK(poses[i].track_id >= 0);
    REQUIRE(mr.eval(*model, ws, poses[i]));
    const float* t = pg.data() + i * stride;
    float e = 0;
    for(size_t f = 0; f < F; f += 97)
      for(int k = 0; k < 3; k++)
        for(int c = 0; c < 3; c++)
          e = std::max(e, std::abs(t[f * 9 + k * 3 + c] - mr.verts[faces[3 * f + k] * 3 + c]));
    CHECK(e < 1e-5f);
    CHECK(soupVolume(t, F) > 0.03);
  }
  // Unused slots are zero.
  for(size_t i = n * stride; i < pg.size(); i += 101)
    CHECK(pg[i] == 0.f);
  // Primary Geometry: one of them.
  CHECK(h.node.outputs.geometry.value.size() == stride);

  // Steady state with meshes, Draw Mesh and Mesh Keypoints on: every mesh
  // buffer (the per-id workspaces with their V*3 rest / posed meshes, 221 KB
  // at LOD 1, the slots' vertices, the 1.3 MB triangle payloads and the
  // Poses Geometry block) is reused. Threshold under the LOD 1 vertex
  // buffers, above ORT's per-run temporaries (see the batched test).
  h.node.inputs.draw_mesh.value = true;
  h.node.inputs.mesh_keypoints.value = true;
  for(int i = 0; i < 3; i++)
    h.run();
  AllocCounter::begin(200 * 1024);
  for(int i = 0; i < 5; i++)
    h.run();
  const int allocs = AllocCounter::end();
  CHECK(allocs == 0);
  CHECK(h.node.outputs.count.value == n);
}

// A landmark model with a dynamic batch AND dynamic H/W: the batched
// multi-instance path must resolve H/W like the single-crop path does, not
// build an [N,3,-1,-1] input tensor.
TEST_CASE("Batched landmarks: dynamic H/W model", "[onnx][pose][model]")
{
  const std::string dyn = SCORE_ONNX_TEST_DATA_DIR "/pose/simcc_dynhw.onnx";
  const std::string det = presets + "/detectors/det-person-yolox-tiny-416.onnx";
  if(!TestPaths::haveAll({dyn, det, body_jpg}))
    SKIP("models or image not found");
  Harness h{body_jpg, det, dyn};
  h.node.inputs.track_ids.value = true;
  h.node.inputs.max_instances.value = 5;
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.smoothing.value = false;
  for(int i = 0; i < 4; i++) // confirm the tracks
    h.run();
  const auto& poses = h.node.outputs.poses.value;
  REQUIRE(poses.size() >= 2);
  // (Its random SimCC scores are low: not every instance gets an id.)
  for(const auto& p : poses)
    CHECK(p.keypoints.size() == 17);
}

// body_params smoothing kinds: only the 2*pi-periodic MHR parameters are
// unwrapped as angles. The distributed twists / spine bends (fractional
// weights over several joints) are linear: wrapping them snaps the mesh.
// With a Body Model the kinds come from its parameter transform; without,
// from the hardcoded equivalent, which must agree.
TEST_CASE("InstantHMR: body params smoothing kinds follow the rig", "[onnx][pose][instanthmr][mhr][model]")
{
  const auto bin = TestPaths::mhrBin(6);
  if(!TestPaths::haveAll({hmr_model, body_jpg, bin}))
    SKIP("models or image not found");
  Harness h{body_jpg, "", hmr_model};
  h.node.inputs.min_confidence.value = 0.f;
  h.run();
  const auto k0 = h.node.paramKindsForTest();
  REQUIRE(k0.size() == 249);
  const std::vector<std::uint8_t> fallback(k0.begin(), k0.end());
  for(int p : {3, 4, 5, 6, 64, 129, 151})
    CHECK(fallback[p] == Onnx::ParamAngle);
  // spine_twist0, *_twist, the upleg/lowleg twists; root t; flexible; scales.
  for(int p : {0, 2, 7, 9, 17, 24, 33, 50, 54, 59, 63, 130, 135, 136, 203})
    CHECK(fallback[p] == Onnx::ParamLinear);
  for(int p = 204; p < 249; p++)
    CHECK(fallback[p] == Onnx::ParamStatic);

  // From the file: identical.
  h.setBodyModel(bin, TestPaths::slurp(bin));
  h.node.inputs.draw_mesh.value = true;
  h.run();
  const auto k1 = h.node.paramKindsForTest();
  REQUIRE(k1.size() == 249);
  CHECK(std::equal(k1.begin(), k1.end(), fallback.begin()));
  const auto model = Onnx::Mhr::Model::loadFile(bin);
  REQUIRE(model);
  const auto per = Onnx::Mhr::periodicParams(*model);
  int n_periodic = 0;
  for(bool b : per)
    n_periodic += b;
  CHECK(n_periodic == 113);
}

// The Body Model reloads when the bytes change under the same file name (a
// re-export), and a failed load is retried with the new bytes.
TEST_CASE("InstantHMR: the body model reloads on new bytes", "[onnx][pose][instanthmr][mhr][model]")
{
  const auto bin6 = TestPaths::mhrBin(6);
  const auto bin3 = TestPaths::mhrBin(3);
  if(!TestPaths::haveAll({hmr_model, body_jpg, bin6, bin3}))
    SKIP("models or image not found");
  Harness h{body_jpg, "", hmr_model};
  h.node.inputs.min_confidence.value = 0.f;
  h.node.inputs.data_format.value = KF::MeshVertices;
  const auto m6 = Onnx::Mhr::Model::loadFile(bin6);
  const auto m3 = Onnx::Mhr::Model::loadFile(bin3);
  REQUIRE(m6);
  REQUIRE(m3);
  h.setBodyModel("body.mhrbin", std::string(4096, 'x')); // broken first
  h.run();
  CHECK(h.node.outputs.geometry.value.empty());
  h.setBodyModel("body.mhrbin", TestPaths::slurp(bin6)); // fixed, same name
  h.run();
  CHECK(h.node.outputs.geometry.value.size() == size_t(m6->numVertices()) * 3);
  h.setBodyModel("body.mhrbin", TestPaths::slurp(bin3)); // re-exported, same name
  h.run();
  CHECK(h.node.outputs.geometry.value.size() == size_t(m3->numVertices()) * 3);
}

// ---------------------------------------------------------------------------
// InstantHMR on a real video (one person walking, then gesturing; frames from
// $ONNX_TEST_INSTANTHMR_VIDEO, SKIP when absent): the whole node over time,
// person detector + InstantHMR + the LOD 3 mesh. Checks that every frame has
// a pose, that nothing is ever non-finite, that smoothing lowers the jitter of
// the mesh and of the translation, that the mesh stays on the person (its
// MHR70 keypoints, projected, vs the 2D keypoints), and that Track IDs keep
// one id. With ONNX_TEST_DUMP_DIR set, also dumps 5 unsmoothed frames with
// their detector boxes, to compare with upstream Python / ORT on the same
// boxes.
namespace
{
std::vector<std::string> videoFrames()
{
  std::vector<std::string> out;
  const std::filesystem::path dir = TestPaths::instantHmrVideo();
  std::error_code ec;
  for(const auto& e : std::filesystem::directory_iterator(dir, ec))
    if(e.path().extension() == ".jpg" || e.path().extension() == ".png")
      out.push_back(e.path().string());
  std::sort(out.begin(), out.end());
  return out;
}

bool allFinite(std::span<const float> v)
{
  return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

bool poseFinite(const OnnxModels::DetectedPose& p)
{
  for(const auto* ks : {&p.keypoints, &p.world})
    for(const auto& k : *ks)
      if(!std::isfinite(k.x) || !std::isfinite(k.y) || !std::isfinite(k.z)
         || !std::isfinite(k.confidence))
        return false;
  return std::isfinite(p.mean_confidence) && allFinite(p.translation)
         && allFinite(p.body_params) && allFinite(p.rig_offset);
}

struct VideoRun
{
  std::vector<std::optional<OnnxModels::DetectedPose>> pose; // primary, per frame
  std::vector<std::vector<float>> verts;                      // MeshVertices
  std::vector<int> ids;       // Track IDs: the largest instance's id (-2: none)
  int non_finite = 0;
};

VideoRun runVideo(
    const std::vector<std::string>& frames, const std::string& bin, bool smoothing,
    bool track)
{
  VideoRun r;
  Harness h{frames.front(), person_det, hmr_model};
  h.setBodyModel(bin, TestPaths::slurp(bin));
  h.node.inputs.smoothing.value = smoothing;
  h.node.inputs.track_ids.value = track;
  h.node.inputs.max_instances.value = 3;
  h.node.inputs.data_format.value = KF::MeshVertices;
  h.node.inputs.mesh_space.value = OnnxModels::MeshSpace::Camera;
  for(const auto& f : frames)
  {
    h.image = QImage(QString::fromStdString(f)).convertToFormat(QImage::Format_RGBA8888);
    h.run();
    r.pose.push_back(h.node.outputs.detection.value);
    r.verts.push_back(h.node.outputs.geometry.value);
    if(!allFinite(h.node.outputs.geometry.value)
       || !allFinite(h.node.outputs.poses_geometry.value)
       || (r.pose.back() && !poseFinite(*r.pose.back())))
      r.non_finite++;
    for(const auto& p : h.node.outputs.poses.value)
      if(!poseFinite(p))
        r.non_finite++;
    int id = -2;
    float best = 0;
    for(const auto& p : h.node.outputs.poses.value)
      if(p.box.w * p.box.h > best)
      {
        best = p.box.w * p.box.h;
        id = p.track_id;
      }
    r.ids.push_back(id);
  }
  return r;
}

// Mean |a - b| over xyz triples.
double meanDist(std::span<const float> a, std::span<const float> b)
{
  if(a.size() != b.size() || a.empty())
    return 0;
  double s = 0;
  for(size_t i = 0; i + 2 < a.size(); i += 3)
    s += std::sqrt(
        double(a[i] - b[i]) * (a[i] - b[i]) + double(a[i + 1] - b[i + 1]) * (a[i + 1] - b[i + 1])
        + double(a[i + 2] - b[i + 2]) * (a[i + 2] - b[i + 2]));
  return s / (a.size() / 3);
}
// Mean |a - 2b + c| over xyz triples: the jerk, blind to steady motion.
double meanAccel(std::span<const float> a, std::span<const float> b, std::span<const float> c)
{
  if(a.size() != b.size() || a.size() != c.size() || a.empty())
    return 0;
  double s = 0;
  for(size_t i = 0; i + 2 < a.size(); i += 3)
  {
    double d2 = 0;
    for(int k = 0; k < 3; k++)
    {
      const double d = double(a[i + k]) - 2. * b[i + k] + c[i + k];
      d2 += d * d;
    }
    s += std::sqrt(d2);
  }
  return s / (a.size() / 3);
}
}

TEST_CASE("InstantHMR: temporal behaviour on a video", "[onnx][pose][instanthmr][mhr][video][model]")
{
  const auto bin = TestPaths::mhrBin(3);
  const auto frames = videoFrames();
  if(!TestPaths::haveAll({hmr_model, person_det, bin}) || frames.size() < 20)
    SKIP("models or video frames not found");
  const int T = static_cast<int>(frames.size());
  auto model = Mhr::Model::loadFile(bin);
  REQUIRE(model);
  Mhr::Workspace ws(*model);

  const auto off = runVideo(frames, bin, false, false);
  const auto on = runVideo(frames, bin, true, false);
  const auto trk = runVideo(frames, bin, true, true);
  const QImage first(QString::fromStdString(frames.front()));
  const int W = first.width(), H = first.height();

  // 1. A pose on every frame, nothing non-finite anywhere.
  for(const auto* r : {&off, &on})
  {
    int missing = 0;
    for(int t = 0; t < T; t++)
      missing += !r->pose[t] || r->verts[t].size() != size_t(model->numVertices()) * 3;
    CHECK(missing == 0);
    CHECK(r->non_finite == 0);
  }
  CHECK(trk.non_finite == 0);

  // Motion per frame from the unsmoothed 2D body joints (0-20, px): splits
  // the clip into "still" and "moving" frames.
  std::vector<double> speed(T, 0.);
  for(int t = 1; t < T; t++)
  {
    const auto& a = off.pose[t]->keypoints;
    const auto& b = off.pose[t - 1]->keypoints;
    double s = 0;
    for(int k = 0; k <= 20; k++)
      s += std::hypot((a[k].x - b[k].x) * W, (a[k].y - b[k].y) * H);
    speed[t] = s / 21;
  }
  auto sorted = speed;
  std::sort(sorted.begin() + 1, sorted.end());
  const double split = sorted[T / 2]; // median: half still-ish, half moving

  // 2. Jitter, smoothing off vs on: frame-to-frame mean vertex displacement
  // (d1) and its second difference (d2: jitter without the steady motion),
  // of the mesh and of the translation.
  struct Jit
  {
    double v1 = 0, v2 = 0, t1 = 0, t2 = 0;
    int n = 0;
  };
  auto jitter = [&](const VideoRun& r, bool still) {
    Jit j;
    for(int t = 2; t < T; t++)
    {
      if((speed[t] <= split) != still)
        continue;
      const auto &p0 = *r.pose[t], &p1 = *r.pose[t - 1], &p2 = *r.pose[t - 2];
      j.v1 += meanDist(r.verts[t], r.verts[t - 1]);
      j.v2 += meanAccel(r.verts[t], r.verts[t - 1], r.verts[t - 2]);
      j.t1 += meanDist(p0.translation, p1.translation);
      j.t2 += meanAccel(p0.translation, p1.translation, p2.translation);
      j.n++;
    }
    if(j.n)
      for(double* x : {&j.v1, &j.v2, &j.t1, &j.t2})
        *x /= j.n;
    return j;
  };
  for(bool still : {true, false})
  {
    const Jit a = jitter(off, still), b = jitter(on, still);
    INFO((still ? "still" : "moving") << " frames, mm/frame off -> on: mesh d1 " << 1e3 * a.v1
         << " -> " << 1e3 * b.v1 << ", d2 " << 1e3 * a.v2 << " -> " << 1e3 * b.v2
         << "; translation d1 " << 1e3 * a.t1 << " -> " << 1e3 * b.t1 << ", d2 "
         << 1e3 * a.t2 << " -> " << 1e3 * b.t2);
    REQUIRE(a.n >= 5);
    CHECK(b.v1 < a.v1);
    CHECK(b.v2 < a.v2);
    CHECK(b.t1 < a.t1);
    CHECK(b.t2 < a.t2);
  }

  // 3. The mesh stays on the person: its MHR70 keypoints (exact readout),
  // placed at the rig origin and projected through the node's camera, vs the
  // node's 2D keypoints. Every frame, smoothed and not.
  for(const auto* r : {&off, &on})
  {
    double mean = 0;
    for(int t = 0; t < T; t++)
    {
      MeshRef mr;
      REQUIRE(mr.eval(*model, ws, *r->pose[t]));
      const auto& kps = r->pose[t]->keypoints;
      double e = 0;
      int n = 0;
      for(int k = 0; k < 70; k++)
      {
        if(kps[k].confidence <= 0.f)
          continue;
        const auto uv = project(mr.kp.data() + 3 * k, W, H);
        e += std::hypot(uv[0] - kps[k].x * W, uv[1] - kps[k].y * H);
        n++;
      }
      REQUIRE(n > 0);
      e /= n;
      mean += e / T;
      INFO("frame " << t << ", smoothing " << (r == &on ? "on" : "off"));
      // Unsmoothed: the model's own 2D and 3D heads agree within 25 px on
      // every frame (1280 px tall). Smoothed: the 3D payload's One-Euro
      // (upstream's 1 Hz rest cutoff) lags the lighter 2D filter on the fast
      // hand gestures at the end of the clip, hence the looser per-frame
      // bound; the clip mean stays tight.
      CHECK(e < (r == &on ? 35.0 : 25.0));
    }
    CHECK(mean < 20.0);
  }
  // 4. Track IDs: one person, one id, on every frame.
  int no_id = 0, switches = 0;
  for(int t = 0; t < T; t++)
  {
    no_id += trk.ids[t] < 0;
    if(t > 0 && trk.ids[t] >= 0 && trk.ids[t - 1] >= 0 && trk.ids[t] != trk.ids[t - 1])
      switches++;
  }
  CHECK(switches == 0);
  CHECK(no_id <= 3); // min_hits confirmation at the start only

  // Parity dump for the Python reference (upstream preprocessing + ORT).
  if(const auto dump = TestPaths::dumpDir(); !dump.empty())
  {
    QJsonArray arr;
    for(int t : {0, T / 5, 2 * T / 5, 3 * T / 5, 4 * T / 5})
    {
      Harness bx{frames[t], person_det};
      bx.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
      bx.node.inputs.detection_class.value = 0;
      bx.node.inputs.smoothing.value = false;
      bx.run();
      Harness h{frames[t], person_det, hmr_model};
      h.node.inputs.smoothing.value = false;
      h.run();
      REQUIRE(!bx.node.outputs.poses.value.empty());
      REQUIRE(h.node.outputs.detection.value.has_value());
      // The two-stage path crops the top-scoring detection.
      const auto& d = *std::max_element(
          bx.node.outputs.poses.value.begin(), bx.node.outputs.poses.value.end(),
          [](const auto& a, const auto& b) { return a.mean_confidence < b.mean_confidence; });
      const auto& p = *h.node.outputs.detection.value;
      QJsonObject o;
      o["frame"] = QString::fromStdString(frames[t]);
      o["box"] = QJsonArray{d.box.x * W, d.box.y * H, (d.box.x + d.box.w) * W, (d.box.y + d.box.h) * H};
      QJsonArray j2, j3, bp, tr, ro;
      for(const auto& k : p.keypoints)
        j2.append(k.x * W), j2.append(k.y * H);
      for(int k = 0; k < 70; k++)
      {
        j3.append(p.world[k].x + p.rig_offset[0]);
        j3.append(p.world[k].y + p.rig_offset[1]);
        j3.append(p.world[k].z + p.rig_offset[2]);
      }
      for(float v : p.body_params)
        bp.append(v);
      for(int c = 0; c < 3; c++)
        tr.append(p.translation[c] - p.rig_offset[c]);
      o["joints_2d"] = j2;
      o["joints_3d"] = j3;
      o["cam_trans"] = tr;
      o["params"] = bp;
      arr.append(o);
    }
    QFile f(QString::fromStdString(dump) + "/instanthmr_video_node.json");
    REQUIRE(f.open(QIODevice::WriteOnly));
    f.write(QJsonDocument(arr).toJson());
  }
}

// ---------------------------------------------------------------------------
// Benchmark (hidden: run with "[.bench]"; the provider follows
// SCORE_ONNX_FORCE_PROVIDER, cpu / cuda / tensorrt). Median and p95 over
// BENCH_ITERS runs (default 200) after a warm-up, per stage of the InstantHMR
// pipeline and for the whole node frame, at 1280x720 and 1920x1080, one
// person (the video frame, single path) and three (body.jpg, Track IDs).
namespace
{
struct BenchStats
{
  double med = 0, p95 = 0;
};

int benchIters()
{
  const char* e = std::getenv("BENCH_ITERS");
  return e && *e ? std::max(5, std::atoi(e)) : 200;
}

template <typename F>
BenchStats timeIt(F&& f, int iters, int warm = 20)
{
  for(int i = 0; i < warm; i++)
    f();
  std::vector<double> ms(iters);
  for(int i = 0; i < iters; i++)
  {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    ms[i] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();
  }
  std::sort(ms.begin(), ms.end());
  return {ms[iters / 2], ms[std::min(iters - 1, int(iters * 0.95))]};
}

void report(const std::string& what, BenchStats s)
{
  const char* prov = std::getenv("SCORE_ONNX_FORCE_PROVIDER");
  std::fprintf(
      stderr, "bench [%s] %-58s median %8.3f ms  p95 %8.3f ms\n",
      prov && *prov ? prov : "default", what.c_str(), s.med, s.p95);
}

// src scaled into a W x H grey canvas, aspect kept, centred.
QImage fitInto(const QImage& src, int W, int H)
{
  QImage c(W, H, QImage::Format_RGBA8888);
  c.fill(QColor(90, 90, 90));
  const QImage s = src.scaled(W, H, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  QPainter p(&c);
  p.drawImage((W - s.width()) / 2, (H - s.height()) / 2, s);
  p.end();
  return c;
}
}

TEST_CASE("InstantHMR pipeline benchmark", "[.bench][instanthmr]")
{
  const auto frames = videoFrames();
  const auto bin1 = TestPaths::mhrBin(1), bin3 = TestPaths::mhrBin(3);
  if(!TestPaths::haveAll({hmr_model, person_det, body_jpg, bin1, bin3}) || frames.empty())
    SKIP("models or images not found");
  const int iters = benchIters();
  REQUIRE(OnnxModels::initOnnxRuntime());
  const QImage one_src
      = QImage(QString::fromStdString(frames.front())).convertToFormat(QImage::Format_RGBA8888);
  const QImage three_src
      = QImage(QString::fromStdString(body_jpg)).convertToFormat(QImage::Format_RGBA8888);

  // --- Stage: detector inference alone (its fixed 416 input). ---
  {
    const auto bytes = TestPaths::slurp(person_det);
    Onnx::OnnxRunContext ctx{bytes, person_det};
    const auto& spec = ctx.readModelSpec();
    std::vector<int64_t> shape = spec.inputs[0].shape;
    shape[0] = 1;
    size_t n = 1;
    for(auto d : shape)
      n *= size_t(d);
    std::vector<float> in(n, 0.5f);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    report("detector infer (yolox-tiny 416)", timeIt([&] {
             Ort::Value ins[1]{Ort::Value::CreateTensor<float>(
                 mem, in.data(), in.size(), shape.data(), shape.size())};
             Ort::Value outs[1]{Ort::Value{nullptr}};
             ctx.infer(spec, ins, outs);
           }, iters));
  }

  // --- Stage: InstantHMR inference, batch 1 and 3. ---
  {
    const auto bytes = TestPaths::slurp(hmr_model);
    Onnx::OnnxRunContext ctx{bytes, hmr_model};
    const auto& spec = ctx.readModelSpec();
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // A new (or a previously seen) batch size: the cost of the first run
    // after the person count changes.
    {
      std::string seq;
      for(int N : {1, 1, 2, 3, 1, 2, 3, 5, 1})
      {
        std::vector<float> img(size_t(N) * 3 * 224 * 224, 0.1f), cliff(size_t(N) * 3, 0.2f);
        std::array<int64_t, 4> is{N, 3, 224, 224};
        std::array<int64_t, 2> cs{N, 3};
        const auto t0 = std::chrono::steady_clock::now();
        Ort::Value ins[2]{
            Ort::Value::CreateTensor<float>(mem, img.data(), img.size(), is.data(), 4),
            Ort::Value::CreateTensor<float>(mem, cliff.data(), cliff.size(), cs.data(), 2)};
        Ort::Value outs[5]{
            Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
            Ort::Value{nullptr}, Ort::Value{nullptr}};
        ctx.infer(spec, ins, outs);
        seq += " N" + std::to_string(N) + ":"
               + std::to_string(int(std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t0)
                                        .count()))
               + "ms";
      }
      std::fprintf(stderr, "bench InstantHMR batch-size switches, first run each:%s\n", seq.c_str());
    }
    for(int N : {1, 3, 1})
    {
      std::vector<float> img(size_t(N) * 3 * 224 * 224, 0.1f), cliff(size_t(N) * 3, 0.2f);
      std::array<int64_t, 4> is{N, 3, 224, 224};
      std::array<int64_t, 2> cs{N, 3};
      report("InstantHMR infer batch " + std::to_string(N), timeIt([&] {
               Ort::Value ins[2]{
                   Ort::Value::CreateTensor<float>(mem, img.data(), img.size(), is.data(), 4),
                   Ort::Value::CreateTensor<float>(mem, cliff.data(), cliff.size(), cs.data(), 2)};
               Ort::Value outs[5]{
                   Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr},
                   Ort::Value{nullptr}, Ort::Value{nullptr}};
               ctx.infer(spec, ins, outs);
             }, iters));
    }
  }

  // Real body parameters of three people (unsmoothed node run on body.jpg).
  std::vector<OnnxModels::DetectedPose> people;
  {
    Harness h{body_jpg, person_det, hmr_model};
    h.node.inputs.track_ids.value = true;
    h.node.inputs.max_instances.value = 3;
    h.node.inputs.smoothing.value = false;
    for(int i = 0; i < 4; i++)
      h.run();
    people = h.node.outputs.poses.value;
    REQUIRE(people.size() >= 3);
    people.resize(3);
  }

  // --- Stages: MHR evaluation, triangle soup, Draw Mesh. ---
  for(const auto& bin : {bin3, bin1})
  {
    auto model = Mhr::Model::loadFile(bin);
    REQUIRE(model);
    const std::string lod = "LOD" + std::to_string(model->lod());
    std::vector<Mhr::Workspace> ws(3);
    std::vector<std::vector<float>> verts(3, std::vector<float>(model->numVertices() * 3));
    for(auto& w : ws)
      w.prepare(*model);
    for(int np : {1, 3})
    {
      const std::string who = std::to_string(np) + (np == 1 ? " person" : " persons");
      report("MHR evaluate " + lod + ", " + who, timeIt([&] {
               for(int p = 0; p < np; p++)
               {
                 const std::span<const float> bp(people[p].body_params);
                 Mhr::evaluate(
                     *model, bp.first(204), bp.subspan(204, 45), {}, ws[p],
                     {verts[p], {}, {}, Mhr::KeypointMethod::Exact});
                 const auto o = rigOrigin(people[p]);
                 Onnx::MeshOps::translate(verts[p], o.data(), verts[p]);
               }
             }, iters));
      std::vector<float> soup(size_t(model->numFaces()) * 9 * np);
      report("triangle soup " + lod + ", " + who, timeIt([&] {
               float* out = soup.data();
               for(int p = 0; p < np; p++)
                 out = Onnx::MeshOps::writeTriangles(verts[p], model->faces(), true, out);
             }, iters));
      for(auto [W, H] : {std::pair{1280, 720}, std::pair{1920, 1080}})
      {
        std::vector<uint8_t> img(size_t(W) * H * 4, 100);
        Onnx::MeshOps::Raster r;
        const float f = std::hypot(float(W), float(H));
        const uint8_t rgb[3] = {200, 120, 60};
        report(
            "Draw Mesh " + lod + ", " + who + " @" + std::to_string(H) + "p",
            timeIt([&] {
              r.begin(W, H);
              for(int p = 0; p < np; p++)
                r.draw(verts[p], model->faces(), f, 0.5f * W, 0.5f * H, rgb);
              r.composite(img.data(), W * 4, 0.6f);
            }, iters));
      }
    }
  }

  // --- A changing person count (Track IDs): 1 then 3 people, switching
  //     every 10 frames. The worst frames show any per-batch-size re-plan. ---
  {
    const QImage one = fitInto(one_src, 1280, 720), three = fitInto(three_src, 1280, 720);
    Harness h{body_jpg, person_det, hmr_model};
    h.node.inputs.track_ids.value = true;
    h.node.inputs.max_instances.value = 3;
    h.node.inputs.hold_frames.value = 0;
    h.node.inputs.track_memory.value = 1;
    int frame = 0;
    std::vector<double> ms;
    for(int i = 0; i < std::max(iters, 60); i++)
    {
      h.image = (frame++ / 10) % 2 ? three : one;
      const auto t0 = std::chrono::steady_clock::now();
      h.run();
      ms.push_back(std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count());
    }
    // Past the first switches (warm-up: a growing batch re-plans once).
    std::vector<double> tail(ms.begin() + 40, ms.end());
    std::sort(tail.begin(), tail.end());
    std::fprintf(
        stderr,
        "bench node frame, person count 1<->3 every 10 frames @720p: median %.2f ms, "
        "p95 %.2f ms, max %.2f ms (after 40 warm-up frames)\n",
        tail[tail.size() / 2], tail[size_t(tail.size() * 0.95)], tail.back());
  }

  // --- The whole node frame. ---
  for(auto [W, H] : {std::pair{1280, 720}, std::pair{1920, 1080}})
  {
    const QImage one = fitInto(one_src, W, H), three = fitInto(three_src, W, H);
    const std::string res = "@" + std::to_string(H) + "p";
    // Detector stage alone: Box Detection (letterbox + infer + decode + draw).
    {
      Harness h{body_jpg, person_det};
      h.image = three;
      h.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
      h.node.inputs.detection_class.value = 0;
      report("node frame: Box Detection only " + res, timeIt([&] { h.run(); }, iters));
    }
    for(int np : {1, 3})
    {
      for(const std::string bin : {std::string{}, bin3, bin1})
      {
        Harness h{body_jpg, person_det, hmr_model};
        h.image = np == 1 ? one : three;
        h.node.inputs.track_ids.value = np > 1;
        h.node.inputs.max_instances.value = np;
        std::string what = "node frame: det + InstantHMR, " + std::to_string(np)
                           + (np == 1 ? " person " : " persons ") + res;
        if(bin.empty())
          h.node.inputs.data_format.value = KF::CameraXYZArray;
        else
        {
          h.setBodyModel(bin, TestPaths::slurp(bin));
          h.node.inputs.data_format.value = KF::MeshTriangles;
          h.node.inputs.draw_mesh.value = true;
          what += bin == bin3 ? " + LOD3 mesh/soup/draw" : " + LOD1 mesh/soup/draw";
        }
        const auto s = timeIt([&] { h.run(); }, iters);
        CHECK(h.node.outputs.count.value == (np > 1 ? np : 0));
        report(what, s);
      }
    }
  }
}
