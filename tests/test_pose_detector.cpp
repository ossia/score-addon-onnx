// Pose Detector detection stage on real models and images (BUG-LEDGER P1, P3,
// P4, P6). Each test SKIPs when its model or image is missing.
#include <tests/TestPaths.hpp>
#include <OnnxModels/PoseDetector_internal.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
const std::string presets = [] {
  return TestPaths::models()
         + "/pose-detector";
}();
const std::string package = TestPaths::posePackage();
const std::string libreonnx = TestPaths::libreonnx();
const std::string wild2 = TestPaths::wild() + "/wild2";
const std::string body_jpg = libreonnx + "/ossia-detection-model-pack/test_images/body.jpg";
const std::string hand_jpg
    = TestPaths::ailia() + "/hand_recognition/blazehand/person_hand.jpg";

std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

bool haveAll(std::initializer_list<std::string> paths)
{
  for(const auto& p : paths)
    if(!std::filesystem::exists(p))
      return false;
  return true;
}

struct Harness
{
  OnnxModels::PoseDetector node;
  QImage image;
  // The ports hold views: keep the names and bytes alive.
  std::string det_name, det_bytes, lm_name, lm_bytes;

  Harness(const std::string& img, const std::string& det, const std::string& lm = {})
      : image{QImage(QString::fromStdString(img)).convertToFormat(QImage::Format_RGBA8888)}
      , det_name{det}
      , det_bytes{slurp(det)}
      , lm_name{lm}
      , lm_bytes{lm.empty() ? std::string{} : slurp(lm)}
  {
    node.inputs.det_model.file.bytes = det_bytes;
    node.inputs.det_model.file.filename = det_name;
    node.inputs.model.file.bytes = lm_bytes;
    node.inputs.model.file.filename = lm_name;
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

TEST_CASE("Gold-YOLO head detector gets [0,1] input", "[onnx][pose]")
{
  const auto gold
      = libreonnx + "/423_6DRepNet360/gold_yolo_n_head_post_0277_0.5071_1x3x480x640.onnx";
  const auto bhh = presets + "/detectors/det-bhh-yolox-n-480x640.onnx";
  if(!haveAll({gold, bhh, body_jpg}))
    SKIP("models or image not found");

  // Raw 0-255 saturated it: every row scored 1.0, with inverted boxes.
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

TEST_CASE("YOLOX COCO detectors: both input ranges find the players", "[onnx][pose]")
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
      if(!haveAll({det, body_jpg}))
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

TEST_CASE("BlazePalm 256 uses its five-layer anchors", "[onnx][pose]")
{
  const auto palm = wild2 + "/pose2__blazepalm__blazepalm.onnx";
  if(!haveAll({palm, hand_jpg}))
    SKIP("model or image not found");

  // The palm fills the lower middle of the frame. With the 4-layer anchors the
  // stride-32 detections decoded 112 model px too high, near the top edge.
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

TEST_CASE("Hand landmarks crop the hand class of a body-head-hand detector", "[onnx][pose]")
{
  const auto bhh = package + "/detectors/det-body-yolox-bhh-320.onnx";
  const auto lm = package + "/landmarks/lm-hand-rtmpose-21.onnx";
  if(!haveAll({bhh, lm, body_jpg}))
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

  // Class 0 (a body) used to be cropped for the hand model; a hand is a small
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

// BUG-LEDGER P11: only the 640 px v8 layout (56x8400) and yolo26's 300x57 were
// decoded, and K was fixed to 17.
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


// The shipped single-stage presets decode as before (same poses, same order).
TEST_CASE("YOLO-pose presets find the players", "[onnx][pose]")
{
  for(const auto& m : {package + "/single_stage/ss-yolov8n-pose.onnx",
                       package + "/single_stage/ss-yolov8s-pose.onnx",
                       package + "/single_stage/ss-yolo26n-pose.onnx",
                       package + "/single_stage/ss-yolo26s-pose.onnx"})
  {
    DYNAMIC_SECTION(m)
    {
      if(!haveAll({m, body_jpg}))
        SKIP("model or image not found");
      Harness h{body_jpg, "", m};
      h.node.inputs.track_ids.value = true;
      h.node.inputs.max_instances.value = 8;
      h.run();
      const auto& poses = h.node.outputs.poses.value;
      CHECK(poses.size() >= 2);
      std::string fp;
      for(const auto& p : poses)
      {
        CHECK(p.keypoints.size() == 17);
        fp += std::to_string(p.keypoints.size()) + ":";
        for(const auto& k : p.keypoints)
          fp += std::to_string((int)(k.x * 1000)) + "," + std::to_string((int)(k.y * 1000)) + " ";
        fp += "| ";
      }
      UNSCOPED_INFO("FP " << m << " " << fp);
      std::fprintf(stderr, "FP %s %s\n", m.c_str(), fp.c_str());
    }
  }
}

// BUG-LEDGER P10: with all output dims dynamic (FaceBoxesProd when ORT's
// shape inference is off, e.g. OpenVINO), the model classified as Unknown.
TEST_CASE("FaceBoxes: dynamic output dims are probed", "[onnx][pose]")
{
  const auto model = package + "/detectors/det-face-faceboxes.onnx";
  if(!haveAll({model}))
    SKIP("model not found");
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto bytes = slurp(model);
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


// BUG-LEDGER P2: a dynamic-input multi-class detector was fed 128x128, but it
// reports boxes in its internal 160x128 space, so x came out 1.25x too wide.
TEST_CASE("Dynamic-input multi-class detector: boxes in the image", "[onnx][pose]")
{
  const std::string det
      = TestPaths::wild() + "/pinto/459__yolov9_n_wholebody25_post_0100_1x3x128x160.onnx";
  if(!haveAll({det, body_jpg}))
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
