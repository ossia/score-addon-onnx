// Pose Detector body and face pipelines: the whole-frame letterbox, MoveNet's
// crop, the SSD detectors' weighted NMS, the YOLO-pose decode, the face and
// whole-body remaps, and the package's models on the test photos. The model
// cases SKIP without the package (ONNX_TEST_POSE_PACKAGE) or the photos
// (ONNX_TEST_IMAGES: body.jpg with several people, face.png with one face).
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <Onnx/helpers/Yolo.hpp>
#include <OnnxModels/PoseDetector_internal.hpp>

#include <QImage>
#include <QPainter>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{
const std::string package = TestPaths::posePackage();
const std::string body_jpg = TestPaths::images() + "/body.jpg";
const std::string face_png = TestPaths::images() + "/face.png";

std::string det(const char* f)
{
  return package + "/detectors/" + f;
}
std::string lm(const char* f)
{
  return package + "/landmarks/" + f;
}
std::string ss(const char* f)
{
  return package + "/single_stage/" + f;
}

struct Harness
{
  OnnxModels::PoseDetector node;
  QImage image;
  std::string det_name, det_bytes, lm_name, lm_bytes;

  Harness(const std::string& det, const std::string& lm)
      : det_name{det}
      , det_bytes{det.empty() ? std::string{} : TestPaths::slurp(det)}
      , lm_name{lm}
      , lm_bytes{TestPaths::slurp(lm)}
  {
    node.inputs.det_model.file.bytes = det_bytes;
    node.inputs.det_model.file.filename = det_name;
    node.inputs.model.file.bytes = lm_bytes;
    node.inputs.model.file.filename = lm_name;
    node.inputs.smoothing.value = false;
    node.inputs.hold_frames.value = 0;
    inlineWorker(node);
  }

  void run(const QImage& img)
  {
    image = img.convertToFormat(QImage::Format_RGBA8888);
    auto& t = node.inputs.image.texture;
    t.bytes = image.bits();
    t.width = image.width();
    t.height = image.height();
    t.changed = true;
    node();
  }

  std::vector<OnnxModels::DetectedPose> poses() const
  {
    if(node.inputs.track_ids.value && !node.outputs.poses.value.empty())
      return node.outputs.poses.value;
    if(node.outputs.detection.value)
      return {*node.outputs.detection.value};
    return {};
  }
};

// `photo` drawn at `r` (pixels) on a grey frame of w x h.
QImage paste(const std::string& photo, int w, int h, QRectF r)
{
  QImage frame(w, h, QImage::Format_RGB888);
  frame.fill(QColor(110, 110, 110));
  QPainter p(&frame);
  p.setRenderHint(QPainter::SmoothPixmapTransform);
  p.drawImage(r, QImage(QString::fromStdString(photo)));
  return frame;
}

// The mean position of the confident keypoints, normalized.
QPointF centroid(const OnnxModels::DetectedPose& pose)
{
  double x = 0, y = 0;
  int n = 0;
  for(const auto& k : pose.keypoints)
    if(k.confidence > 0.3f)
    {
      x += k.x;
      y += k.y;
      n++;
    }
  return n ? QPointF{x / n, y / n} : QPointF{-1, -1};
}
}

TEST_CASE("The whole-frame letterbox keeps every pixel of the frame", "[pose]")
{
  // A 640x480 frame in a 640x640 input: 80 px bands above and below.
  const auto M = Onnx::ROI::wholeFrameFitAffine(640, 480, 640, 640);
  const auto tl = Onnx::ROI::applyAffine(M, 0, 80);
  const auto br = Onnx::ROI::applyAffine(M, 640, 560);
  CHECK(tl.x == Catch::Approx(0).margin(1e-3));
  CHECK(tl.y == Catch::Approx(0).margin(1e-3));
  CHECK(br.x == Catch::Approx(640).margin(1e-3));
  CHECK(br.y == Catch::Approx(480).margin(1e-3));

  // A wide frame: bands left and right of a 1280x480 frame in 256x256.
  const auto W = Onnx::ROI::wholeFrameFitAffine(1280, 480, 256, 256);
  CHECK(Onnx::ROI::applyAffine(W, 0, 0).y == Catch::Approx(-((256 - 96) / 2.) * 5).margin(1e-2));
  CHECK(Onnx::ROI::applyAffine(W, 256, 128).x == Catch::Approx(1280).margin(1e-2));
}

TEST_CASE("MoveNet's crop is centred on the hips and covers the body", "[pose]")
{
  std::vector<Onnx::ROI::ScoredPoint> kp(17, {0, 0, 0});
  auto set = [&](int i, float x, float y) { kp[i] = {x, y, 0.9f}; };
  set(5, 280, 200); // shoulders
  set(6, 360, 200);
  set(11, 290, 300); // hips
  set(12, 350, 300);
  set(15, 270, 460); // ankles
  set(16, 370, 460);
  Onnx::ROI::Rect r;
  REQUIRE(Onnx::ROI::moveNetCropRect([&](int i) { return kp[i]; }, 1280, 720, r));
  CHECK(r.cx == Catch::Approx(320));
  CHECK(r.cy == Catch::Approx(300));
  // Torso: 100 px above the hips * 1.9; body: 160 px below * 1.2 -> 192.
  CHECK(r.w == Catch::Approx(2 * 192));
  CHECK(r.angle == 0.f);

  // No torso, no crop: the whole frame again.
  kp[5].score = kp[6].score = 0.f;
  CHECK_FALSE(Onnx::ROI::moveNetCropRect([&](int i) { return kp[i]; }, 1280, 720, r));
}

TEST_CASE("Weighted NMS averages the anchors of one subject", "[pose]")
{
  using Onnx::Detection::Detection;
  std::vector<Detection> dets{
      Detection{0.50f, 0.50f, 0.2f, 0.2f, 0.9f, -1, {{0.50f, 0.40f}}},
      Detection{0.52f, 0.50f, 0.2f, 0.2f, 0.3f, -1, {{0.54f, 0.40f}}},
      Detection{0.10f, 0.10f, 0.1f, 0.1f, 0.6f, -1, {{0.10f, 0.05f}}}};
  const auto out = Onnx::Detection::weightedNms(dets, 0.3f);
  REQUIRE(out.size() == 2);
  CHECK(out[0].score == Catch::Approx(0.9f));
  CHECK(out[0].xc == Catch::Approx((0.9 * 0.50 + 0.3 * 0.52) / 1.2));
  CHECK(out[0].keypoints[0].x == Catch::Approx((0.9 * 0.50 + 0.3 * 0.54) / 1.2));
  CHECK(out[1].xc == Catch::Approx(0.10f));
}

TEST_CASE("YOLO-pose decode: NMS over the anchors, every keypoint kept", "[pose]")
{
  // A v8 head [1, 5+3K, A] with K = 2: two anchors on one person (IoU ~0.8),
  // one elsewhere, the rest empty (more anchors than channels, as in a real
  // head). Channel-major: data[f * A + a].
  constexpr int K = 2, A = 16, F = 5 + 3 * K;
  float data[F * A]{};
  auto set = [&](int a, std::initializer_list<float> v) {
    int f = 0;
    for(float x : v)
      data[f++ * A + a] = x;
  };
  set(0, {100, 100, 50, 100, 0.9f, 90, 60, 0.95f, 110, 140, 0.1f});
  set(1, {104, 100, 50, 100, 0.8f, 92, 61, 0.9f, 111, 141, 0.2f});
  set(2, {400, 300, 60, 120, 0.7f, 400, 250, 0.8f, 400, 350, 0.6f});
  const int64_t shape[3] = {1, F, A};
  REQUIRE(OnnxModels::initOnnxRuntime());
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value out[1] = {Ort::Value::CreateTensor<float>(mem, data, F * A, shape, 3)};

  std::vector<OnnxModels::Yolo::YOLO_pose::Decoded> poses;
  REQUIRE(OnnxModels::Yolo::YOLO_pose::decode(out, poses, 0.3f) == K);
  REQUIRE(poses.size() == 2);
  CHECK(poses[0].confidence == Catch::Approx(0.9f));
  CHECK(poses[1].confidence == Catch::Approx(0.7f));
  // The low-confidence keypoint is still there, with its confidence.
  REQUIRE(poses[0].keypoints.size() == 3 * K);
  CHECK(poses[0].keypoints[3] == Catch::Approx(110));
  CHECK(poses[0].keypoints[5] == Catch::Approx(0.1f));
  CHECK(poses[0].box.x == Catch::Approx(75));
}

TEST_CASE("WFLW-98 and COCO-WholeBody faces and hands remap", "[pose]")
{
  using namespace Onnx::Skel;
  const auto wflw = mappingFor(SourceSkeleton::Wflw98, TargetSkeleton::Dlib68);
  REQUIRE(wflw.size() == 68);
  CHECK(wflw[8].idx[0] == 16);  // chin: the middle of the 33 contour points
  CHECK(wflw[36].idx[0] == 60); // eye corners
  CHECK(wflw[45].idx[0] == 72);
  CHECK(wflw[67].idx[0] == 95);

  const auto face = mappingFor(SourceSkeleton::Wholebody133, TargetSkeleton::Dlib68);
  REQUIRE(face.size() == 68);
  CHECK(face[0].idx[0] == 23);
  CHECK(face[67].idx[0] == 90);
  const auto hand = mappingFor(SourceSkeleton::Wholebody133, TargetSkeleton::Hand21);
  REQUIRE(hand.size() == 21);
  CHECK(hand[0].idx[0] == 112);
  // The body goes through the COCO-17 tables.
  const auto body = mappingFor(SourceSkeleton::Wholebody133, TargetSkeleton::Halpe26);
  const auto coco = mappingFor(SourceSkeleton::Coco17, TargetSkeleton::Halpe26);
  CHECK(body.data() == coco.data());
}

// A person at the right edge of a wide frame: the single-stage models see the
// whole frame letterboxed, not its centre square.
TEST_CASE("Single-stage models see people at the frame's edges", "[onnx][pose][model]")
{
  for(const auto& m : {ss("ss-yolov8n-pose.onnx"), ss("ss-yolo26n-pose.onnx"),
                       ss("ss-rtmo-s.onnx"), ss("ss-movenet-thunder.onnx")})
  {
    DYNAMIC_SECTION(m)
    {
      if(!TestPaths::haveAll({m, body_jpg}))
        SKIP("model or image not found");
      Harness h{"", m};
      h.node.inputs.track_ids.value = m.find("movenet") == std::string::npos;
      h.node.inputs.max_instances.value = 8;
      const QImage photo(QString::fromStdString(body_jpg));
      // Right of the frame's centre 480x480 square, which ends at x = 1040.
      const double pw = 560, ph = photo.height() * pw / photo.width();
      h.run(paste(body_jpg, 1600, 480, QRectF{1040, (480 - ph) / 2, pw, ph}));
      const auto poses = h.poses();
      REQUIRE(!poses.empty());
      for(const auto& p : poses)
        CHECK(centroid(p).x() * 1600 > 1040);
    }
  }
}

// MoveNet finds the person in the whole frame, then crops around them, and
// with Track ROI keeps the crop on them as they move.
TEST_CASE("MoveNet follows a person across the frame", "[onnx][pose][model]")
{
  const auto m = ss("ss-movenet-thunder.onnx");
  if(!TestPaths::haveAll({m, body_jpg}))
    SKIP("model or image not found");
  Harness h{"", m};
  h.node.inputs.track_roi.value = true;
  const QImage photo(QString::fromStdString(body_jpg));
  const double ph = 360, pw = photo.width() * ph / photo.height();
  int followed = 0;
  for(int f = 0; f < 20; f++)
  {
    const double x = 40 + f * (1280 - pw - 80) / 19.;
    h.run(paste(body_jpg, 1280, 720, QRectF{x, 200, pw, ph}));
    const auto poses = h.poses();
    if(poses.size() != 1 || poses[0].keypoints.size() != 17)
      continue;
    const auto c = centroid(poses[0]);
    if(c.x() * 1280 > x && c.x() * 1280 < x + pw && c.y() * 720 > 200
       && c.y() * 720 < 560)
      followed++;
  }
  CHECK(followed == 20);
}

// BlazePose's joints carry MediaPipe's visibility; the pose its presence.
TEST_CASE("BlazePose: visibility per joint, presence per pose", "[onnx][pose][model]")
{
  const auto d = det("det-body-blazepose-224.onnx"), l = lm("lm-body-blazepose-full-195.onnx");
  if(!TestPaths::haveAll({d, l, body_jpg}))
    SKIP("model or image not found");
  Harness h{d, l};
  h.run(QImage(QString::fromStdString(body_jpg)));
  const auto poses = h.poses();
  REQUIRE(poses.size() == 1);
  const auto& p = poses[0];
  REQUIRE(p.keypoints.size() == 33);
  CHECK(p.mean_confidence > 0.5f);
  float lo = 1.f, hi = 0.f;
  for(const auto& k : p.keypoints)
  {
    CHECK(k.confidence >= 0.f);
    CHECK(k.confidence <= 1.f);
    lo = std::min(lo, k.confidence);
    hi = std::max(hi, k.confidence);
  }
  CHECK(hi - lo > 0.1f); // not the same score everywhere
}

// Peppa-Pig emits WFLW's 98 points; the Skeleton control turns them into dlib-68.
TEST_CASE("Peppa-Pig's 98 face points remap to dlib-68", "[onnx][pose][model]")
{
  const auto d = det("det-face-blazeface-256.onnx"), l = lm("lm-face-peppapig.onnx");
  if(!TestPaths::haveAll({d, l, face_png}))
    SKIP("model or image not found");
  Harness h{d, l};
  h.node.inputs.skeleton_type.value = Onnx::Skel::TargetSkeleton::Dlib68;
  h.run(QImage(QString::fromStdString(face_png)));
  const auto poses = h.poses();
  REQUIRE(poses.size() == 1);
  const auto& k = poses[0].keypoints;
  REQUIRE(k.size() == 68);
  // Outer eye corners left to right, both above the mouth corners.
  CHECK(k[36].x < k[45].x);
  CHECK(k[36].y < k[48].y);
  CHECK(k[45].y < k[54].y);
}

// Every face of a group, each with its own crop.
TEST_CASE("Multi-face FaceMesh finds every face", "[onnx][pose][model]")
{
  const auto d = det("det-face-blazeface-256.onnx"), l = lm("lm-face-facemeshv2-478.onnx");
  if(!TestPaths::haveAll({d, l, face_png}))
    SKIP("model or image not found");
  Harness h{d, l};
  h.node.inputs.track_ids.value = true;
  h.node.inputs.max_instances.value = 6;
  QImage frame(1280, 720, QImage::Format_RGB888);
  frame.fill(QColor(90, 100, 110));
  {
    QPainter p(&frame);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    const QImage face(QString::fromStdString(face_png));
    for(int i = 0; i < 3; i++)
      p.drawImage(QRectF{40. + i * 420, 160, 380, 380 * face.height() / double(face.width())}, face);
  }
  h.run(frame);
  const auto poses = h.poses();
  CHECK(poses.size() == 3);
  std::vector<double> xs;
  for(const auto& p : poses)
    xs.push_back(centroid(p).x());
  std::sort(xs.begin(), xs.end());
  for(size_t i = 1; i < xs.size(); i++)
    CHECK(xs[i] - xs[i - 1] > 0.2);
}
