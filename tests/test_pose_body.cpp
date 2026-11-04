// Pose Detector body and face pipelines: the whole-frame letterbox, MoveNet's
// crop, the SSD detectors' weighted NMS, the YOLO-pose decode, the face and
// whole-body remaps, and the package's models on the test photos. The model
// cases SKIP without the package (ONNX_TEST_POSE_PACKAGE) or the photos
// (ONNX_TEST_IMAGES: body.jpg with several people, face.png with one face).
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <Onnx/helpers/Heatmap.hpp>
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

TEST_CASE("DEIMv2 decode: one class, normalized boxes, no NMS", "[pose]")
{
  // label_xyxy_score [1, 4, 6]: two hands (class 32), a body (0), and a hand
  // under the threshold.
  float data[4 * 6] = {
      32, 0.10f, 0.20f, 0.30f, 0.60f, 0.9f, //
      32, 0.12f, 0.22f, 0.32f, 0.62f, 0.5f, //
      0,  0.00f, 0.00f, 0.50f, 1.00f, 0.95f, //
      32, 0.70f, 0.70f, 0.80f, 0.80f, 0.1f};
  const int64_t shape[3] = {1, 4, 6};
  REQUIRE(OnnxModels::initOnnxRuntime());
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value out[1] = {Ort::Value::CreateTensor<float>(mem, data, 4 * 6, shape, 3)};

  const auto hands = Onnx::Detection::decodeLabelXyxyScore(out, 32, 0.3f);
  // A DETR head predicts each object once: overlapping boxes both stay.
  REQUIRE(hands.size() == 2);
  CHECK(hands[0].class_id == 32);
  CHECK(hands[0].xc == Catch::Approx(0.2f));
  CHECK(hands[0].yc == Catch::Approx(0.4f));
  CHECK(hands[0].w == Catch::Approx(0.2f));
  CHECK(hands[0].h == Catch::Approx(0.4f));
  CHECK(Onnx::Detection::decodeLabelXyxyScore(out, -1, 0.3f).size() == 3);
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
  // COCO-17 is the body's first 17 joints.
  const auto coco17 = mappingFor(SourceSkeleton::Wholebody133, TargetSkeleton::Coco17);
  REQUIRE(coco17.size() == 17);
  CHECK(coco17[0].idx[0] == 0);
  CHECK(coco17[16].idx[0] == 16);
  // The other hand layouts, from the whole body's right hand.
  const auto inter = mappingFor(SourceSkeleton::Wholebody133, TargetSkeleton::InterHand21);
  REQUIRE(inter.size() == 21);
  CHECK(inter[0].idx[0] == 112 + 4);  // thumb tip first
  CHECK(inter[20].idx[0] == 112);     // wrist last
  // Other bodies go through the COCO-17 tables.
  const auto body = mappingFor(SourceSkeleton::Wholebody133, TargetSkeleton::Halpe26);
  const auto coco = mappingFor(SourceSkeleton::Coco17, TargetSkeleton::Halpe26);
  CHECK(body.data() == coco.data());
}

TEST_CASE("Min Size measures the keypoints' box against the frame height", "[pose]")
{
  OnnxModels::DetectedPose p;
  // A 0.1 x 0.2 (normalized) box in a 1280 x 720 frame: 128 x 144 px.
  p.keypoints = {{0.4f, 0.3f, 0.f, 1.f}, {0.5f, 0.5f, 0.f, 1.f}, {0.45f, 0.4f, 0.f, 1.f}};
  CHECK(OnnxModels::poseSize(p, 1280, 720) == Catch::Approx(std::sqrt(128. * 144.) / 720.));
  // Keypoints collapsed on a vertical line: no area, however tall.
  for(auto& k : p.keypoints)
    k.x = 0.4f;
  CHECK(OnnxModels::poseSize(p, 1280, 720) == 0.f);
  CHECK(OnnxModels::poseSize(OnnxModels::DetectedPose{}, 1280, 720) == 0.f);
}

TEST_CASE("Hand layouts: MediaPipe, OpenPose, InterHand2.6M and MANO", "[pose]")
{
  using namespace Onnx::Skel;
  struct Layout
  {
    TargetSkeleton t;
    size_t n;
  };
  for(auto [t, n] : {Layout{TargetSkeleton::Hand21, 21}, {TargetSkeleton::OpenPoseHand21, 21},
                     {TargetSkeleton::InterHand21, 21}, {TargetSkeleton::Mano16, 16},
                     {TargetSkeleton::Mano21, 21}})
  {
    // Every hand-capable source maps to every hand layout.
    for(auto s : {SourceSkeleton::Hands21, SourceSkeleton::Wholebody133, SourceSkeleton::Mhr70})
      CHECK(mappingFor(s, t).size() == n);
    CHECK(namesFor(t).size() == n);
    for(const auto& b : edgesFor(t))
    {
      CHECK(b.a >= 0);
      CHECK(b.b < static_cast<int>(n));
    }
  }

  const auto mano = mappingFor(SourceSkeleton::Hands21, TargetSkeleton::Mano21);
  CHECK(mano[0].idx[0] == 0);   // wrist
  CHECK(mano[1].idx[0] == 5);   // index MCP
  CHECK(mano[7].idx[0] == 17);  // pinky before ring, as in smplx
  CHECK(mano[13].idx[0] == 1);  // thumb CMC
  CHECK(mano[16].idx[0] == 4);  // fingertips appended, thumb first
  CHECK(mano[20].idx[0] == 20);
  CHECK(std::string_view{namesFor(TargetSkeleton::Mano16)[15]} == "THUMB3");

  // MediaPipe draws the palm's outline, OpenPose a bone to each finger.
  auto has = [](std::span<const Bone> bones, int a, int b) {
    return std::any_of(bones.begin(), bones.end(), [&](const Bone& x) {
      return (x.a == a && x.b == b) || (x.a == b && x.b == a);
    });
  };
  CHECK(has(edgesFor(TargetSkeleton::Hand21), 5, 9));
  CHECK_FALSE(has(edgesFor(TargetSkeleton::Hand21), 0, 9));
  CHECK(has(edgesFor(TargetSkeleton::OpenPoseHand21), 0, 9));
}

TEST_CASE("DARK finds a heatmap peak between the cells", "[pose]")
{
  // A 48x64 Gaussian (sigma 2, as mmpose renders them) centred off-grid.
  constexpr int W = 48, H = 64;
  const float cx = 20.3f, cy = 31.7f;
  std::vector<float> hm(W * H);
  for(int y = 0; y < H; y++)
    for(int x = 0; x < W; x++)
      hm[y * W + x] = std::exp(-((x - cx) * (x - cx) + (y - cy) * (y - cy)) / 8.f);
  const auto it = std::max_element(hm.begin(), hm.end());
  const int peak = int(it - hm.begin());
  const auto o = Onnx::Heatmap::darkOffset(hm.data(), W, H, peak % W, peak / W, *it);
  CHECK(peak % W + o.x == Catch::Approx(cx).margin(0.05));
  CHECK(peak / W + o.y == Catch::Approx(cy).margin(0.05));

  // A peak on the border cells is not refined.
  const auto edge = Onnx::Heatmap::darkOffset(hm.data(), W, H, 0, 10, 1.f);
  CHECK(edge.x == 0.f);
  CHECK(edge.y == 0.f);

  // Maps far smaller than the blur kernel read nothing outside them.
  std::vector<float> tiny{0.1f, 0.5f, 0.2f, 0.1f, 0.4f, 0.9f, 0.5f, 0.2f, 0.1f, 0.3f, 0.2f, 0.1f};
  const auto t = Onnx::Heatmap::darkOffset(tiny.data(), 4, 3, 1, 1, 0.9f);
  CHECK(std::isfinite(t.x));
  CHECK(std::abs(t.x) <= 1.f);
  CHECK(std::abs(t.y) <= 1.f);
}

TEST_CASE("BlazePose landmarks move to the heatmap's centre of mass", "[pose]")
{
  // A [64, 64, 2] heatmap of logits; landmark 1 peaks at cell (40.4, 12.6).
  constexpr int S = 64, K = 2;
  std::vector<float> hm(S * S * K, -10.f);
  for(int r = 0; r < S; r++)
    for(int c = 0; c < S; c++)
    {
      const float d2 = (c - 40.4f) * (c - 40.4f) + (r - 12.6f) * (r - 12.6f);
      hm[(r * S + c) * K + 1] = 6.f - d2;
    }
  // The regression is a cell and a half off; the refinement pulls it back
  // (to the centre of mass of the 7x7 cells around the regressed point, so
  // not all the way: MediaPipe's window is centred on the regression).
  float u = 42.f / S, v = 14.f / S;
  REQUIRE(Onnx::Heatmap::refineFromHeatmap(hm.data(), S, S, K, 1, u, v));
  CHECK(std::abs(u * S - 40.4f) < 0.25f);
  CHECK(std::abs(v * S - 12.6f) < 0.25f);

  // Landmark 0's heatmap is unsure everywhere: left as regressed.
  float u0 = 0.5f, v0 = 0.5f;
  CHECK_FALSE(Onnx::Heatmap::refineFromHeatmap(hm.data(), S, S, K, 0, u0, v0));
  CHECK(u0 == 0.5f);
  // Outside the map: nothing to read.
  float uo = 1.5f, vo = -0.2f;
  CHECK_FALSE(Onnx::Heatmap::refineFromHeatmap(hm.data(), S, S, K, 1, uo, vo));
}

TEST_CASE("Detector Tiles cover the frame with a bounded number of tiles", "[pose]")
{
  std::vector<Onnx::Detection::Tile> tiles;
  auto covers = [&](int W, int H) {
    int x0 = W, y0 = H, x1 = 0, y1 = 0;
    for(const auto& t : tiles)
    {
      if(t.x < 0 || t.y < 0 || t.x + t.w > W || t.y + t.h > H)
        return false;
      x0 = std::min(x0, t.x);
      y0 = std::min(y0, t.y);
      x1 = std::max(x1, t.x + t.w);
      y1 = std::max(y1, t.y + t.h);
    }
    return x0 == 0 && y0 == 0 && x1 == W && y1 == H;
  };

  // 16:9, 3 rows: 6 x 3 tiles of 288 px, as the hand presets were tuned on.
  Onnx::Detection::tileGrid(1280, 720, 3, tiles);
  CHECK(tiles.size() == 18);
  CHECK(tiles[0].w == 288);
  CHECK(covers(1280, 720));

  // Ultra-wide and degenerate frames: larger tiles, not more of them.
  Onnx::Detection::tileGrid(3840, 1080, 4, tiles);
  CHECK(tiles.size() <= 32);
  CHECK(covers(3840, 1080));
  Onnx::Detection::tileGrid(1920, 16, 2, tiles);
  CHECK(tiles.size() <= 8);
  CHECK(covers(1920, 16));

  Onnx::Detection::tileGrid(1280, 720, 1, tiles);
  CHECK(tiles.empty());
}

TEST_CASE("Tile detections merge into the whole frame's, people in crowds kept", "[pose]")
{
  using Onnx::Detection::Detection;
  // Whole frame: an adult and a child in front of them (nested), and two
  // people overlapping at IoU ~0.35. None is merged with another.
  std::vector<Detection> frame{
      {0.30f, 0.50f, 0.20f, 0.60f, 0.9f, 0, {}},
      {0.30f, 0.60f, 0.08f, 0.25f, 0.8f, 0, {}},
      {0.70f, 0.50f, 0.20f, 0.60f, 0.9f, 0, {}},
      {0.80f, 0.50f, 0.20f, 0.60f, 0.85f, 0, {}}};
  std::vector<Detection> tiles{
      {0.71f, 0.50f, 0.20f, 0.60f, 0.95f, 0, {}}, // the same person, seen better
      {0.30f, 0.35f, 0.18f, 0.28f, 0.7f, 0, {}},  // the adult, cut by a tile edge
      {0.70f, 0.50f, 0.05f, 0.05f, 0.6f, 1, {}},  // another class inside a person
      {0.10f, 0.10f, 0.03f, 0.08f, 0.5f, 0, {}}}; // someone far away
  const auto out = Onnx::Detection::mergeTileDetections(frame, tiles, 0.45f, 0.8f);
  REQUIRE(out.size() == 6);
  for(int i : {0, 1, 3})
    CHECK(out[i].xc == frame[i].xc);
  // The better-scored tile view replaced the whole frame's.
  CHECK(out[2].xc == Catch::Approx(0.71f));
  CHECK(out[4].class_id == 1);
  CHECK(out[5].xc == Catch::Approx(0.10f));
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
      // With Track IDs, a person shows once its track is confirmed (3 frames).
      const QImage frame = paste(body_jpg, 1600, 480, QRectF{1040, (480 - ph) / 2, pw, ph});
      for(int i = 0; i < 3; i++)
        h.run(frame);
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
  // A face shows once its track is confirmed (3 frames).
  for(int i = 0; i < 3; i++)
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
