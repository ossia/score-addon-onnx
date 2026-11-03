// Pose Detector hand pipelines: the hand ROI rules, the hand landmark decode,
// and every detector + landmark pairing of the pose-detector package on hand
// photos. The model cases SKIP without the package (ONNX_TEST_POSE_PACKAGE)
// or the hand photos (ONNX_TEST_HAND_IMAGES: MediaPipe's test images
// thumb_up.jpg, victory.jpg, pointing_up.jpg, fist.jpg, from
// storage.googleapis.com/mediapipe-assets).
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/PoseDetector_internal.hpp>

#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace
{
const std::string package = TestPaths::posePackage();
const std::string hands = TestPaths::handImages();

std::string det(const char* f)
{
  return package + "/detectors/" + f;
}
std::string lm(const char* f)
{
  return package + "/landmarks/" + f;
}

// One hand-tracking configuration, as a preset would set it.
struct Config
{
  std::string det, lm;
  int tiles = 1;
  bool track_ids = true;
  float min_conf = 0.5f;
};

struct Harness
{
  OnnxModels::PoseDetector node;
  QImage image;
  std::string det_name, det_bytes, lm_name, lm_bytes;

  explicit Harness(const Config& c)
      : det_name{c.det}
      , det_bytes{TestPaths::slurp(c.det)}
      , lm_name{c.lm}
      , lm_bytes{TestPaths::slurp(c.lm)}
  {
    node.inputs.det_model.file.bytes = det_bytes;
    node.inputs.det_model.file.filename = det_name;
    node.inputs.model.file.bytes = lm_bytes;
    node.inputs.model.file.filename = lm_name;
    node.inputs.detector_tiles.value = c.tiles;
    node.inputs.track_ids.value = c.track_ids;
    node.inputs.max_instances.value = 8;
    node.inputs.min_confidence.value = c.min_conf;
    // Raw per-frame output: no temporal filtering, nothing held over.
    node.inputs.smoothing.value = false;
    node.inputs.hold_frames.value = 0;
    node.inputs.track_memory.value = 1;
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

  // The instances of this frame: the multi-instance outlet, or the single one.
  std::vector<OnnxModels::DetectedPose> poses() const
  {
    if(node.inputs.track_ids.value)
      return node.outputs.poses.value;
    if(node.outputs.detection.value)
      return {*node.outputs.detection.value};
    return {};
  }
};

// A frame with hand photos pasted at known places on a noise background; the
// hand of each photo is roughly its middle 2/3.
struct Scene
{
  QImage frame;
  std::vector<QRectF> tiles; // normalized
};

Scene makeScene(const std::vector<std::pair<std::string, QRectF>>& photos)
{
  Scene s;
  s.frame = QImage(1280, 720, QImage::Format_RGB888);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> u(40, 200);
  {
    QImage small(16, 9, QImage::Format_RGB888);
    for(int y = 0; y < 9; y++)
      for(int x = 0; x < 16; x++)
        small.setPixel(x, y, qRgb(u(rng), u(rng), u(rng)));
    s.frame = small.scaled(1280, 720, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                  .convertToFormat(QImage::Format_RGB888);
  }
  QPainter p(&s.frame);
  p.setRenderHint(QPainter::SmoothPixmapTransform);
  for(const auto& [file, r] : photos)
  {
    const QRectF px{r.x() * 1280, r.y() * 720, r.width() * 1280, r.height() * 720};
    p.drawImage(px, QImage(QString::fromStdString(hands + "/" + file)));
    s.tiles.push_back(r);
  }
  return s;
}

// How many tiles have a pose whose 21 keypoints lie within them.
int handsFound(const Scene& s, const std::vector<OnnxModels::DetectedPose>& poses)
{
  int found = 0;
  for(const auto& t : s.tiles)
  {
    const QRectF loose = t.adjusted(
        -0.05 * t.width(), -0.05 * t.height(), 0.05 * t.width(), 0.05 * t.height());
    for(const auto& p : poses)
    {
      if(p.keypoints.size() != 21)
        continue;
      const bool inside = std::all_of(
          p.keypoints.begin(), p.keypoints.end(),
          [&](const auto& k) { return loose.contains(k.x, k.y); });
      if(inside)
      {
        ++found;
        break;
      }
    }
  }
  return found;
}

// Hand photos are about square; a tile `h` of the frame height high.
QRectF tile(double x, double y, double h)
{
  return {x, y, h * 720. / 1280., h};
}
}

// --- No model -----------------------------------------------------------------

TEST_CASE("Hand ROI from landmarks follows the hand's rotation", "[pose][hand]")
{
  // An upright right hand, wrist at the bottom, fingers up (image px).
  std::vector<Onnx::Vec2> up(21);
  up[0] = {100, 200};
  const float mcp_x[5] = {70, 85, 100, 115, 128};
  for(int f = 0; f < 5; f++)
    for(int j = 0; j < 4; j++)
      up[1 + 4 * f + j] = {mcp_x[f], 160.f - 25.f * j};
  const auto r = Onnx::ROI::handRectFromLandmarks([&](int i) { return up[i]; });
  // Upright: no rotation; the 0.1 shift moves the centre toward the fingers.
  CHECK(r.angle == Catch::Approx(0.f).margin(1e-4));
  CHECK(r.w == Catch::Approx(r.h));
  CHECK(r.cy < 200.f);
  CHECK(r.cx == Catch::Approx(100.f).margin(10.f));

  // The same hand turned by a quarter turn about the wrist: the rect turns
  // with it, same size, centre rotated likewise.
  const float a = float(M_PI) / 2.f;
  auto rot = [&](Onnx::Vec2 p) {
    const float x = p.x - 100.f, y = p.y - 200.f;
    return Onnx::Vec2{
        100.f + x * std::cos(a) - y * std::sin(a),
        200.f + x * std::sin(a) + y * std::cos(a)};
  };
  const auto q = Onnx::ROI::handRectFromLandmarks([&](int i) { return rot(up[i]); });
  CHECK(q.w == Catch::Approx(r.w).epsilon(1e-3));
  const auto c = rot({r.cx, r.cy});
  CHECK(q.cx == Catch::Approx(c.x).margin(0.5f));
  CHECK(q.cy == Catch::Approx(c.y).margin(0.5f));
  CHECK(std::remainder(q.angle - (r.angle + a), 2.f * float(M_PI))
        == Catch::Approx(0.f).margin(1e-3));
}

TEST_CASE("Hand landmarks: the screen landmarks are the first 63 output", "[pose][hand]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<float> screen(63), world(63);
  for(int i = 0; i < 63; i++)
  {
    screen[i] = 100.f + i;      // crop pixels
    world[i] = 0.001f * i;      // metres
  }
  float presence = 0.9f, handedness = 0.95f;
  std::array<int64_t, 2> s63{1, 63}, s1{1, 1};
  Onnx::ModelSpec spec;
  spec.inputs.push_back({});
  spec.inputs[0].shape = {1, 224, 224, 3};

  // hand_landmark_full order: screen, presence, handedness, world.
  Ort::Value outs[4]{
      Ort::Value::CreateTensor<float>(mem, screen.data(), 63, s63.data(), 2),
      Ort::Value::CreateTensor<float>(mem, &presence, 1, s1.data(), 2),
      Ort::Value::CreateTensor<float>(mem, &handedness, 1, s1.data(), 2),
      Ort::Value::CreateTensor<float>(mem, world.data(), 63, s63.data(), 2)};
  std::optional<Onnx::MediaPipeHands::HandResult> r;
  REQUIRE(Onnx::MediaPipeHands::processOutput(spec, outs, r));
  REQUIRE(r->landmarks.size() == 21);
  CHECK(r->landmarks[1].x == Catch::Approx(103.f / 224.f));
  CHECK(r->hand_flag == Catch::Approx(0.9f));
  CHECK(r->is_right_hand);
  REQUIRE(r->world.size() == 21);
  CHECK(r->world[1].x == Catch::Approx(0.003f));

  // The legacy model: presence, handedness, landmarks; no world.
  spec.inputs[0].shape = {1, 256, 256, 3};
  Ort::Value legacy[3]{
      Ort::Value::CreateTensor<float>(mem, &presence, 1, s1.data(), 2),
      Ort::Value::CreateTensor<float>(mem, &handedness, 1, s1.data(), 2),
      Ort::Value::CreateTensor<float>(mem, screen.data(), 63, s63.data(), 2)};
  REQUIRE(Onnx::MediaPipeHands::processOutput(spec, legacy, r));
  CHECK(r->landmarks[1].x == Catch::Approx(103.f / 256.f));
  CHECK(r->world.empty());

  // An absent hand is rejected on presence.
  presence = 0.1f;
  CHECK_FALSE(Onnx::MediaPipeHands::processOutput(spec, legacy, r));
}

// --- Models -------------------------------------------------------------------

namespace
{
// The hand model pairings of the package (the hand presets).
const Config palm128_legacy{det("det-hand-palm-128.onnx"), lm("lm-hand-mediapipe-63.onnx")};
const Config palm192_full{det("det-hand-palm-192.onnx"), lm("lm-hand-mediapipe-full-63.onnx")};
const Config palm128_rtmpose{det("det-hand-palm-128.onnx"), lm("lm-hand-rtmpose-21.onnx")};
const Config rtmdet_rtmpose{det("det-hand-rtmdet-320.onnx"), lm("lm-hand-rtmpose-21.onnx")};
const Config rtmdet_rtmpose_post{
    det("det-hand-rtmdet-320.onnx"), lm("lm-hand-rtmpose-post-21.onnx")};
const Config rtmdet_full{det("det-hand-rtmdet-320.onnx"), lm("lm-hand-mediapipe-full-63.onnx")};

bool haveHands()
{
  return TestPaths::haveAll(
      {hands + "/thumb_up.jpg", hands + "/victory.jpg", hands + "/pointing_up.jpg",
       hands + "/fist.jpg"});
}
}

TEST_CASE("Every hand pairing finds a hand filling a third of the frame", "[onnx][pose][hand][model]")
{
  // The single-hand presets: no tiles, no tracking.
  for(Config c :
      {palm128_legacy, palm192_full, palm128_rtmpose, rtmdet_rtmpose,
       rtmdet_rtmpose_post, rtmdet_full})
  {
    DYNAMIC_SECTION(c.det << " + " << c.lm)
    {
      if(!haveHands() || !TestPaths::haveAll({c.det, c.lm}))
        SKIP("models or images not found");
      c.track_ids = false;
      c.min_conf = 0.3f;
      Harness h{c};
      const auto s = makeScene({{"thumb_up.jpg", tile(0.4, 0.3, 0.45)}});
      h.run(s.frame);
      CHECK(handsFound(s, h.poses()) == 1);
    }
  }
}

TEST_CASE("The legacy 128 palm detector gets [-1,1] input", "[onnx][pose][hand][model]")
{
  if(!haveHands() || !TestPaths::haveAll({palm128_legacy.det}))
    SKIP("models or images not found");
  // A hand a fifth of the frame high: in [0,1] the legacy palm misses most of
  // these; in [-1,1] it finds them all.
  Config c = palm128_legacy;
  Harness h{c};
  h.node.inputs.workflow.value = OnnxModels::PoseWorkflow::BoxDetection;
  int found = 0;
  for(const char* f : {"thumb_up.jpg", "victory.jpg", "pointing_up.jpg", "fist.jpg"})
  {
    const auto s = makeScene({{f, tile(0.55, 0.45, 0.3)}});
    h.run(s.frame);
    for(const auto& p : h.node.outputs.poses.value)
    {
      const float cx = p.box.x + p.box.w / 2, cy = p.box.y + p.box.h / 2;
      if(s.tiles[0].contains(cx, cy))
      {
        ++found;
        break;
      }
    }
  }
  CHECK(found >= 3);
}

TEST_CASE("Detector Tiles find small hands anywhere in the frame", "[onnx][pose][hand][model]")
{
  struct Case
  {
    Config c;
    int min_small; // of 4 hands an eighth of the frame high
  };
  // From the Python benchmark of these pairings (recall at 9-12% of the
  // frame height): palm192 ~1.0, palm128 [-1,1] ~0.9, RTMDet ~0.5.
  for(auto [c, min_small] :
      {Case{palm192_full, 4}, Case{palm128_legacy, 3}, Case{palm128_rtmpose, 3},
       Case{rtmdet_full, 2}, Case{rtmdet_rtmpose, 2}})
  {
    DYNAMIC_SECTION(c.det << " + " << c.lm)
    {
      if(!haveHands() || !TestPaths::haveAll({c.det, c.lm}))
        SKIP("models or images not found");
      const auto s = makeScene(
          {{"thumb_up.jpg", tile(0.03, 0.05, 0.16)},
           {"victory.jpg", tile(0.85, 0.08, 0.16)},
           {"pointing_up.jpg", tile(0.10, 0.78, 0.16)},
           {"fist.jpg", tile(0.80, 0.80, 0.16)}});

      // Whole frame only: these are too small for every detector.
      c.tiles = 1;
      // RTMPose-hand: real small hands score ~0.55-0.7, empty crops < 0.25.
      c.min_conf = c.lm == lm("lm-hand-rtmpose-21.onnx") ? 0.4f : 0.5f;
      {
        Harness h{c};
        h.run(s.frame);
        h.run(s.frame); // a track is emitted once confirmed
        CHECK(handsFound(s, h.poses()) <= 1);
      }
      c.tiles = 3;
      Harness h{c};
      h.run(s.frame);
      h.run(s.frame);
      const auto poses = h.poses();
      CHECK(handsFound(s, poses) >= min_small);
      // No hand made up in the background.
      CHECK(static_cast<int>(poses.size()) <= handsFound(s, poses) + 1);
    }
  }
}

TEST_CASE("Multi-hand tracking: the rolling tile scan finds and keeps every hand", "[onnx][pose][hand][model]")
{
  for(Config c : {palm192_full, palm128_legacy})
  {
    DYNAMIC_SECTION(c.det)
    {
      if(!haveHands() || !TestPaths::haveAll({c.det, c.lm}))
        SKIP("models or images not found");
      c.tiles = 3;
      Harness h{c};
      h.node.inputs.track_roi.value = true;
      h.node.inputs.detector_cadence.value = 1;
      h.node.inputs.smoothing.value = true;
      h.node.inputs.hold_frames.value = 6;
      h.node.inputs.track_memory.value = 30;

      // Four small hands drifting across the frame.
      auto scene = [&](int f) {
        const double d = 0.004 * f;
        return makeScene(
            {{"thumb_up.jpg", tile(0.03 + d, 0.05, 0.16)},
             {"victory.jpg", tile(0.85 - d, 0.08 + d, 0.16)},
             {"pointing_up.jpg", tile(0.10 + d, 0.78 - d, 0.16)},
             {"fist.jpg", tile(0.80 - d, 0.80, 0.16)}});
      };
      // 3 rows of tiles on 16:9 are 18 tiles, 2 per detection pass: the scan
      // covers the frame in 9 frames.
      std::vector<std::vector<int>> ids;
      double ms = 0;
      for(int f = 0; f < 30; f++)
      {
        const auto s = scene(f);
        const auto t0 = std::chrono::steady_clock::now();
        h.run(s.frame);
        ms += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
        if(f < 12)
          continue;
        const auto poses = h.poses();
        CHECK(handsFound(s, poses) == 4);
        // No hand twice. A detection not yet confirmed as a track (id -1) is
        // emitted too: the legacy pair has the odd one-frame background blip.
        std::vector<int> frame_ids;
        for(const auto& p : poses)
          if(p.track_id >= 0)
            frame_ids.push_back(p.track_id);
        CHECK(frame_ids.size() == 4);
        CHECK(poses.size() <= 5);
        std::sort(frame_ids.begin(), frame_ids.end());
        ids.push_back(frame_ids);
      }
      // The same four ids throughout.
      for(const auto& f : ids)
        CHECK(f == ids.front());

      // The cost: all tiles every frame instead.
      Config all = c;
      all.tiles = 3;
      Harness a{all};
      double ms_all = 0;
      for(int f = 0; f < 10; f++)
      {
        const auto s = scene(f);
        const auto t0 = std::chrono::steady_clock::now();
        a.run(s.frame);
        ms_all += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      }
      std::printf(
          "%s: rolling scan %.1f ms/frame, every tile every frame %.1f ms/frame\n",
          c.det.c_str(), ms / 30, ms_all / 10);
    }
  }
}

TEST_CASE("Single hand with Track ROI follows a moving and turning hand", "[onnx][pose][hand][model]")
{
  // The Hand MediaPipe Full preset: one hand, the landmark-derived ROI.
  for(Config c : {palm192_full, palm128_legacy})
  {
    DYNAMIC_SECTION(c.det)
    {
      if(!haveHands() || !TestPaths::haveAll({c.det, c.lm}))
        SKIP("models or images not found");
      c.track_ids = false;
      c.tiles = 2;
      Harness h{c};
      h.node.inputs.track_roi.value = true;
      h.node.inputs.smoothing.value = true;
      const QImage photo(QString::fromStdString(hands + "/victory.jpg"));
      int found = 0;
      for(int f = 0; f < 40; f++)
      {
        // Across the frame while turning a quarter turn.
        Scene s = makeScene({});
        const QRectF r = tile(0.1 + 0.015 * f, 0.25, 0.3);
        {
          QPainter p(&s.frame);
          p.setRenderHint(QPainter::SmoothPixmapTransform);
          const QPointF c{(r.x() + r.width() / 2) * 1280, (r.y() + r.height() / 2) * 720};
          p.translate(c);
          p.rotate(2.25 * f);
          p.drawImage(
              QRectF{-r.width() * 640, -r.height() * 360, r.width() * 1280, r.height() * 720},
              photo);
        }
        // The turned photo's bounds.
        const double grow = 1.42;
        s.tiles.push_back(QRectF{
            r.x() + r.width() / 2 - r.width() * grow / 2,
            r.y() + r.height() / 2 - r.height() * grow / 2, r.width() * grow,
            r.height() * grow});
        h.run(s.frame);
        found += handsFound(s, h.poses());
      }
      // A tracked crop that loses the hand re-detects on the same frame.
      CHECK(found == 40);
    }
  }
}

TEST_CASE("hand_landmark_full: world landmarks in metres", "[onnx][pose][hand][model]")
{
  if(!haveHands() || !TestPaths::haveAll({palm192_full.det, palm192_full.lm}))
    SKIP("models or images not found");
  Config c = palm192_full;
  c.track_ids = false;
  Harness h{c};
  const auto s = makeScene({{"victory.jpg", tile(0.4, 0.2, 0.5)}});
  h.run(s.frame);
  REQUIRE(h.node.outputs.detection.value.has_value());
  const auto& w = h.node.outputs.detection.value->world;
  REQUIRE(w.size() == 21);
  // A hand spans about 15-25 cm.
  float lo = 1e9f, hi = -1e9f;
  for(const auto& k : w)
  {
    lo = std::min(lo, k.y);
    hi = std::max(hi, k.y);
  }
  CHECK(hi - lo > 0.08f);
  CHECK(hi - lo < 0.4f);
}

// The Python benchmark's composite set, replayed through the node:
// ONNX_TEST_HAND_SET is a folder with gt.json (frame path -> hands, each the
// 21 reference keypoints in pixels and the hand size). Prints, per pairing,
// recall, false positives, mean error / hand size, per frame set.
TEST_CASE("Hand pipelines benchmark", "[.bench][hand]")
{
  const auto dir = TestPaths::env("ONNX_TEST_HAND_SET");
  QFile f(QString::fromStdString(dir + "/gt.json"));
  if(dir.empty() || !f.open(QIODevice::ReadOnly))
    SKIP("ONNX_TEST_HAND_SET not set");
  const auto gt = QJsonDocument::fromJson(f.readAll()).object();

  struct Row
  {
    const char* name;
    Config c;
  };
  auto with = [](Config c, int tiles, float conf) {
    c.tiles = tiles;
    c.min_conf = conf;
    return c;
  };
  for(const auto& [name, c] :
      {Row{"palm128+legacy256", with(palm128_legacy, 1, 0.5f)},
       Row{"palm128+legacy256 tiles2", with(palm128_legacy, 2, 0.5f)},
       Row{"palm192+full224", with(palm192_full, 1, 0.5f)},
       Row{"palm192+full224 tiles3", with(palm192_full, 3, 0.5f)},
       Row{"palm128+rtmpose tiles2", with(palm128_rtmpose, 2, 0.4f)},
       Row{"rtmdet+rtmpose tiles2", with(rtmdet_rtmpose, 2, 0.4f)},
       Row{"rtmdet+full224 tiles2", with(rtmdet_full, 2, 0.5f)}})
  {
    if(!TestPaths::haveAll({c.det, c.lm}))
      continue;
    Harness h{c};
    struct Acc
    {
      int n = 0, tp = 0, fp = 0;
      double err = 0;
    };
    std::map<std::string, Acc> acc;
    std::vector<double> ms;
    for(auto it = gt.begin(); it != gt.end(); ++it)
    {
      const QString path = it.key();
      QString set = QFileInfo(path).baseName();
      set.truncate(set.lastIndexOf('_'));
      auto& a = acc[set.toStdString()];
      const QImage img(path);
      h.run(img);
      const auto t0 = std::chrono::steady_clock::now();
      h.run(img);
      ms.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
              .count());
      const auto poses = h.poses();
      // Greedy matching by error (mean distance / hand size), cut at 0.35.
      struct Pair
      {
        double e;
        int g, p;
      };
      std::vector<Pair> pairs;
      const auto handsArr = it.value().toArray();
      for(int g = 0; g < handsArr.size(); g++)
      {
        const auto hand = handsArr[g].toObject();
        const auto kps = hand["kps"].toArray();
        const double size = hand["size"].toDouble();
        for(int p = 0; p < (int)poses.size(); p++)
        {
          if(poses[p].keypoints.size() != 21)
            continue;
          double e = 0;
          for(int k = 0; k < 21; k++)
          {
            const auto q = kps[k].toArray();
            e += std::hypot(
                poses[p].keypoints[k].x * img.width() - q[0].toDouble(),
                poses[p].keypoints[k].y * img.height() - q[1].toDouble());
          }
          pairs.push_back({e / 21 / size, g, p});
        }
      }
      std::sort(pairs.begin(), pairs.end(), [](auto& x, auto& y) { return x.e < y.e; });
      std::vector<bool> ug(handsArr.size()), up(poses.size());
      int tp = 0;
      for(const auto& pr : pairs)
        if(pr.e < 0.35 && !ug[pr.g] && !up[pr.p])
        {
          ug[pr.g] = up[pr.p] = true;
          ++tp;
          a.err += pr.e;
        }
      a.n += handsArr.size();
      a.tp += tp;
      a.fp += (int)poses.size() - tp;
    }
    std::sort(ms.begin(), ms.end());
    std::printf("%-26s", name);
    for(const auto& [set, a] : acc)
      std::printf(
          " | %s %d/%d fp%d e%.3f", set.c_str(), a.tp, a.n, a.fp,
          a.tp ? a.err / a.tp : 0.);
    std::printf(" | %.1f ms\n", ms.empty() ? 0. : ms[ms.size() / 2]);
  }
}
