// Replays a Pose Detector preset over a set of still images and writes what
// the node outputs, for scoring against a dataset's annotations (COCO's
// keypoint AP, see the scripts this is run from). Hidden: run it by name with
//   ONNX_TEST_POSE_BENCH_PRESET  the .scp preset
//   ONNX_TEST_POSE_BENCH_ROOT    the library root its <LIBRARY>: paths are under
//   ONNX_TEST_POSE_BENCH_LIST    lines of "<image id> <image path>"
//   ONNX_TEST_POSE_BENCH_OUT     the JSON to write
//   ONNX_TEST_POSE_BENCH_MULTI   "1" turns Track IDs on: every instance out
//   ONNX_TEST_POSE_BENCH_SKELETON  a Skeleton index to output instead
//   ONNX_TEST_POSE_BENCH_SET     overrides, "<port>=<number>;...", e.g. "4=0.1;12=16"
//   ONNX_TEST_POSE_BENCH_AS_IS   "1" keeps the preset's smoothing, hold and track memory
//   ONNX_TEST_POSE_BENCH_REPEAT  frames per image (default 1): the output of
//                                the last, so tracked ROIs come into play;
//                                every frame is timed ("frame_ms")
// Each image is an independent frame: the node forgets it before the next.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/PoseDetector_internal.hpp>

#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace
{
std::string libraryPath(QString v, const QString& root)
{
  if(v.startsWith("<LIBRARY>:"))
    v = root + '/' + v.mid(10);
  return v.toStdString();
}

struct File
{
  std::string name, bytes;
};

// Sets the controls a preset names, by port index (PoseDetector.hpp order).
void applyPreset(
    OnnxModels::PoseDetector& node, const QJsonArray& preset, const QString& root,
    File& lm, File& det)
{
  auto& in = node.inputs;
  for(const auto& e : preset)
  {
    const auto entry = e.toArray();
    const int idx = entry[0].toInt();
    const auto obj = entry[1].toObject();
    const auto v = obj.begin().value();
    const bool b = v.isBool() ? v.toBool() : v.toDouble() != 0.;
    const int i = v.toInt();
    const float f = v.toDouble();
    switch(idx)
    {
      case 1:
        lm = {libraryPath(v.toString(), root), {}};
        break;
      case 2:
        in.workflow.value = static_cast<decltype(in.workflow.value)>(i);
        break;
      case 4:
        in.min_confidence.value = f;
        break;
      case 7:
        det = {libraryPath(v.toString(), root), {}};
        break;
      case 8:
        in.track_roi.value = b;
        break;
      case 11:
        in.track_ids.value = b;
        break;
      case 12:
        in.max_instances.value = i;
        break;
      case 13:
        in.detector_cadence.value = i;
        break;
      case 19:
        in.detection_class.value = i;
        break;
      case 21:
        in.motion_gate.value = static_cast<decltype(in.motion_gate.value)>(i);
        break;
      case 22:
        in.max_speed.value = f;
        break;
      case 23:
        in.birth_gate.value = b;
        break;
      case 24:
        in.strict_confirm.value = b;
        break;
      case 25:
        in.skeleton_type.value = static_cast<decltype(in.skeleton_type.value)>(i);
        break;
      case 9:
        in.smoothing.value = b;
        break;
      case 10:
        in.smoothing_amount.value = f;
        break;
      case 27:
        in.track_memory.value = i;
        break;
      case 30:
        in.hold_frames.value = i;
        break;
      case 35:
        // A generic lambda, so the bench still builds against older nodes.
        [i](auto& ports) {
          if constexpr(requires { ports.detector_tiles; })
            ports.detector_tiles.value = i;
        }(in);
        break;
      default:
        break;
    }
  }
}

void feed(OnnxModels::PoseDetector& node, const QImage& img)
{
  auto& t = node.inputs.image.texture;
  t.bytes = const_cast<unsigned char*>(img.constBits());
  t.width = img.width();
  t.height = img.height();
  t.changed = true;
  node();
}

QJsonArray points(const std::vector<OnnxModels::PoseKeypoint>& kps, int w, int h)
{
  QJsonArray a;
  for(const auto& k : kps)
  {
    a.append(k.x * w);
    a.append(k.y * h);
    a.append(k.confidence);
  }
  return a;
}
}

TEST_CASE("Pose Detector preset replay over an image set", "[.bench][pose]")
{
  const auto preset_path = TestPaths::env("ONNX_TEST_POSE_BENCH_PRESET");
  const auto root = QString::fromStdString(TestPaths::env("ONNX_TEST_POSE_BENCH_ROOT"));
  const auto list = TestPaths::env("ONNX_TEST_POSE_BENCH_LIST");
  const auto out_path = TestPaths::env("ONNX_TEST_POSE_BENCH_OUT");
  if(preset_path.empty() || list.empty() || out_path.empty())
    SKIP("ONNX_TEST_POSE_BENCH_{PRESET,LIST,OUT} are not set");

  QFile f{QString::fromStdString(preset_path)};
  REQUIRE(f.open(QIODevice::ReadOnly));
  auto preset = QJsonDocument::fromJson(f.readAll()).object()["Preset"].toArray();
  for(const auto& kv : QString::fromStdString(TestPaths::env("ONNX_TEST_POSE_BENCH_SET"))
                           .split(';', Qt::SkipEmptyParts))
  {
    const auto p = kv.split('=');
    const double v = p.value(1).toDouble();
    preset.append(QJsonArray{p[0].toInt(), QJsonObject{{"Float", v}}});
  }

  OnnxModels::PoseDetector node;
  File lm, det;
  applyPreset(node, preset, root, lm, det);
  if(TestPaths::env("ONNX_TEST_POSE_BENCH_MULTI") == "1")
    node.inputs.track_ids.value = true;
  if(auto sk = TestPaths::env("ONNX_TEST_POSE_BENCH_SKELETON"); !sk.empty())
    node.inputs.skeleton_type.value
        = static_cast<decltype(node.inputs.skeleton_type.value)>(std::stoi(sk));
  if(!lm.name.empty()) // box presets have no landmark model
  {
    lm.bytes = TestPaths::slurp(lm.name);
    REQUIRE(!lm.bytes.empty());
    node.inputs.model.file.bytes = lm.bytes;
    node.inputs.model.file.filename = lm.name;
  }
  if(!det.name.empty())
  {
    det.bytes = TestPaths::slurp(det.name);
    node.inputs.det_model.file.bytes = det.bytes;
    node.inputs.det_model.file.filename = det.name;
  }
  // Stills: no temporal filtering, nothing held over to the next image
  // (unless ONNX_TEST_POSE_BENCH_AS_IS=1: the preset's own settings).
  if(TestPaths::env("ONNX_TEST_POSE_BENCH_AS_IS") != "1")
  {
    node.inputs.smoothing.value = false;
    node.inputs.hold_frames.value = 0;
    node.inputs.track_memory.value = 1;
  }
  inlineWorker(node);

  const QImage blank = [] {
    QImage i(64, 64, QImage::Format_RGBA8888);
    i.fill(Qt::black);
    return i;
  }();

  const int repeat = std::max(1, std::atoi(TestPaths::env("ONNX_TEST_POSE_BENCH_REPEAT").c_str()));
  QJsonArray results;
  std::ifstream in{list};
  std::string line;
  double ms = 0;
  int frames = 0;
  QJsonArray frame_ms;
  while(std::getline(in, line))
  {
    std::istringstream ls{line};
    long long id;
    std::string path;
    if(!(ls >> id >> path))
      continue;
    const QImage img
        = QImage(QString::fromStdString(path)).convertToFormat(QImage::Format_RGBA8888);
    if(img.isNull())
      continue;
    for(int k = 0; k < 2; k++)
      feed(node, blank);
    for(int k = 0; k < repeat; k++)
    {
      const auto t0 = std::chrono::steady_clock::now();
      feed(node, img);
      const double t
          = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
      frame_ms.append(t);
      if(k == repeat - 1)
        ms += t;
    }
    frames++;

    std::vector<OnnxModels::DetectedPose> poses = node.outputs.poses.value;
    // Single-instance workflows only fill the detection outlet.
    if(!node.inputs.track_ids.value || poses.empty())
    {
      poses.clear();
      if(node.outputs.detection.value)
        poses.push_back(*node.outputs.detection.value);
    }
    const int w = img.width(), h = img.height();
    for(const auto& p : poses)
    {
      QJsonObject r;
      r["image_id"] = id;
      r["keypoints"] = points(p.keypoints, w, h);
      r["score"] = p.mean_confidence;
      r["bbox"] = QJsonArray{p.box.x * w, p.box.y * h, p.box.w * w, p.box.h * h};
      r["track_id"] = p.track_id;
      results.append(r);
    }
  }
  QJsonObject doc;
  doc["results"] = results;
  doc["ms_per_frame"] = frames ? ms / frames : 0.;
  doc["frame_ms"] = frame_ms;
  QFile out{QString::fromStdString(out_path)};
  REQUIRE(out.open(QIODevice::WriteOnly));
  out.write(QJsonDocument(doc).toJson(QJsonDocument::Compact));
  std::printf("%d frames, %.1f ms/frame\n", frames, frames ? ms / frames : 0.);
}
