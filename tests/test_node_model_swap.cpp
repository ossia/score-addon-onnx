// Model swap and error recovery of the Resnet / EmotionNet nodes: runs the
// real node sources on real models.
//
// Models are read from $ONNX_TEST_MODELS; each test SKIPs when its models are
// missing.
#include <tests/TestPaths.hpp>
#include <OnnxModels/ENet.hpp>
#include <OnnxModels/Resnet.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
// What avendish's raw_file_storage::load does: repoint the bytes and the file
// name, and call update() only when the name changed.
template <typename Node, typename Port>
void load(Node& node, Port& port, const std::string& bytes, const std::string& name)
{
  const bool changed = port.file.filename != name;
  port.file.bytes = bytes;
  port.file.filename = name;
  if(changed)
    port.update(node);
}

template <typename Node>
void tick(Node& node, std::vector<unsigned char>& img)
{
  auto& tex = node.inputs.image.texture;
  tex.bytes = img.data();
  tex.width = 224;
  tex.height = 224;
  tex.changed = true;
  node();
}

template <typename Node>
std::vector<std::pair<std::string, float>> detections(const Node& node)
{
  std::vector<std::pair<std::string, float>> res;
  for(const auto& d : node.outputs.detection.value)
    res.emplace_back(d.name, d.probability);
  return res;
}

std::vector<unsigned char> testImage()
{
  std::vector<unsigned char> img(224 * 224 * 4);
  for(std::size_t i = 0; i < img.size(); i++)
    img[i] = (unsigned char)((i * 37) % 251);
  return img;
}

void setupResnet(OnnxModels::ResnetDetector& r, const std::string& classes)
{
  r.inputs.classes.file.filename = classes;
  r.inputs.classes.update(r);
  r.inputs.resolution.value = {224, 224};
}
}

TEST_CASE("Resnet: swapping the model file runs the new model", "[onnx][resnet][model]")
{
  const auto dir = TestPaths::models();
  const auto r18 = dir + "/resnet/resnet18-imagenet.onnx";
  const auto alex = dir + "/resnet/alexnet.onnx";
  const auto classes = dir + "/resnet/imagenet_classes.txt";
  if(!TestPaths::haveAll({r18, alex, classes}))
    SKIP("models not found in " << dir);

  auto img = testImage();
  const auto r18_bytes = TestPaths::slurp(r18);
  const auto alex_bytes = TestPaths::slurp(alex);

  OnnxModels::ResnetDetector swapped;
  setupResnet(swapped, classes);
  load(swapped, swapped.inputs.model, r18_bytes, r18);
  tick(swapped, img);
  const auto before = detections(swapped);
  REQUIRE(!before.empty());

  load(swapped, swapped.inputs.model, alex_bytes, alex);
  tick(swapped, img);

  OnnxModels::ResnetDetector fresh;
  setupResnet(fresh, classes);
  load(fresh, fresh.inputs.model, alex_bytes, alex);
  tick(fresh, img);

  REQUIRE(!detections(fresh).empty());
  CHECK(detections(swapped) == detections(fresh));
  CHECK(detections(swapped) != before);
}

TEST_CASE("Resnet: a frame the model rejects does not disable the node", "[onnx][resnet]")
{
  // Declares [1,3,H,W] but only runs at 8x8: another resolution fails on that
  // frame only.
  const std::string model = SCORE_ONNX_TEST_DATA_DIR "/classify/dyn8.onnx";
  const std::string classes = SCORE_ONNX_TEST_DATA_DIR "/classify/classes4.txt";

  auto img = testImage();
  const auto bytes = TestPaths::slurp(model);

  OnnxModels::ResnetDetector r;
  setupResnet(r, classes);
  r.inputs.resolution.value = {8, 8};
  load(r, r.inputs.model, bytes, model);
  tick(r, img);
  const auto good = detections(r);
  REQUIRE(!good.empty());

  r.inputs.resolution.value = {16, 16};
  tick(r, img);
  CHECK(!r.inputs.model.current_model_invalid);
  CHECK(detections(r).empty());

  // Back to a valid resolution: the node recovers without reloading the model.
  r.inputs.resolution.value = {8, 8};
  tick(r, img);
  CHECK(!r.inputs.model.current_model_invalid);
  CHECK(detections(r) == good);
}

// The tensor is shaped and filled at the model's size: a fixed-size model
// ignores the resolution knob.
TEST_CASE("Resnet: a fixed-size model ignores the resolution knob", "[onnx][resnet][model]")
{
  const auto dir = TestPaths::models();
  const auto r18 = dir + "/resnet/resnet18-imagenet.onnx";
  const auto classes = dir + "/resnet/imagenet_classes.txt";
  if(!TestPaths::haveAll({r18, classes}))
    SKIP("models not found in " << dir);
  auto img = testImage();
  const auto bytes = TestPaths::slurp(r18);

  std::vector<std::vector<std::pair<std::string, float>>> res;
  for(int k : {224, 256, 128})
  {
    OnnxModels::ResnetDetector r;
    setupResnet(r, classes);
    r.inputs.resolution.value = {k, k};
    load(r, r.inputs.model, bytes, r18);
    tick(r, img);
    CHECK(!r.inputs.model.current_model_invalid);
    res.push_back(detections(r));
  }
  REQUIRE(!res[0].empty());
  CHECK(res[1] == res[0]);
  CHECK(res[2] == res[0]);
}

TEST_CASE("Resnet: a file that is not a model marks it invalid", "[onnx][resnet]")
{
  OnnxModels::ResnetDetector r;
  r.inputs.resolution.value = {224, 224};
  const std::string garbage(4096, 'x');
  load(r, r.inputs.model, garbage, "/nonexistent/garbage.onnx");
  auto img = testImage();
  tick(r, img);
  CHECK(r.inputs.model.current_model_invalid);
  CHECK(detections(r).empty());
}

TEST_CASE("EmotionNet: swapping the model file runs the new model", "[onnx][emotionnet][model]")
{
  const auto dir = TestPaths::models();
  const auto afew = dir + "/emotionnet/enet_b0_8_best_afew.onnx";
  const auto vgaf = dir + "/emotionnet/enet_b0_8_best_vgaf.onnx";
  if(!TestPaths::haveAll({afew, vgaf}))
    SKIP("models not found in " << dir);

  auto img = testImage();
  const auto afew_bytes = TestPaths::slurp(afew);
  const auto vgaf_bytes = TestPaths::slurp(vgaf);

  OnnxModels::EmotionNetDetector swapped;
  swapped.inputs.resolution.value = {224, 224};
  load(swapped, swapped.inputs.model, afew_bytes, afew);
  tick(swapped, img);
  const auto before = detections(swapped);
  REQUIRE(!before.empty());

  load(swapped, swapped.inputs.model, vgaf_bytes, vgaf);
  tick(swapped, img);

  OnnxModels::EmotionNetDetector fresh;
  fresh.inputs.resolution.value = {224, 224};
  load(fresh, fresh.inputs.model, vgaf_bytes, vgaf);
  tick(fresh, img);

  REQUIRE(!detections(fresh).empty());
  CHECK(detections(swapped) == detections(fresh));
  CHECK(detections(swapped) != before);
}

// The top-5 ranking is per node and per model: a later model with another
// class count on the same thread ranks its own range, even with fewer than 5
// classes.
namespace
{
std::vector<OnnxModels::Resnet::recognition_type>
top(OnnxModels::Resnet& r, std::vector<float> logits)
{
  std::vector<int64_t> shape{1, (int64_t)logits.size()};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value v[1]{Ort::Value::CreateTensor<float>(
      mem, logits.data(), logits.size(), shape.data(), shape.size())};
  std::vector<OnnxModels::Resnet::recognition_type> out;
  r.processOutput({}, v, out);
  return out;
}

std::vector<std::string> names(const std::vector<OnnxModels::Resnet::recognition_type>& v)
{
  std::vector<std::string> res;
  for(auto& e : v)
    res.push_back(e.name);
  return res;
}

OnnxModels::Resnet withClasses(int n)
{
  OnnxModels::Resnet r;
  for(int i = 0; i < n; i++)
    r.classes.push_back("c" + std::to_string(i));
  return r;
}
}

TEST_CASE("Resnet top-5 with models of different class counts", "[onnx][resnet]")
{
  REQUIRE(OnnxModels::initOnnxRuntime()); // no node here does it for us
  auto small = withClasses(3);
  auto large = withClasses(1000);
  auto medium = withClasses(10);

  // Fewer classes than 5: all of them, best first, nothing out of range.
  CHECK(names(top(small, {0.f, 2.f, 1.f})) == std::vector<std::string>{"c1", "c2", "c0"});

  std::vector<float> big(1000, 0.f);
  big[999] = 5.f;
  big[500] = 4.f;
  big[3] = 3.f;
  big[42] = 2.f;
  big[7] = 1.f;
  CHECK(names(top(large, big)) == std::vector<std::string>{"c999", "c500", "c3", "c42", "c7"});

  // After a larger model on the same thread, a smaller one still ranks its own range.
  std::vector<float> ten(10, 0.f);
  ten[9] = 5.f;
  ten[0] = 4.f;
  ten[5] = 3.f;
  ten[2] = 2.f;
  ten[8] = 1.f;
  CHECK(names(top(medium, ten)) == std::vector<std::string>{"c9", "c0", "c5", "c2", "c8"});
}

// A model that already outputs probabilities gets no second softmax, which
// would flatten them towards 1/N.
TEST_CASE("Resnet: probabilities are not softmaxed again", "[onnx][resnet]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  auto r = withClasses(3);
  auto out = top(r, {0.1f, 0.85f, 0.05f});
  REQUIRE(out.size() == 3);
  CHECK(out[0].name == "c1");
  CHECK(std::abs(out[0].probability - 0.85f) < 1e-6f);
  // Logits still go through the softmax.
  auto l = top(r, {0.f, 2.f, 1.f});
  CHECK(std::abs(l[0].probability - 0.665241f) < 1e-4f);
}

// The multi-task models end with valence and arousal, which are not part of
// the emotions' softmax.
TEST_CASE("EmotionNet: valence and arousal are passed through", "[onnx][emotionnet]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  std::vector<float> logits{1, 2, 3, 4, 5, 6, 7, 8, 0.25f, -0.5f};
  std::vector<int64_t> shape{1, 10};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value v[1]{Ort::Value::CreateTensor<float>(
      mem, logits.data(), logits.size(), shape.data(), shape.size())};
  OnnxModels::EmotionNet e;
  std::vector<OnnxModels::EmotionNet::recognition_type> out;
  e.processOutput({}, v, out);
  REQUIRE(out.size() == 10);
  float sum = 0;
  for(int i = 0; i < 8; i++)
    sum += out[i].probability;
  CHECK(std::abs(sum - 1.f) < 1e-5f);
  CHECK(out[7].name == "Surprise");
  CHECK(out[8].name == "Valence");
  CHECK(out[8].probability == 0.25f);
  CHECK(out[9].name == "Arousal");
  CHECK(out[9].probability == -0.5f);
}

// FER+ takes one 64x64 luma plane and has its own label order, not
// AffectNet's.
TEST_CASE("EmotionNet: FER+ gets luma and its labels", "[onnx][emotionnet][model]")
{
  const auto model = TestPaths::models() + "/image-processor/emotion-ferplus-8.onnx";
  const std::string face
      = TestPaths::images() + "/face.png";
  if(!TestPaths::haveAll({model, face}))
    SKIP("model or image not found");
  QImage q = QImage(QString::fromStdString(face))
                 .convertToFormat(QImage::Format_RGBA8888)
                 .scaled(224, 224, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
  std::vector<unsigned char> img(q.constBits(), q.constBits() + 224 * 224 * 4);
  const auto bytes = TestPaths::slurp(model);

  OnnxModels::EmotionNetDetector e;
  e.inputs.resolution.value = {224, 224};
  load(e, e.inputs.model, bytes, model);
  tick(e, img);
  REQUIRE(!e.inputs.model.current_model_invalid);
  const auto d = detections(e);
  REQUIRE(d.size() == 8);
  CHECK(d[0].first == "Neutral");
  CHECK(d[7].first == "Contempt");
  auto best = std::max_element(
      d.begin(), d.end(), [](auto& a, auto& b) { return a.second < b.second; });
  INFO("top: " << best->first << " " << best->second);
  CHECK((best->first == "Neutral" || best->first == "Happiness"));
}
