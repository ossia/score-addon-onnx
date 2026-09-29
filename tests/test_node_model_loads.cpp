// The Image Processor, Video Processor, Image Generator and Pose Detector load
// their models on the worker: a tick never builds a session, the model that
// runs keeps running until the new one arrives, the model of a file that was
// replaced while it loaded is dropped (and the latest file loaded after it),
// and a file that cannot be loaded is not loaded again until another one is
// picked.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/AudioAnalyzer.hpp>
#include <OnnxModels/AudioProcessor.hpp>
#include <OnnxModels/GeometryProcessor.hpp>
#include <OnnxModels/ImageGenerator.hpp>
#include <OnnxModels/ImageProcessor.hpp>
#include <OnnxModels/PoseDetector.hpp>
#include <OnnxModels/SequenceProcessor.hpp>
#include <OnnxModels/TextToken.hpp>
#include <OnnxModels/VideoProcessor.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace
{
const std::string dir = SCORE_ONNX_TEST_DATA_DIR "/image/";
const std::string identity = dir + "identity_dyn.onnx";
const std::string black = dir + "black_dyn.onnx";
const std::string missing = dir + "missing.onnx";

template <typename Job>
int loadsIn(const std::deque<std::unique_ptr<Job>>& jobs)
{
  return (int)std::count_if(jobs.begin(), jobs.end(), [](const auto& j) {
    return isModelJob(*j) && j->load.kind == decltype(j->load)::Kind::Load;
  });
}

// Image and Video Processor: a red image through a model that returns it
// (identity) or returns black.
template <typename Node, typename Job>
struct ImageNode
{
  Node node;
  QueuedWorker<Node, Job> w;
  QImage image{64, 64, QImage::Format_RGBA8888};
  std::string name, bytes;

  ImageNode()
  {
    image.fill(Qt::red);
    node.inputs.normalization.value = OnnxModels::InputNormalization::DivBy255;
    node.inputs.resolution.value = {64, 64}; // inference runs inline
    w.attach(node);
  }
  void pick(const std::string& path)
  {
    name = path;
    bytes = path == missing ? std::string(64, 'x') : TestPaths::slurp(path);
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model.update(node);
  }
  // Whether the tick produced an image.
  bool tick()
  {
    auto& t = node.inputs.image.texture;
    t.bytes = image.bits();
    t.width = image.width();
    t.height = image.height();
    t.changed = true;
    node.outputs.image.texture.changed = false;
    node();
    return node.outputs.image.texture.changed;
  }
  int red() const { return node.outputs.image.texture.bytes[0]; }
  int loads() const { return loadsIn(w.jobs); }
};

template <typename Node, typename Job>
void loadsOnTheWorker()
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  ImageNode<Node, Job> n;
  n.pick(identity);
  CHECK_FALSE(n.tick()); // no model yet: the tick does not build one
  REQUIRE(n.loads() == 1);
  n.w.drain(n.node);
  REQUIRE(n.tick());
  CHECK(n.red() == 255);
}

template <typename Node, typename Job>
void previousModelRuns()
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  ImageNode<Node, Job> n;
  n.pick(identity);
  n.tick();
  n.w.drain(n.node);

  n.pick(black);
  REQUIRE(n.tick());
  CHECK(n.red() == 255); // the identity model still runs
  REQUIRE(n.loads() == 1);
  n.w.run(n.node);
  REQUIRE(n.tick());
  CHECK(n.red() == 0);
  n.w.drain(n.node); // frees the identity model
  CHECK(n.w.jobs.empty());
}

template <typename Node, typename Job>
void outdatedLoadDropped()
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  ImageNode<Node, Job> n;
  n.pick(identity);
  n.tick();
  n.pick(black);
  n.tick();
  REQUIRE(n.loads() == 1); // black waits for identity
  n.w.run(n.node);         // identity, replaced while it loaded: black next
  CHECK_FALSE(n.tick());
  n.w.drain(n.node);
  REQUIRE(n.tick());
  CHECK(n.red() == 0);
}

template <typename Node, typename Job>
void failedLoadNotRetried()
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  ImageNode<Node, Job> n;
  n.pick(identity);
  n.tick();
  n.w.drain(n.node);
  REQUIRE(n.tick());

  // The file picked next cannot load: the node stops, rather than keep
  // running the previous model as if it were the new one.
  n.pick(missing);
  n.tick();
  REQUIRE(n.loads() == 1);
  n.w.drain(n.node);
  CHECK(n.node.inputs.model.current_model_invalid);
  for(int i = 0; i < 5; i++)
    CHECK_FALSE(n.tick());
  CHECK(n.w.jobs.empty());

  n.pick(identity);
  CHECK_FALSE(n.node.inputs.model.current_model_invalid);
  n.tick();
  CHECK(n.loads() == 1);
  n.w.drain(n.node);
  CHECK(n.tick());
}

// A file that fails to load, replaced before its failure reached the node:
// the failure is not the new file's, which still loads.
template <typename Node, typename Job>
void failureOfAReplacedFile()
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  ImageNode<Node, Job> n;
  n.pick(missing);
  n.tick();
  REQUIRE(n.loads() == 1);
  n.pick(identity);
  n.w.run(n.node); // the failure arrives before the tick that sees identity
  CHECK_FALSE(n.node.inputs.model.current_model_invalid);
  n.tick();
  REQUIRE(n.loads() == 1);
  n.w.drain(n.node);
  REQUIRE(n.tick());
  CHECK(n.red() == 255);
}

using Image = OnnxModels::ImageProcessor;
using ImageJob = OnnxModels::InferJob;
using Video = OnnxModels::VideoProcessor;
using VideoJob = OnnxModels::VideoInferJob;
}

TEST_CASE("Image Processor: the model loads on the worker", "[onnx][image][loads]")
{
  loadsOnTheWorker<Image, ImageJob>();
}
TEST_CASE("Image Processor: the previous model runs until the new one arrives", "[onnx][image][loads]")
{
  previousModelRuns<Image, ImageJob>();
}
TEST_CASE("Image Processor: a model replaced while it loads is dropped", "[onnx][image][loads]")
{
  outdatedLoadDropped<Image, ImageJob>();
}
TEST_CASE("Image Processor: a model that cannot load is not loaded again", "[onnx][image][loads]")
{
  failedLoadNotRetried<Image, ImageJob>();
}

TEST_CASE("Image Processor: the failure of a file replaced meanwhile is not the new file's", "[onnx][image][loads]")
{
  failureOfAReplacedFile<Image, ImageJob>();
}

TEST_CASE("Video Processor: the model loads on the worker", "[onnx][video][loads]")
{
  loadsOnTheWorker<Video, VideoJob>();
}
TEST_CASE("Video Processor: the previous model runs until the new one arrives", "[onnx][video][loads]")
{
  previousModelRuns<Video, VideoJob>();
}
TEST_CASE("Video Processor: a model replaced while it loads is dropped", "[onnx][video][loads]")
{
  outdatedLoadDropped<Video, VideoJob>();
}
TEST_CASE("Video Processor: a model that cannot load is not loaded again", "[onnx][video][loads]")
{
  failedLoadNotRetried<Video, VideoJob>();
}

TEST_CASE("Video Processor: the failure of a file replaced meanwhile is not the new file's", "[onnx][video][loads]")
{
  failureOfAReplacedFile<Video, VideoJob>();
}

namespace
{
// Image Generator: gen_bytes' first value, 20, is a pixel of 20; gen_tanh's
// is not.
struct Generator
{
  OnnxModels::ImageGenerator node;
  QueuedWorker<OnnxModels::ImageGenerator, OnnxModels::GenJob> w;
  std::string name, bytes, map_name, map_bytes;

  Generator() { w.attach(node); }
  void pick(const std::string& path, const std::string& mapping = {})
  {
    name = path;
    bytes = path == missing ? std::string(64, 'x') : TestPaths::slurp(path);
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model.update(node);
    map_name = mapping;
    map_bytes = mapping.empty() ? std::string{} : std::string(64, 'x');
    node.inputs.mapping_model.file.bytes = map_bytes;
    node.inputs.mapping_model.file.filename = map_name;
  }
  void tick()
  {
    node.outputs.image.texture.changed = false;
    node();
  }
  int loads() const { return loadsIn(w.jobs); }
};
}

TEST_CASE("Image Generator: the models load on the worker", "[onnx][generator][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Generator g;
  g.pick(dir + "gen_bytes.onnx");
  g.tick();
  REQUIRE(g.w.jobs.size() == 1); // the load, and no generation yet
  REQUIRE(g.loads() == 1);
  g.w.drain(g.node);
  g.tick();
  g.w.drain(g.node);
  auto& out = g.node.outputs.image.texture;
  REQUIRE(out.changed);
  CHECK(out.bytes[0] == 20);
}

TEST_CASE("Image Generator: a model replaced while it loads is dropped", "[onnx][generator][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Generator g;
  g.pick(dir + "gen_tanh.onnx");
  g.tick();
  g.pick(dir + "gen_bytes.onnx");
  g.tick();
  REQUIRE(g.loads() == 1); // gen_bytes waits for gen_tanh
  g.w.run(g.node);         // gen_tanh, replaced while it loaded
  g.w.drain(g.node);
  g.tick();
  g.w.drain(g.node);
  auto& out = g.node.outputs.image.texture;
  REQUIRE(out.changed);
  CHECK(out.bytes[0] == 20);
}

TEST_CASE("Image Generator: files that cannot load are not loaded again", "[onnx][generator][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Generator g;
  // The mapping model cannot be read: the chain does not load.
  g.pick(dir + "gen_bytes.onnx", missing);
  g.tick();
  g.w.drain(g.node);
  for(int i = 0; i < 5; i++)
    g.tick();
  CHECK(g.w.jobs.empty());
  CHECK_FALSE(g.node.outputs.image.texture.changed);

  // Another file on either port loads again.
  g.pick(dir + "gen_bytes.onnx");
  g.tick();
  CHECK(g.loads() == 1);
  g.w.drain(g.node);
  g.tick();
  g.w.drain(g.node);
  CHECK(g.node.outputs.image.texture.changed);
}

namespace
{
struct Pose
{
  OnnxModels::PoseDetector node;
  QueuedWorker<OnnxModels::PoseDetector, OnnxModels::PoseJob> w;
  QImage image{64, 64, QImage::Format_RGBA8888};
  std::string name, bytes, body;

  Pose()
  {
    image.fill(Qt::darkCyan);
    node.inputs.min_confidence.value = 0.f;
    w.attach(node);
  }
  void pick(const std::string& path)
  {
    name = path;
    bytes = path == missing ? std::string(64, 'x') : TestPaths::slurp(path);
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model.update(node);
  }
  // Whether the tick produced an image.
  bool tick()
  {
    auto& t = node.inputs.image.texture;
    t.bytes = image.bits();
    t.width = image.width();
    t.height = image.height();
    t.changed = true;
    node.outputs.image.texture.changed = false;
    node();
    return node.outputs.image.texture.changed;
  }
  template <typename Slot>
  int loads(Slot OnnxModels::PoseJob::*slot) const
  {
    return (int)std::count_if(w.jobs.begin(), w.jobs.end(), [&](const auto& j) {
      return ((*j).*slot).kind == Slot::Kind::Load;
    });
  }
};
const std::string simcc = SCORE_ONNX_TEST_DATA_DIR "/pose/simcc_dynhw.onnx";
}

TEST_CASE("Pose Detector: the landmark model loads on the worker", "[onnx][pose][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Pose p;
  p.pick(simcc);
  CHECK_FALSE(p.tick()); // waits for its model
  REQUIRE(p.loads(&OnnxModels::PoseJob::landmark) == 1);
  p.w.drain(p.node);
  CHECK(p.tick());
  CHECK(p.w.jobs.empty());
}

TEST_CASE("Pose Detector: a model that cannot load is not loaded again", "[onnx][pose][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Pose p;
  p.pick(missing);
  p.tick();
  REQUIRE(p.loads(&OnnxModels::PoseJob::landmark) == 1);
  p.w.drain(p.node);
  CHECK(p.node.inputs.model.current_model_invalid);
  for(int i = 0; i < 5; i++)
    CHECK_FALSE(p.tick());
  CHECK(p.w.jobs.empty());

  p.pick(simcc);
  p.tick();
  p.w.drain(p.node);
  CHECK(p.tick());
}

TEST_CASE("Pose Detector: the failure of a file replaced meanwhile is not the new file's", "[onnx][pose][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Pose p;
  p.pick(missing);
  p.tick();
  REQUIRE(p.loads(&OnnxModels::PoseJob::landmark) == 1);
  p.pick(simcc);
  p.w.run(p.node); // the failure arrives before the tick that sees simcc
  CHECK_FALSE(p.node.inputs.model.current_model_invalid);
  p.tick();
  REQUIRE(p.loads(&OnnxModels::PoseJob::landmark) == 1);
  p.w.drain(p.node);
  CHECK(p.tick());
}

TEST_CASE("Pose Detector: a Body Model that cannot be read is not read again", "[onnx][pose][mhr][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Pose p;
  p.body = std::string(64, 'x');
  p.node.inputs.body_model.file.bytes = p.body;
  p.node.inputs.body_model.file.filename = missing;
  p.tick();
  REQUIRE(p.loads(&OnnxModels::PoseJob::body) == 1);
  p.w.drain(p.node);
  for(int i = 0; i < 5; i++)
    p.tick();
  CHECK(p.w.jobs.empty());
}

// Every node on OnnxModels::ModelLoader: a file that cannot be loaded is
// asked for once, however many ticks follow.
namespace
{
const std::string notAModel = SCORE_ONNX_TEST_DATA_DIR "/audio/make_fixtures.py";

template <typename Node, typename Job, typename Tick>
int loadsOfAFailingFile(Node& node, Tick tick)
{
  QueuedWorker<Node, Job> w;
  w.attach(node);
  const auto bytes = TestPaths::slurp(notAModel);
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = notAModel;
  node.inputs.model.update(node);
  int loads = 0;
  for(int i = 0; i < 5; i++)
  {
    tick();
    for(auto& j : w.jobs)
      loads += isLoadJob(*j);
    w.drain(node);
  }
  CHECK(node.inputs.model.current_model_invalid);
  return loads;
}

// A file that cannot be loaded, replaced before its failure reached the
// node: the failure is not the new file's, which is asked for at the next
// tick.
template <typename Node, typename Job, typename Tick>
int loadsAfterAReplacedFailure(Node& node, Tick tick)
{
  QueuedWorker<Node, Job> w;
  w.attach(node);
  const auto bytes = TestPaths::slurp(notAModel);
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = notAModel;
  node.inputs.model.update(node);
  tick();
  REQUIRE(w.jobs.size() == 1);

  const std::string other = bytes, other_name = notAModel + ".replaced";
  node.inputs.model.file.bytes = other;
  node.inputs.model.file.filename = other_name;
  node.inputs.model.update(node);
  w.run(node); // the failure arrives before the tick that sees the new file
  CHECK_FALSE(node.inputs.model.current_model_invalid);
  tick();
  int loads = 0;
  for(auto& j : w.jobs)
    loads += isLoadJob(*j);
  w.drain(node);
  return loads;
}

struct AudioBuffers
{
  std::vector<float> in = std::vector<float>(512, 0.25f), out = std::vector<float>(512);
  float* ins[1]{in.data()};
  float* outs[1]{out.data()};
};
}

TEST_CASE("Geometry Processor: a model that cannot load is asked for once", "[onnx][geometry][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::GeometryProcessor node;
  CHECK(loadsOfAFailingFile<OnnxModels::GeometryProcessor, OnnxModels::GeomInferJob>(
            node, [&] { node(); })
        == 1);
}

TEST_CASE("Sequence Processor: a model that cannot load is asked for once", "[onnx][sequence][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::SequenceProcessor node;
  node.inputs.in.value = {1.f, 2.f};
  CHECK(loadsOfAFailingFile<OnnxModels::SequenceProcessor, OnnxModels::SeqInferJob>(
            node, [&] { node(); })
        == 1);
}

TEST_CASE("Text Token: a model that cannot load is asked for once", "[onnx][texttoken][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::TextToken node;
  AudioBuffers b;
  node.prepare({.rate = 48000., .output_channels = 1, .frames = 512});
  node.outputs.audio.samples = b.outs;
  node.outputs.audio.channels = 1;
  CHECK(loadsOfAFailingFile<OnnxModels::TextToken, OnnxModels::TokenInferJob>(
            node, [&] { node(512); })
        == 1);
}

TEST_CASE("Audio Processor: a model that cannot load is asked for once", "[onnx][audio][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::AudioProcessor node;
  AudioBuffers b;
  node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = 512});
  node.inputs.audio.samples = b.ins;
  node.inputs.audio.channels = 1;
  node.outputs.audio.samples = b.outs;
  node.outputs.audio.channels = 1;
  CHECK(loadsOfAFailingFile<OnnxModels::AudioProcessor, OnnxModels::AudioInferJob>(
            node, [&] { node(512); })
        == 1);
}

TEST_CASE("Audio Analyzer: a model that cannot load is asked for once", "[onnx][audio][analyzer][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::AudioAnalyzer node;
  AudioBuffers b;
  node.prepare({.rate = 16000., .input_channels = 1, .frames = 512});
  node.inputs.audio.samples = b.ins;
  node.inputs.audio.channels = 1;
  CHECK(loadsOfAFailingFile<OnnxModels::AudioAnalyzer, OnnxModels::AnalyzerJob>(
            node, [&] { node(512); })
        == 1);
}

TEST_CASE("Every loader node: the failure of a file replaced meanwhile is not the new file's", "[onnx][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  SECTION("Geometry Processor")
  {
    OnnxModels::GeometryProcessor node;
    CHECK(loadsAfterAReplacedFailure<OnnxModels::GeometryProcessor, OnnxModels::GeomInferJob>(
              node, [&] { node(); })
          == 1);
  }
  SECTION("Sequence Processor")
  {
    OnnxModels::SequenceProcessor node;
    node.inputs.in.value = {1.f, 2.f};
    CHECK(loadsAfterAReplacedFailure<OnnxModels::SequenceProcessor, OnnxModels::SeqInferJob>(
              node, [&] { node(); })
          == 1);
  }
  SECTION("Text Token")
  {
    OnnxModels::TextToken node;
    AudioBuffers b;
    node.prepare({.rate = 48000., .output_channels = 1, .frames = 512});
    node.outputs.audio.samples = b.outs;
    node.outputs.audio.channels = 1;
    CHECK(loadsAfterAReplacedFailure<OnnxModels::TextToken, OnnxModels::TokenInferJob>(
              node, [&] { node(512); })
          == 1);
  }
  SECTION("Audio Processor")
  {
    OnnxModels::AudioProcessor node;
    AudioBuffers b;
    node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = 512});
    node.inputs.audio.samples = b.ins;
    node.inputs.audio.channels = 1;
    node.outputs.audio.samples = b.outs;
    node.outputs.audio.channels = 1;
    CHECK(loadsAfterAReplacedFailure<OnnxModels::AudioProcessor, OnnxModels::AudioInferJob>(
              node, [&] { node(512); })
          == 1);
  }
  SECTION("Audio Analyzer")
  {
    OnnxModels::AudioAnalyzer node;
    AudioBuffers b;
    node.prepare({.rate = 16000., .input_channels = 1, .frames = 512});
    node.inputs.audio.samples = b.ins;
    node.inputs.audio.channels = 1;
    CHECK(loadsAfterAReplacedFailure<OnnxModels::AudioAnalyzer, OnnxModels::AnalyzerJob>(
              node, [&] { node(512); })
          == 1);
  }
}

// A settings change builds a pipeline from the running session, and only the
// latest settings are built after the one in flight.
TEST_CASE("Audio Processor: settings changed while one builds build once more", "[onnx][audio][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::AudioProcessor node;
  AudioBuffers b;
  node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = 512});
  node.inputs.model_rate.value = 48000;
  node.inputs.audio.samples = b.ins;
  node.inputs.audio.channels = 1;
  node.outputs.audio.samples = b.outs;
  node.outputs.audio.channels = 1;
  QueuedWorker<OnnxModels::AudioProcessor, OnnxModels::AudioInferJob> w;
  w.attach(node);
  const std::string name = SCORE_ONNX_TEST_DATA_DIR "/audio/identity_512.onnx";
  const auto bytes = TestPaths::slurp(name);
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = name;
  node(512);
  w.drain(node);
  const auto* session = node.sessionForTest();
  REQUIRE(session);

  node.inputs.overlap.value = OnnxModels::AudioOverlap::Half;
  node(512);
  node.inputs.overlap.value = OnnxModels::AudioOverlap::ThreeQuarters;
  node(512);
  node.inputs.overlap.value = OnnxModels::AudioOverlap::None;
  node(512);
  REQUIRE(w.jobs.size() == 1); // the others wait for the first
  int loads = 0;
  while(!w.jobs.empty())
  {
    loads += isLoadJob(*w.jobs.front());
    w.run(node);
  }
  CHECK(loads == 2); // Half, then None: ThreeQuarters was never built
  CHECK(node.sessionForTest() == session);
}
