// A job of the previous model finishing after a model swap (BUG-LEDGER X14):
// Image Processor, Geometry Processor and Image Generator published its
// result, and the Image Generator also latched its Auto mapping for the new
// model. Each now drops the results of an older generation.
#include <OnnxModels/GeometryProcessor.hpp>
#include <OnnxModels/ImageGenerator.hpp>
#include <OnnxModels/ImageProcessor.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

const std::string dir = SCORE_ONNX_TEST_DATA_DIR "/image/";

// A node whose jobs wait until complete(), as on a busy worker.
template <typename Node, typename Job>
struct Deferred
{
  Node node;
  std::string name, bytes;
  std::unique_ptr<Job> held;

  Deferred()
  {
    node.worker.request = [this](std::unique_ptr<Job> job) { held = std::move(job); };
  }
  void load(const std::string& path)
  {
    name = path;
    bytes = slurp(path);
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model.update(node);
  }
  void complete()
  {
    REQUIRE(held);
    if(auto done = Node::worker::work(std::move(held)))
      done(node);
  }
};
}

TEST_CASE("Image Processor: a job of the old model is dropped after a swap", "[onnx][image]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred<OnnxModels::ImageProcessor, OnnxModels::InferJob> p;
  QImage image{64, 64, QImage::Format_RGBA8888};
  image.fill(Qt::red);
  p.node.inputs.normalization.value = OnnxModels::InputNormalization::DivBy255;
  p.node.inputs.resolution.value = {768, 768}; // above 512²: the worker path
  auto tick = [&] {
    auto& t = p.node.inputs.image.texture;
    t.bytes = image.bits();
    t.width = image.width();
    t.height = image.height();
    t.changed = true;
    p.node.outputs.image.texture.changed = false;
    p.node();
  };

  p.load(dir + "identity_dyn.onnx");
  tick();
  REQUIRE(p.held);
  p.load(dir + "black_dyn.onnx");
  tick(); // reloads while the identity job runs
  p.complete();
  CHECK_FALSE(p.node.outputs.image.texture.changed); // the red image is stale

  tick();
  p.complete();
  auto& out = p.node.outputs.image.texture;
  REQUIRE(out.changed);
  CHECK(out.bytes[0] == 0); // black: the new model's
}

TEST_CASE("Geometry Processor: a job of the old model is dropped after a swap", "[onnx][geometry]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred<OnnxModels::GeometryProcessor, OnnxModels::GeomInferJob> g;
  constexpr int n = 10000; // above 8192 points: the worker path
  auto tick = [&] {
    g.node.inputs.cloud.value.assign(n * 3, 2.f);
    g.node.inputs.point_count.value = n;
    g.node.outputs.cloud.value.clear();
    g.node();
  };

  g.load(dir + "cloud_plus1.onnx");
  tick();
  REQUIRE(g.held);
  g.load(dir + "cloud_zero.onnx");
  tick();
  g.complete();
  CHECK(g.node.outputs.cloud.value.empty()); // the +1 cloud is stale

  tick();
  g.complete();
  REQUIRE(!g.node.outputs.cloud.value.empty());
  CHECK(g.node.outputs.cloud.value[0] == 0.f);
}

TEST_CASE("Image Generator: the old model's Auto mapping does not carry over", "[onnx][generator]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred<OnnxModels::ImageGenerator, OnnxModels::GenJob> g;
  auto tick = [&] {
    g.node.outputs.image.texture.changed = false;
    g.node();
  };

  // tanh-range generator: Auto would pick Denormalize.
  g.load(dir + "gen_tanh.onnx");
  tick();
  REQUIRE(g.held);
  // 0..255 generator: its first value, 20, is a pixel of 20.
  g.load(dir + "gen_bytes.onnx");
  tick();
  g.complete();
  CHECK_FALSE(g.node.outputs.image.texture.changed);

  tick();
  g.complete();
  auto& out = g.node.outputs.image.texture;
  REQUIRE(out.changed);
  // With tanh's Denormalize latched, (20 + 1) * 127.5 saturates at 255.
  CHECK(out.bytes[0] == 20);
}
