// Node failures (BUG-LEDGER X12): every generic node set its model invalid on
// any exception, and only picking another file cleared it, so one frame the
// model could not run (a resolution it rejects) disabled the node for good,
// silently. Load failures stay sticky; per-frame failures are reported once
// and skipped, with a backoff while they repeat.
#include <OnnxModels/ImageProcessor.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <iterator>
#include <string>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

struct Harness
{
  OnnxModels::ImageProcessor node;
  QImage image{64, 64, QImage::Format_RGBA8888};
  std::string name, bytes;

  Harness()
  {
    image.fill(Qt::darkCyan);
    node.inputs.normalization.value = OnnxModels::InputNormalization::DivBy255;
    node.worker.request = [this](std::unique_ptr<OnnxModels::InferJob> job) {
      if(auto done = OnnxModels::ImageProcessor::worker::work(std::move(job)))
        done(node);
    };
  }
  void load(const std::string& path, std::string content)
  {
    name = path;
    bytes = std::move(content);
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model.update(node);
  }
  void tick()
  {
    auto& t = node.inputs.image.texture;
    t.bytes = image.bits();
    t.width = image.width();
    t.height = image.height();
    t.changed = true;
    node.outputs.image.texture.changed = false;
    node();
  }
};

// Returns its input, but cannot run a bright image (tests/data/image).
const std::string brightFails = SCORE_ONNX_TEST_DATA_DIR "/image/bright_fails.onnx";
}

TEST_CASE("FailureLog: backoff while the same failure repeats", "[onnx][failures]")
{
  OnnxModels::FailureLog log;
  CHECK(log.ready());
  log.failed("Node", "m.onnx", "boom");
  CHECK(log.ready()); // first failure: retried on the next tick
  log.failed("Node", "m.onnx", "boom");
  CHECK_FALSE(log.ready()); // then every 2nd tick
  CHECK(log.ready());
  log.failed("Node", "m.onnx", "boom");
  int skipped = 0;
  while(!log.ready())
    skipped++;
  CHECK(skipped == 3); // every 4th
  for(int i = 0; i < 20; i++)
    log.failed("Node", "m.onnx", "boom");
  skipped = 0;
  while(!log.ready())
    skipped++;
  CHECK(skipped == 63); // capped
  log.failed("Node", "m.onnx", "another error"); // a new failure starts over
  CHECK(log.ready());
  log.succeeded();
  CHECK(log.ready());
  CHECK(log.last.empty());
}

TEST_CASE("Image Processor: a frame the model rejects does not disable it", "[onnx][failures]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Harness h;
  h.load(brightFails, slurp(brightFails));

  h.image.fill(Qt::white);
  for(int i = 0; i < 5; i++)
    h.tick();
  CHECK_FALSE(h.node.inputs.model.current_model_invalid);
  CHECK_FALSE(h.node.outputs.image.texture.changed);
  CHECK(h.node.failures.consecutive >= 2);

  // A darker image runs again, after the backoff.
  h.image.fill(Qt::darkCyan);
  bool produced = false;
  for(int i = 0; i < 80 && !produced; i++)
  {
    h.tick();
    produced = h.node.outputs.image.texture.changed;
  }
  CHECK(produced);
  CHECK(h.node.outputs.image.texture.width == 32);
  CHECK(h.node.failures.consecutive == 0);
}

TEST_CASE("Image Processor: a model that cannot load stays disabled until another file", "[onnx][failures]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Harness h;
  h.load("not-a-model.onnx", std::string(64, 'x'));
  h.tick();
  CHECK(h.node.inputs.model.current_model_invalid);
  h.tick();
  CHECK(h.node.inputs.model.current_model_invalid);

  h.load(brightFails, slurp(brightFails));
  CHECK_FALSE(h.node.inputs.model.current_model_invalid);
  h.tick();
  CHECK(h.node.outputs.image.texture.changed);
}
