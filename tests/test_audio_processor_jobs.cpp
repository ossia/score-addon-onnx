// The worker jobs of the Audio Processor across a pipeline change and a model
// that fails to load: inference jobs never overlap, and a file that failed is
// loaded again when it is picked again.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/AudioProcessor.hpp>
#include <OnnxModels/GeometryProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace
{
using Job = OnnxModels::AudioInferJob;

// Builds and disposals run right away; inference jobs wait, as on a busy
// worker.
struct Deferred
{
  OnnxModels::AudioProcessor node;
  std::string name, bytes;
  std::deque<std::unique_ptr<Job>> held;
  int builds = 0;
  std::vector<float> in = std::vector<float>(512, 0.25f), out = std::vector<float>(512);
  float* ins[1]{in.data()};
  float* outs[1]{out.data()};

  Deferred()
  {
    node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = 512});
    node.inputs.model_rate.value = 48000;
    node.inputs.audio.samples = ins;
    node.inputs.audio.channels = 1;
    node.outputs.audio.samples = outs;
    node.outputs.audio.channels = 1;
    node.worker.request = [this](std::unique_ptr<Job> job) {
      if(!isModelJob(*job))
      {
        held.push_back(std::move(job));
        return;
      }
      if(isLoadJob(*job))
        ++builds;
      runJob(node, std::move(job));
    };
  }
  void load(const std::string& path)
  {
    name = path;
    bytes = TestPaths::slurp(path);
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model.update(node);
  }
  void ticks(int n)
  {
    for(int i = 0; i < n; i++)
      node(512);
  }
  void complete()
  {
    REQUIRE(!held.empty());
    auto job = std::move(held.front());
    held.pop_front();
    runJob(node, std::move(job));
  }
};
}

TEST_CASE("Audio Processor: a pipeline change does not let two jobs overlap", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred d;
  // A free-length model on blocks above one second runs on the worker.
  d.node.inputs.block.value = 49152;
  d.load(SCORE_ONNX_TEST_DATA_DIR "/audio/identity_dyn.onnx");
  d.ticks(100);
  REQUIRE(d.builds == 1);
  REQUIRE(d.held.size() == 1);

  // A new setting: a new pipeline, while the old one's job still runs.
  d.node.inputs.overlap.value = OnnxModels::AudioOverlap::Half;
  d.ticks(200);
  REQUIRE(d.builds == 2);
  CHECK(d.held.size() == 1); // no second job before the first returned

  // It returns: its block is dropped, and the new pipeline's job starts.
  d.complete();
  d.ticks(1);
  CHECK(d.held.size() == 1);
  d.complete();
}

TEST_CASE("Audio Processor: a file that failed loads again when picked again", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred d;
  d.load(SCORE_ONNX_TEST_DATA_DIR "/audio/identity_512.onnx");
  d.ticks(2);
  REQUIRE(d.builds == 1);
  REQUIRE_FALSE(d.node.inputs.model.current_model_invalid);

  // Not a model: the build fails, the running pipeline stays silent.
  d.load(SCORE_ONNX_TEST_DATA_DIR "/audio/make_fixtures.py");
  d.ticks(2);
  REQUIRE(d.builds == 2);
  CHECK(d.node.inputs.model.current_model_invalid);
  d.ticks(4);
  CHECK(d.builds == 2); // not retried on its own

  // Picked again (say, after fixing it): it is loaded again.
  d.load(SCORE_ONNX_TEST_DATA_DIR "/audio/make_fixtures.py");
  d.ticks(1);
  CHECK(d.builds == 3);
}

TEST_CASE("Geometry Processor: a file that failed loads again when picked again", "[onnx][geometry]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::GeometryProcessor node;
  int builds = 0;
  node.worker.request = [&](std::unique_ptr<OnnxModels::GeomInferJob> job) {
    if(isLoadJob(*job))
      ++builds;
    runJob(node, std::move(job));
  };
  const std::string name = SCORE_ONNX_TEST_DATA_DIR "/image/make_fixtures.py";
  const auto bytes = TestPaths::slurp(name);
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = name;
  node.inputs.model.update(node);
  node();
  REQUIRE(builds == 1);
  CHECK(node.inputs.model.current_model_invalid);
  node();
  CHECK(builds == 1);

  // Picked again: a new mapping of the file.
  const auto again = TestPaths::slurp(name);
  node.inputs.model.file.bytes = again;
  node.inputs.model.update(node);
  node();
  CHECK(builds == 2);
}
