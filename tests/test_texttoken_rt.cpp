// Text Token Processor on the audio thread: the model (and a sibling
// voices.bin) loads in a worker job, every inference is a worker job, and
// replaced utterances and models are freed on the worker.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/TextToken.hpp>

#include <catch2/catch_test_macros.hpp>

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace
{
using Job = OnnxModels::TokenInferJob;

struct Deferred
{
  OnnxModels::TextToken node;
  std::string name, bytes;
  std::deque<std::unique_ptr<Job>> jobs;
  std::vector<float> out = std::vector<float>(64, 1.f);
  float* outs[1]{out.data()};

  explicit Deferred(const std::string& model)
      : name{model}
      , bytes{TestPaths::slurp(model)}
  {
    node.prepare({.rate = 48000., .output_channels = 1, .frames = 64});
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.outputs.audio.samples = outs;
    node.outputs.audio.channels = 1;
    node.worker.request = [this](std::unique_ptr<Job> job) { jobs.push_back(std::move(job)); };
  }
  Job::Kind complete()
  {
    REQUIRE(!jobs.empty());
    auto job = std::move(jobs.front());
    jobs.pop_front();
    const auto kind = job->kind;
    runJob(node, std::move(job));
    return kind;
  }
};
}

TEST_CASE("Text Token: the model loads and a text encoder runs on the worker", "[onnx][texttoken]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred d{SCORE_ONNX_TEST_DATA_DIR "/tts/int32/encoder.onnx"};
  d.node.inputs.tokens.value = {5, 6, 7};
  d.node(64);
  REQUIRE(d.jobs.size() == 1);
  CHECK(d.jobs.front()->kind == Job::Kind::Build);
  CHECK(d.complete() == Job::Kind::Build);

  d.node(64);
  REQUIRE(d.jobs.size() == 1);
  CHECK(d.jobs.front()->kind == Job::Kind::Infer); // not run inline
  CHECK(d.node.outputs.data.value.empty());
  CHECK(d.complete() == Job::Kind::Infer);
  CHECK(d.node.outputs.data.value.size() == 3);
}

TEST_CASE("Text Token: a replaced utterance is freed on the worker", "[onnx][texttoken]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred d{SCORE_ONNX_TEST_DATA_DIR "/tts/sidecar/tts.onnx"};
  d.node.inputs.tokens.value = {1, 2};
  d.node(64);
  d.complete(); // Build
  d.node(64);
  CHECK(d.complete() == Job::Kind::Infer); // the first utterance
  CHECK(d.jobs.empty());

  d.node.inputs.tokens.value = {3, 4};
  d.node(64);
  CHECK(d.complete() == Job::Kind::Infer); // replaces it
  REQUIRE(d.jobs.size() == 1);
  CHECK(d.jobs.front()->kind == Job::Kind::Dispose);
  CHECK(!d.jobs.front()->utterance.samples.empty());
  CHECK(d.complete() == Job::Kind::Dispose);
}
