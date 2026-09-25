// Audio Processor on the audio thread (BUG-LEDGER A9): a model change built
// the ORT session there, and a settings change (Model Rate, Overlap, Mel,
// Block) reallocated the rings, resamplers and vocoder tables there. Both now
// happen in a worker job; the audio thread swaps the result in and hands the
// old pipeline back to the worker to be freed.
#include <OnnxModels/AudioAnalyzer.hpp>
#include <OnnxModels/AudioProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

using Job = OnnxModels::AudioInferJob;

struct Deferred
{
  OnnxModels::AudioProcessor node;
  std::string name = SCORE_ONNX_TEST_DATA_DIR "/audio/identity_512.onnx";
  std::string bytes = slurp(name);
  std::deque<std::unique_ptr<Job>> jobs;
  std::vector<float> in = std::vector<float>(512, 0.25f), out = std::vector<float>(512, 1.f);
  float* ins[1]{in.data()};
  float* outs[1]{out.data()};

  Deferred()
  {
    node.prepare({.rate = 48000., .input_channels = 1, .output_channels = 1, .frames = 512});
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.model_rate.value = 48000;
    node.inputs.audio.samples = ins;
    node.inputs.audio.channels = 1;
    node.outputs.audio.samples = outs;
    node.outputs.audio.channels = 1;
    node.worker.request = [this](std::unique_ptr<Job> job) { jobs.push_back(std::move(job)); };
  }
  void tick() { node(512); }
  // Runs the oldest queued job on "the worker" and applies its result.
  Job::Kind complete()
  {
    REQUIRE(!jobs.empty());
    auto job = std::move(jobs.front());
    jobs.pop_front();
    const auto kind = job->kind;
    if(auto done = OnnxModels::AudioProcessor::worker::work(std::move(job)))
      done(node);
    return kind;
  }
  bool silent() const
  {
    for(float x : out)
      if(x != 0.f)
        return false;
    return true;
  }
};
}

TEST_CASE("Audio Processor: the model loads on the worker", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred d;
  d.tick();
  REQUIRE(d.jobs.size() == 1);
  CHECK(d.jobs.front()->kind == Job::Kind::Build);
  CHECK(d.jobs.front()->build.path == d.name);
  CHECK(d.silent()); // nothing to play until the pipeline arrives
  d.tick();
  CHECK(d.jobs.size() == 1); // not requested twice

  CHECK(d.complete() == Job::Kind::Build);
  for(int i = 0; i < 4; i++)
    d.tick();
  CHECK(d.jobs.empty()); // a light model: run inline
  CHECK(std::abs(d.out.back() - 0.25f) < 1e-6f);
}

TEST_CASE("Audio Processor: a settings change is prepared on the worker", "[onnx][audio]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred d;
  d.tick();
  d.complete();
  for(int i = 0; i < 4; i++)
    d.tick();

  // Overlap: a new pipeline with the same session.
  d.node.inputs.overlap.value = OnnxModels::AudioOverlap::Half;
  d.tick();
  REQUIRE(d.jobs.size() == 1);
  auto& build = *d.jobs.front();
  CHECK(build.kind == Job::Kind::Build);
  CHECK(build.ctx != nullptr); // reused, not loaded again
  const auto session = build.ctx;

  // Meanwhile the running pipeline keeps playing.
  d.tick();
  CHECK_FALSE(d.silent());

  CHECK(d.complete() == Job::Kind::Build);
  // The old pipeline goes back to the worker, as its last owner.
  REQUIRE(d.jobs.size() == 1);
  CHECK(d.jobs.front()->kind == Job::Kind::Dispose);
  REQUIRE(d.jobs.front()->pipeline);
  CHECK(d.jobs.front()->pipeline.use_count() == 1);
  CHECK(d.jobs.front()->pipeline->ctx == session); // the same session
  CHECK(d.complete() == Job::Kind::Dispose);
}

// Same for the Audio Analyzer, which has no other worker job.
TEST_CASE("Audio Analyzer: the model loads on the worker", "[onnx][audio][analyzer]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::AudioAnalyzer node;
  const std::string name = SCORE_ONNX_TEST_DATA_DIR "/aux/audio_flag.onnx";
  const std::string bytes = slurp(name);
  node.prepare({.rate = 16000., .input_channels = 1, .frames = 512});
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = name;
  std::vector<float> block(512, 0.25f);
  float* ch[1]{block.data()};
  node.inputs.audio.samples = ch;
  node.inputs.audio.channels = 1;
  std::deque<std::unique_ptr<OnnxModels::AnalyzerJob>> jobs;
  node.worker.request = [&](std::unique_ptr<OnnxModels::AnalyzerJob> j) { jobs.push_back(std::move(j)); };

  node(512);
  REQUIRE(jobs.size() == 1);
  CHECK(jobs.front()->kind == OnnxModels::AnalyzerJob::Kind::Build);
  CHECK(node.outputs.data.value.empty());
  auto done = OnnxModels::AudioAnalyzer::worker::work(std::move(jobs.front()));
  jobs.pop_front();
  REQUIRE(done);
  done(node);
  for(int i = 0; i < 4; i++)
    node(512);
  CHECK(std::abs(node.outputs.value1.value - 0.25f) < 1e-3f);
  CHECK(jobs.empty());
}
