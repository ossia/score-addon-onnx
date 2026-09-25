// Sequence Processor on the processing thread (BUG-LEDGER S8): a new model
// built its session and ran the load probe there, every tick copied the
// payload and the states into fresh vectors and staged the inference in new
// ones, and a Window change dropped the ring so the next frames reallocated
// it. The model now loads (and is probed) in a worker job; the synchronous
// path reuses the pipeline's buffers.
#include <OnnxModels/SequenceProcessor.hpp>
#include <tests/AllocCounter.hpp>
#include <tests/TestWorker.hpp>

#include <catch2/catch_test_macros.hpp>

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
using Job = OnnxModels::SeqInferJob;
const std::string counter = SCORE_ONNX_TEST_DATA_DIR "/sequence/counter.onnx";
}
TEST_CASE("Sequence Processor: the model loads and is probed on the worker", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::SequenceProcessor node;
  const auto bytes = slurp(counter);
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = counter;
  std::deque<std::unique_ptr<Job>> jobs;
  node.worker.request = [&](std::unique_ptr<Job> j) { jobs.push_back(std::move(j)); };

  node.inputs.in.value = {1.f, 2.f};
  node();
  REQUIRE(jobs.size() == 1);
  CHECK(jobs.front()->kind == Job::Kind::Build);
  CHECK(node.outputs.out.value.empty()); // nothing runs before the pipeline
  node();
  CHECK(jobs.size() == 1); // not requested twice

  auto done = OnnxModels::SequenceProcessor::worker::work(std::move(jobs.front()));
  jobs.pop_front();
  REQUIRE(done);
  done(node);
  node();
  CHECK(node.outputs.out.value == std::vector<float>{1.f, 2.f}); // x + state 0
  CHECK(jobs.empty()); // a small model: run inline
}

TEST_CASE("Sequence Processor: a synchronous tick allocates no buffer", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::SequenceProcessor node;
  inlineWorker(node);
  const auto bytes = slurp(counter);
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = counter;
  std::vector<float> payload(512, 1.f); // 2 KB, run inline
  for(int i = 0; i < 5; i++) // load, then size every buffer
  {
    node.inputs.in.value = payload;
    node();
  }
  REQUIRE(node.outputs.out.value.size() == 512);
  int allocs = 0;
  for(int i = 0; i < 20; i++)
  {
    node.inputs.in.value = payload; // the upstream's message, not the node's
    AllocCounter::begin(1024);
    node();
    allocs += AllocCounter::end();
  }
  CHECK(allocs == 0);
}
