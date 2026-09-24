// Sequence Processor with a job in flight (BUG-LEDGER S6) and the choice of
// its data input (S7).
#include <OnnxModels/SequenceProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
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

const std::string fixtures = SCORE_ONNX_TEST_DATA_DIR "/sequence/";

// Jobs wait until complete(), as on a busy worker.
struct Deferred
{
  OnnxModels::SequenceProcessor node;
  std::string name, bytes;
  std::unique_ptr<OnnxModels::SeqInferJob> held;
  int requests = 0;

  explicit Deferred(const std::string& path) { load(path); }
  void load(const std::string& path)
  {
    name = path;
    bytes = slurp(path);
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.worker.request = [this](std::unique_ptr<OnnxModels::SeqInferJob> job) {
      ++requests;
      held = std::move(job);
    };
  }
  void complete()
  {
    REQUIRE(held);
    if(auto done = OnnxModels::SequenceProcessor::worker::work(std::move(held)))
      done(node);
  }
  float state() const { return node.states.at(0).values.at(0); }
  // More than 1 << 20 values: the worker path.
  void tick(bool payload = true)
  {
    node.inputs.in.value.assign(payload ? (1 << 20) + 8 : 0, 1.f);
    node();
  }
};
}

// S6: Reset zeroed the recurrent state, but the job in flight then wrote its
// own state back, and its output with it.
TEST_CASE("Sequence Processor: Reset during a job restarts the recurrence", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Deferred s{fixtures + "counter.onnx"};
  s.tick();
  s.complete();
  REQUIRE(s.state() == 1.f);
  REQUIRE(s.node.outputs.out.value.at(0) == 1.f); // x + 0

  s.tick(); // in flight with state 1
  s.node.inputs.reset.value.emplace();
  s.tick(false);
  s.node.inputs.reset.value.reset();
  REQUIRE(s.state() == 0.f);
  s.complete();
  CHECK(s.state() == 0.f);
  CHECK(s.node.outputs.out.value.at(0) == 1.f); // not the stale 2

  s.tick();
  s.complete();
  CHECK(s.node.outputs.out.value.at(0) == 1.f); // restarted from 0
  CHECK(s.state() == 1.f);
}

// S6, same cause: a job of the previous model finished after a swap.
TEST_CASE("Sequence Processor: a job of the old model is dropped after a swap", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  const auto copy = std::filesystem::temp_directory_path() / "score_onnx_counter_copy.onnx";
  std::filesystem::copy_file(
      fixtures + "counter.onnx", copy, std::filesystem::copy_options::overwrite_existing);

  Deferred s{fixtures + "counter.onnx"};
  s.tick();
  s.complete();
  s.tick(); // in flight, state 1
  s.load(copy.string());
  s.tick(false); // reloads: state 0
  REQUIRE(s.state() == 0.f);
  s.complete();
  CHECK(s.state() == 0.f);
  std::filesystem::remove(copy);
}

// S7: the first input that was not a recurrent state became the data input,
// even a scalar control: the probe passed, and every frame then failed.
TEST_CASE("Sequence Processor: a scalar declared first stays a control", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::SequenceProcessor node;
  const std::string path = fixtures + "scalar_first.onnx";
  const auto bytes = slurp(path);
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = path;
  node.inputs.param1.value = 3.f;
  node.inputs.in.value = {1.f, 2.f, 3.f, 4.f};
  node();
  CHECK_FALSE(node.inputs.model.current_model_invalid);
  CHECK(node.outputs.out.value == std::vector<float>{3.f, 6.f, 9.f, 12.f});
}
