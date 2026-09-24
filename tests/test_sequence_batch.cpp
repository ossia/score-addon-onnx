// Sequence Processor: models that cannot run at batch 1 (BUG-LEDGER S5).
// Informer declares a dynamic batch but its graph bakes 2: it failed inside
// infer() on every tick, and one failure disabled the node without a word.
// A zero inference at load now finds the batch the graph runs at, or reports
// why it runs at none.
#include <OnnxModels/SequenceProcessor.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

struct Seq
{
  OnnxModels::SequenceProcessor node;
  std::string name, bytes;
  explicit Seq(const std::string& path)
      : name{path}
      , bytes{slurp(path)}
  {
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.worker.request = [this](std::unique_ptr<OnnxModels::SeqInferJob> job) {
      if(auto done = OnnxModels::SequenceProcessor::worker::work(std::move(job)))
        done(node);
    };
  }
};

const std::string fixtures = SCORE_ONNX_TEST_DATA_DIR "/sequence/";
}

TEST_CASE("Sequence Processor: a graph with a baked batch of 2 runs", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Seq s{fixtures + "batch2.onnx"};
  s.node.inputs.in.value = {1.f, 2.f, 3.f};
  s.node();
  CHECK_FALSE(s.node.inputs.model.current_model_invalid);
  CHECK(s.node.outputs.out.value == std::vector<float>{2.f, 4.f, 6.f});

  // And again: the node keeps running.
  s.node.inputs.in.value = {-1.f, 0.f, 1.f};
  s.node();
  CHECK(s.node.outputs.out.value == std::vector<float>{-2.f, 0.f, 2.f});
}

TEST_CASE("Sequence Processor: a model that runs at no batch is refused at load", "[onnx][sequence]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  Seq s{fixtures + "broken.onnx"};
  s.node.inputs.in.value = {1.f, 2.f, 3.f, 4.f};
  s.node();
  CHECK(s.node.inputs.model.current_model_invalid);
  CHECK(s.node.outputs.out.value.empty());
}

TEST_CASE("Sequence Processor: Informer forecasts after one window", "[onnx][sequence]")
{
  const char* env = std::getenv("ONNX_TEST_WILD_MODELS");
  const std::string model = std::string(env ? env : "/mnt/sdd1/models/wild")
                            + "/informer2020__informer_ETTh1.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  REQUIRE(OnnxModels::initOnnxRuntime());

  // batch_x [?,96,7]: Sliding over 7-value frames, 24 steps of 7 out.
  Seq s{model};
  for(int t = 0; t < 96; t++)
  {
    s.node.inputs.in.value.assign(7, 0.f);
    for(int f = 0; f < 7; f++)
      s.node.inputs.in.value[f] = 0.5f * std::sin(0.2f * (float)t + (float)f);
    s.node();
    REQUIRE_FALSE(s.node.inputs.model.current_model_invalid);
  }
  CHECK(s.node.outputs.out.value.size() == 24 * 7);
}
