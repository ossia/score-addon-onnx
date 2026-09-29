// The Language Model and the Vision Language Model load their models with
// OnnxModels::ModelLoader: files that cannot be loaded are asked for once,
// however many ticks follow, and the failure is shown on the Response.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/FastVLM.hpp>
#include <OnnxModels/QwenLLM.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

namespace
{
const std::string notAModel = SCORE_ONNX_TEST_DATA_DIR "/audio/make_fixtures.py";

template <typename Port>
void pick(Port& port, const std::string& bytes)
{
  port.file.bytes = bytes;
  port.file.filename = notAModel;
}

template <typename Node, typename Job>
int loads(Node& node, QueuedWorker<Node, Job>& w)
{
  int n = 0;
  for(int i = 0; i < 5; i++)
  {
    node();
    for(auto& j : w.jobs)
      n += isLoadJob(*j);
    w.drain(node);
  }
  return n;
}
}

TEST_CASE("Language Model: files that cannot load are asked for once", "[onnx][llm][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::QwenLLMNode node;
  QueuedWorker<OnnxModels::QwenLLMNode, OnnxModels::LlmJob> w;
  w.attach(node);
  const auto bytes = TestPaths::slurp(notAModel);
  pick(node.inputs.model, bytes);
  pick(node.inputs.tokenizer, bytes);
  CHECK(loads(node, w) == 1);
  CHECK(node.outputs.response.value.starts_with("Cannot load the model"));
}

TEST_CASE("Vision Language Model: files that cannot load are asked for once", "[onnx][vlm][loads]")
{
  REQUIRE(OnnxModels::initOnnxRuntime());
  OnnxModels::FastVLMNode node;
  QueuedWorker<OnnxModels::FastVLMNode, OnnxModels::VlmJob> w;
  w.attach(node);
  const auto bytes = TestPaths::slurp(notAModel);
  pick(node.inputs.visionEncoder, bytes);
  pick(node.inputs.embedTokens, bytes);
  pick(node.inputs.decoder, bytes);
  pick(node.inputs.tokenizer, bytes);
  CHECK(loads(node, w) == 1);
  CHECK(node.outputs.response.value.starts_with("Cannot load the model"));
}
