// Recurrent-state pairing done once by classifyModel (BUG-LEDGER AA1). The
// nodes used to each pair by first same-shape match, so silero's h and c both
// read hn. Signatures are those of the real exports.
#include <Onnx/helpers/ModelArchetype.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>

using namespace Onnx;
using DT = TensorElemType;

namespace
{
int pairOf(const ModelArchetype& m, const std::string& in)
{
  for(const auto& p : m.inputs)
    if(p.name == in)
      return p.state_pair;
  return -2;
}
std::string outName(const ModelArchetype& m, int o)
{
  return o >= 0 && o < (int)m.outputs.size() ? m.outputs[o].name : "<none>";
}
std::string pairedOutput(const ModelArchetype& m, const std::string& in)
{
  return outName(m, pairOf(m, in));
}
}

TEST_CASE("state pairing: silero v4 (h -> hn, c -> cn)", "[onnx][archetype][state]")
{
  ArchIO io;
  io.inputs = {{"input", {-1, -1}, DT::Float}, {"sr", {}, DT::Int64},
               {"h", {2, -1, 64}, DT::Float}, {"c", {2, -1, 64}, DT::Float}};
  io.outputs = {{"output", {-1, 1}, DT::Float}, {"hn", {2, -1, 64}, DT::Float},
                {"cn", {2, -1, 64}, DT::Float}};
  const auto m = classifyModel(io);
  CHECK(m.stateful);
  CHECK(pairedOutput(m, "h") == "hn");
  CHECK(pairedOutput(m, "c") == "cn");
  CHECK(m.outputs[0].state_pair == -1);
}

TEST_CASE("state pairing: silero sherpa export (h -> new_h, c -> new_c)", "[onnx][archetype][state]")
{
  ArchIO io;
  io.inputs = {{"x", {1, 512}, DT::Float}, {"h", {2, 1, 64}, DT::Float},
               {"c", {2, 1, 64}, DT::Float}};
  io.outputs = {{"prob", {-1, 1}, DT::Float}, {"new_h", {2, -1, 64}, DT::Float},
                {"new_c", {2, -1, 64}, DT::Float}};
  const auto m = classifyModel(io);
  CHECK(m.stateful);
  CHECK(pairedOutput(m, "h") == "new_h");
  CHECK(pairedOutput(m, "c") == "new_c");
}

TEST_CASE("state pairing: output order does not steal the name partner", "[onnx][archetype][state]")
{
  ArchIO io;
  io.inputs = {{"input", {1, 512}, DT::Float}, {"h", {2, 1, 64}, DT::Float},
               {"c", {2, 1, 64}, DT::Float}};
  io.outputs = {{"output", {1, 1}, DT::Float}, {"cn", {2, 1, 64}, DT::Float},
                {"hn", {2, 1, 64}, DT::Float}};
  const auto m = classifyModel(io);
  CHECK(pairedOutput(m, "h") == "hn");
  CHECK(pairedOutput(m, "c") == "cn");
}

TEST_CASE("state pairing: dtln2 (shape pairing with a dynamic batch)", "[onnx][archetype][state]")
{
  ArchIO io;
  io.inputs = {{"input_4", {1, 1, 512}, DT::Float}, {"input_5", {1, 2, 128, 2}, DT::Float}};
  io.outputs = {{"conv1d_3", {-1, 1, 512}, DT::Float},
                {"tf_op_layer_stack_5", {-1, 2, 128, 2}, DT::Float}};
  const auto m = classifyModel(io);
  CHECK(m.stateful);
  CHECK(pairedOutput(m, "input_5") == "tf_op_layer_stack_5");
  CHECK(pairOf(m, "input_4") == -1); // the primary input is never state
  CHECK(m.outputs[0].state_pair == -1);
}

TEST_CASE("state pairing: RVM (symbolic r#i -> r#o by name)", "[onnx][archetype][state]")
{
  ArchIO io;
  io.inputs = {{"src", {-1, 3, -1, -1}, DT::Float},  {"r1i", {-1, -1, -1, -1}, DT::Float},
               {"r2i", {-1, -1, -1, -1}, DT::Float}, {"r3i", {-1, -1, -1, -1}, DT::Float},
               {"r4i", {-1, -1, -1, -1}, DT::Float}, {"downsample_ratio", {1}, DT::Float}};
  io.outputs = {{"fgr", {-1, 3, -1, -1}, DT::Float}, {"pha", {-1, 1, -1, -1}, DT::Float},
                {"r1o", {-1, 16, -1, -1}, DT::Float}, {"r2o", {-1, 20, -1, -1}, DT::Float},
                {"r3o", {-1, 40, -1, -1}, DT::Float}, {"r4o", {-1, 64, -1, -1}, DT::Float}};
  const auto m = classifyModel(io);
  CHECK(m.stateful);
  CHECK(m.suggested == NodeKind::VideoProcessor);
  CHECK(pairedOutput(m, "r1i") == "r1o");
  CHECK(pairedOutput(m, "r2i") == "r2o");
  CHECK(pairedOutput(m, "r3i") == "r3o");
  CHECK(pairedOutput(m, "r4i") == "r4o");
  CHECK(m.outputs[0].state_pair == -1); // fgr
  CHECK(m.outputs[1].state_pair == -1); // pha
  CHECK(m.outputs[0].arch != PortArchetype::RecurrentState);
}

TEST_CASE("state pairing: a same-shape autoencoder is not stateful", "[onnx][archetype][state]")
{
  ArchIO io;
  io.inputs = {{"x", {1, 96, 7}, DT::Float}};
  io.outputs = {{"y", {1, 96, 7}, DT::Float}};
  const auto m = classifyModel(io);
  CHECK(!m.stateful);
  CHECK(m.inputs[0].state_pair == -1);
  CHECK(m.outputs[0].state_pair == -1);
}
