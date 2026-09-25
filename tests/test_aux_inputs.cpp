// Inputs a node does not own (Onnx/helpers/AuxInputs.hpp; BUG-LEDGER S2, S3,
// G2, IG1, AA4, T6): the planning rules on real signatures, and the Sequence
// Processor feeding Silero's sr and a scalar control.
#include <tests/TestPaths.hpp>
#include <OnnxModels/AudioAnalyzer.hpp>
#include <OnnxModels/GeometryProcessor.hpp>
#include <OnnxModels/ImageGenerator.hpp>
#include <OnnxModels/SequenceProcessor.hpp>

#include <Onnx/helpers/AuxInputs.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <random>
#include <string>
#include <vector>

using Onnx::AuxFill;
using Onnx::TensorElemType;
using Port = Onnx::ModelSpec::Port;

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

const Onnx::AuxPlan* planFor(const std::vector<Onnx::AuxPlan>& plans, int index)
{
  for(auto& p : plans)
    if(p.index == index)
      return &p;
  return nullptr;
}
}

TEST_CASE("Aux planning rules on real signatures", "[onnx][aux]")
{
  SECTION("Silero v4: sr gets the rate, h/c are owned")
  {
    std::vector<Port> in{
        {"input", {-1, -1}, TensorElemType::Float},
        {"sr", {}, TensorElemType::Int64},
        {"h", {2, -1, 64}, TensorElemType::Float},
        {"c", {2, -1, 64}, TensorElemType::Float}};
    const int owned[]{0, 2, 3};
    auto plans = Onnx::planAuxInputs(in, owned);
    REQUIRE(plans.size() == 1);
    CHECK(plans[0].index == 1);
    CHECK(plans[0].fill == AuxFill::SampleRate);
  }
  SECTION("CLAP: bool `longer` is false")
  {
    std::vector<Port> in{
        {"mel_fusion", {1, 4, 1001, 64}, TensorElemType::Float},
        {"longer", {1, 1}, TensorElemType::Bool}};
    const int owned[]{0};
    auto plans = Onnx::planAuxInputs(in, owned);
    REQUIRE(plans.size() == 1);
    CHECK(plans[0].fill == AuxFill::Zeros);
  }
  SECTION("BERT: the attention mask follows the tokens, token types are zeros")
  {
    std::vector<Port> in{
        {"input_ids", {-1, -1}, TensorElemType::Int64},
        {"attention_mask", {-1, -1}, TensorElemType::Int64},
        {"token_type_ids", {-1, -1}, TensorElemType::Int64}};
    const int owned[]{0};
    auto plans = Onnx::planAuxInputs(in, owned);
    REQUIRE(plans.size() == 2);
    CHECK(planFor(plans, 1)->fill == AuxFill::Ones);
    CHECK(planFor(plans, 1)->link);
    CHECK(planFor(plans, 2)->fill == AuxFill::Zeros);
  }
  SECTION("punctuation: valid_ids ones, label_lens the length")
  {
    std::vector<Port> in{
        {"token_ids", {-1, -1}, TensorElemType::Int32},
        {"valid_ids", {-1, -1}, TensorElemType::Int32},
        {"label_lens", {-1}, TensorElemType::Int32}};
    const int owned[]{0};
    auto plans = Onnx::planAuxInputs(in, owned);
    CHECK(planFor(plans, 1)->fill == AuxFill::Ones);
    CHECK(planFor(plans, 2)->fill == AuxFill::LinkedLength);
  }
  SECTION("EigenGAN: the z_* latents get noise only on generator nodes")
  {
    std::vector<Port> in{{"eps", {1, 512}, TensorElemType::Float}};
    for(int k = 0; k < 6; k++)
      in.push_back({"z_" + std::to_string(k), {1, 6}, TensorElemType::Float});
    const int owned[]{0};
    auto gen = Onnx::planAuxInputs(in, owned, {.seeded_noise = true});
    REQUIRE(gen.size() == 6);
    for(auto& p : gen)
      CHECK(p.fill == AuxFill::SeededNoise);
    auto plain = Onnx::planAuxInputs(in, owned);
    for(auto& p : plain)
      CHECK(p.fill == AuxFill::Zeros);
  }
  SECTION("scalars take Param 1 then Param 2, then zeros")
  {
    std::vector<Port> in{
        {"points", {1, -1, 3}, TensorElemType::Float},
        {"t", {1}, TensorElemType::Float},
        {"k", {}, TensorElemType::Int64},
        {"u", {1, 1}, TensorElemType::Float}};
    const int owned[]{0};
    auto plans = Onnx::planAuxInputs(in, owned);
    CHECK(planFor(plans, 1)->param_index == 0);
    CHECK(planFor(plans, 2)->param_index == 1);
    CHECK(planFor(plans, 3)->fill == AuxFill::Zeros);
  }
  SECTION("informer's time marks and decoder input stay zeros")
  {
    std::vector<Port> in{
        {"batch_x", {-1, 96, 7}, TensorElemType::Float},
        {"batch_x_mark", {-1, 96, 4}, TensorElemType::Float},
        {"dec_inp", {-1, 72, 7}, TensorElemType::Float},
        {"batch_y_mark", {-1, 72, 4}, TensorElemType::Float}};
    const int owned[]{0};
    for(auto& p : Onnx::planAuxInputs(in, owned))
      CHECK(p.fill == AuxFill::Zeros);
  }
  SECTION("sr is a whole word")
  {
    std::vector<Port> in{
        {"x", {1, 16}, TensorElemType::Float},
        {"sru_state", {1, 1}, TensorElemType::Float},
        {"input_sr", {1}, TensorElemType::Int64}};
    const int owned[]{0};
    auto plans = Onnx::planAuxInputs(in, owned);
    CHECK(planFor(plans, 1)->fill == AuxFill::Param);
    CHECK(planFor(plans, 2)->fill == AuxFill::SampleRate);
  }
}

TEST_CASE("Aux filling: typed storage and linked shapes", "[onnx][aux]")
{
  REQUIRE(OnnxModels::initOnnxRuntime()); // no node here does it for us
  const float params[]{2.6f, 0.f};
  const int64_t tokens[]{1, 6};
  Onnx::AuxHost host{.params = params, .sample_rate = 16000., .primary_shape = tokens};
  std::vector<uint8_t> store;
  std::vector<int64_t> shape;

  Onnx::AuxPlan mask{.index = 1, .dt = TensorElemType::Int64, .shape = {-1, -1},
                     .fill = AuxFill::Ones, .link = true};
  auto m = Onnx::fillAux(mask, host, store, shape);
  CHECK(shape == std::vector<int64_t>{1, 6});
  CHECK(m.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  for(int i = 0; i < 6; i++)
    CHECK(m.GetTensorData<int64_t>()[i] == 1);

  Onnx::AuxPlan sr{.index = 1, .dt = TensorElemType::Int64, .shape = {},
                   .fill = AuxFill::SampleRate};
  auto r = Onnx::fillAux(sr, host, store, shape);
  CHECK(shape.empty()); // rank 0 stays rank 0
  CHECK(r.GetTensorData<int64_t>()[0] == 16000);

  Onnx::AuxPlan flag{.index = 2, .dt = TensorElemType::Bool, .shape = {1, 1},
                     .fill = AuxFill::Param, .param_index = 0};
  auto b = Onnx::fillAux(flag, host, store, shape);
  CHECK(b.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
  CHECK(b.GetTensorData<bool>()[0]);

  Onnx::AuxPlan i32{.index = 3, .dt = TensorElemType::Int32, .shape = {1},
                    .fill = AuxFill::Param, .param_index = 0};
  CHECK(Onnx::fillAux(i32, host, store, shape).GetTensorData<int32_t>()[0] == 3);

  Onnx::AuxPlan len{.index = 4, .dt = TensorElemType::Int32, .shape = {-1},
                    .fill = AuxFill::LinkedLength};
  CHECK(Onnx::fillAux(len, host, store, shape).GetTensorData<int32_t>()[0] == 6);

  // Seeded noise is reproducible per (seed, input index).
  Onnx::AuxPlan z{.index = 5, .dt = TensorElemType::Float, .shape = {1, 6},
                  .fill = AuxFill::SeededNoise};
  host.seed = 42;
  std::vector<uint8_t> s1, s2;
  std::vector<int64_t> sh1, sh2;
  auto a = Onnx::fillAux(z, host, s1, sh1);
  auto c = Onnx::fillAux(z, host, s2, sh2);
  CHECK(s1 == s2);
  CHECK(a.GetTensorData<float>()[0] != 0.f);
}

TEST_CASE("Sequence Processor: Silero gets sr = 16000", "[onnx][aux][sequence]")
{
  std::string model = TestPaths::wild() + "/wild/silero-vad__silero_vad.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  const auto bytes = slurp(model);

  // The diagnosis repro's speech-like signal: a 220 Hz tone, 3 Hz AM, noise.
  // Python ORT gives a mean probability of 0.79 with sr=16000, 0.004 with 0.
  std::vector<float> sig(16000);
  std::mt19937 rng(0);
  std::normal_distribution<float> nd;
  for(int i = 0; i < 16000; i++)
  {
    const double t = i / 16000.;
    sig[i] = float(
        0.3 * std::sin(2 * std::numbers::pi * 220 * t)
            * (1 + std::sin(2 * std::numbers::pi * 3 * t))
        + 0.02 * nd(rng));
  }

  OnnxModels::SequenceProcessor node;
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;
  double sum = 0;
  int n = 0;
  for(int k = 0, b = 0; k + 512 <= 16000; k += 512, b++)
  {
    node.inputs.in.value.assign(sig.begin() + k, sig.begin() + k + 512);
    node();
    REQUIRE(!node.inputs.model.current_model_invalid);
    REQUIRE(!node.outputs.out.value.empty());
    if(b >= 5)
    {
      sum += node.outputs.out.value[0];
      n++;
    }
  }
  CHECK(sum / n > 0.5);
}

TEST_CASE("Sequence Processor: Param 1 drives a scalar input", "[onnx][aux][sequence]")
{
  const std::string model = SCORE_ONNX_TEST_DATA_DIR "/aux/scale.onnx";
  const auto bytes = slurp(model);
  OnnxModels::SequenceProcessor node;
  node.inputs.model.file.bytes = bytes;
  node.inputs.model.file.filename = model;
  node.inputs.param1.value = 2.5f;
  node.inputs.in.value = {1, 2, 3, 4};
  node();
  CHECK(node.outputs.out.value == std::vector<float>{2.5f, 5.f, 7.5f, 10.f});
}

// BUG-LEDGER G2: only input 0 was bound, so a second input made ORT refuse the
// run, and a cloud declared second went in under the other input's name.
TEST_CASE("Geometry Processor: every input bound, Param 1 drives a scalar", "[onnx][aux][geometry]")
{
  for(const char* name : {"geom_2in", "geom_cloud_second"})
  {
    DYNAMIC_SECTION(name)
    {
      const std::string model = std::string(SCORE_ONNX_TEST_DATA_DIR "/aux/") + name + ".onnx";
      const auto bytes = slurp(model);
      OnnxModels::GeometryProcessor node;
      node.inputs.model.file.bytes = bytes;
      node.inputs.model.file.filename = model;
      node.inputs.normalization.value = OnnxModels::PointNormalization::None;
      node.inputs.param1.value = 0.5f;
      node.inputs.cloud.value = {0, 0, 0, 1, 2, 3};
      node();
      REQUIRE(!node.inputs.model.current_model_invalid);
      CHECK(node.outputs.cloud.value == std::vector<float>{0.5f, 0.5f, 0.5f, 1.5f, 2.5f, 3.5f});
    }
  }
}

// BUG-LEDGER IG1: only input 0 was fed, so ORT refused EigenGAN (eps + six
// z_* latents) and the node never produced an image.
namespace
{
struct GenHarness
{
  OnnxModels::ImageGenerator node;
  std::string name, bytes;
  explicit GenHarness(const std::string& model)
      : name{model}
      , bytes{slurp(model)}
  {
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.worker.request = [this](std::unique_ptr<OnnxModels::GenJob> job) {
      if(auto done = OnnxModels::ImageGenerator::worker::work(std::move(job)))
        done(node);
    };
  }
};
}

TEST_CASE("Image Generator: every input of a multi-input generator is fed", "[onnx][aux][generator]")
{
  const std::string eigengan
      = TestPaths::pintoZoo() + "/161_EigenGAN-Tensorflow/saved_model_Anime/model_float32.onnx";
  if(!std::filesystem::exists(eigengan))
    SKIP("model not found: " << eigengan);

  GenHarness g{eigengan};
  g.node();
  REQUIRE(!g.node.inputs.model.current_model_invalid);
  auto& tex = g.node.outputs.image.texture;
  REQUIRE(tex.bytes);
  CHECK(tex.width == 256);
  CHECK(tex.height == 256);
  std::vector<unsigned char> first(tex.bytes, tex.bytes + 256 * 256 * 4);

  // Another seed draws other z_* latents as well as another eps.
  g.node.inputs.seed.value = 7;
  g.node();
  REQUIRE(tex.bytes);
  CHECK(std::vector<unsigned char>(tex.bytes, tex.bytes + 256 * 256 * 4) != first);
}

TEST_CASE("Image Generator: single-input StyleGAN2 unchanged", "[onnx][aux][generator]")
{
  const std::string model = TestPaths::models()
                            + "/image-generator/stylegan2-ffhq-1024.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  GenHarness g{model};
  g.node();
  REQUIRE(g.node.outputs.image.texture.bytes);
  CHECK(g.node.outputs.image.texture.width == 1024);
}

// BUG-LEDGER AA4: a bool aux input was fed as float, which ORT refuses, and a
// rank-4 spectrogram input failed on every block.
namespace
{
struct AnalyzerHarness
{
  OnnxModels::AudioAnalyzer node;
  std::string name, bytes;
  std::vector<float> block;
  float* chans[1]{};

  explicit AnalyzerHarness(const std::string& model)
      : name{model}
      , bytes{slurp(model)}
      , block(512, 0.25f)
  {
    node.prepare({.rate = 16000., .input_channels = 1, .frames = 512});
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    chans[0] = block.data();
    node.inputs.audio.samples = chans;
    node.inputs.audio.channels = 1;
    node.worker.request = [this](std::unique_ptr<OnnxModels::AnalyzerJob> job) {
      if(auto done = OnnxModels::AudioAnalyzer::worker::work(std::move(job)))
        done(node);
    };
  }
};
}

TEST_CASE("Audio Analyzer: a bool input gets a bool", "[onnx][aux][analyzer]")
{
  AnalyzerHarness a{SCORE_ONNX_TEST_DATA_DIR "/aux/audio_flag.onnx"};
  for(int i = 0; i < 8; i++)
    a.node(512);
  REQUIRE(!a.node.inputs.model.current_model_invalid);
  // mean(0.25) + 100 * false
  CHECK(std::abs(a.node.outputs.value1.value - 0.25f) < 1e-3f);
}

TEST_CASE("Audio Analyzer: CLAP's rank-4 input is refused, not retried", "[onnx][aux][analyzer]")
{
  const std::string clap
      = TestPaths::wild() + "/wild2/audio_processing__clap__CLAP_audio_LAION-Audio-630K_with_fusion.onnx";
  if(!std::filesystem::exists(clap))
    SKIP("model not found: " << clap);
  AnalyzerHarness a{clap};
  for(int i = 0; i < 8; i++)
    a.node(512);
  // Refused once at load (disabled until another file), not run and failing
  // on every block.
  CHECK(a.node.inputs.model.current_model_invalid);
  CHECK(a.node.outputs.data.value.empty());
}

TEST_CASE("Audio Analyzer: Silero still gets its rate", "[onnx][aux][analyzer]")
{
  const std::string model = TestPaths::models()
                            + "/audio-analyzer/silero-vad-v4-16k.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);

  AnalyzerHarness a{model};
  std::mt19937 rng(0);
  std::normal_distribution<float> nd;
  double sum = 0;
  int n = 0;
  for(int k = 0; k < 32; k++)
  {
    for(int i = 0; i < 512; i++)
    {
      const double t = (k * 512 + i) / 16000.;
      a.block[i] = float(
          0.3 * std::sin(2 * std::numbers::pi * 220 * t)
              * (1 + std::sin(2 * std::numbers::pi * 3 * t))
          + 0.02 * nd(rng));
    }
    a.node(512);
    REQUIRE(!a.node.inputs.model.current_model_invalid);
    if(k >= 8)
    {
      sum += a.node.outputs.value1.value;
      n++;
    }
  }
  CHECK(sum / n > 0.5);
}

// BUG-LEDGER IG2: a mapping net's w [1,512] could not chain into a w+
// synthesis [1,18,512]; the node returned silently every tick.
TEST_CASE("Image Generator: mapping w chains into a w+ synthesis", "[onnx][generator]")
{
  CHECK(OnnxModels::ImageGenerator::chainCompatible({1, 512}, {1, 18, 512}));
  CHECK(OnnxModels::ImageGenerator::chainCompatible({1, 512}, {1, 512}));
  CHECK(!OnnxModels::ImageGenerator::chainCompatible({1, 512}, {1, 17, 500}));

  const std::string dir = TestPaths::models();
  const std::string synth = dir + "/image-generator/e4e-ffhq-decoder-wplus.onnx";
  const std::string map = dir + "/sequence-processor/mobilestylegan-ffhq-mapping.onnx";
  if(!std::filesystem::exists(synth) || !std::filesystem::exists(map))
    SKIP("models not found");
  GenHarness g{synth};
  const auto map_bytes = slurp(map);
  g.node.inputs.mapping_model.file.bytes = map_bytes;
  g.node.inputs.mapping_model.file.filename = map;
  g.node();
  REQUIRE(!g.node.inputs.model.current_model_invalid);
  REQUIRE(g.node.outputs.image.texture.bytes);
  CHECK(g.node.outputs.image.texture.width > 0);
}

// BUG-LEDGER IG3: Auto was a per-frame min/max stretch; StyleGAN2 is a tanh
// generator, for which Denormalize is right.
TEST_CASE("Image Generator: Auto picks Denormalize for a [-1,1] generator", "[onnx][generator]")
{
  const std::string model = TestPaths::models()
                            + "/image-generator/stylegan2-ffhq-1024.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  GenHarness a{model}, d{model};
  d.node.inputs.output_mode.value = OnnxModels::GenOutputMode::Denormalize;
  a.node();
  d.node();
  auto& ta = a.node.outputs.image.texture;
  auto& td = d.node.outputs.image.texture;
  REQUIRE(ta.bytes);
  REQUIRE(td.bytes);
  const std::size_t n = (std::size_t)ta.width * ta.height * 4;
  CHECK(std::equal(ta.bytes, ta.bytes + n, td.bytes));
}
