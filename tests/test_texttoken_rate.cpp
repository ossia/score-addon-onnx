// Text Token Processor TTS output (BUG-LEDGER T1, T2):
//  - T1: Piper's rank-4 [B,1,1,N] waveform was classified as a Data vector,
//    so the audio outlet stayed silent;
//  - T2: the model rate was guessed from port names only and was 22050 for
//    every Piper / Kitten / Kokoro model. It now comes from the "sample_rate"
//    metadata, then the Piper <model>.onnx.json sidecar, then the name guess.
//
// The rate is observed through the audio outlet: the fixtures output 64
// samples, which the node resamples to the 48 kHz host rate.
#include <OnnxModels/TextToken.hpp>

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

struct Harness
{
  OnnxModels::TextToken node;
  std::string name; // the port's filename is a view: keep it alive
  std::string bytes;
  float sample{};
  float* chans[1]{&sample};

  explicit Harness(const std::string& model)
      : name{model}
      , bytes{slurp(model)}
  {
    node.prepare({.rate = 48000., .frames = 1});
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.outputs.audio.samples = chans;
    node.outputs.audio.channels = 1;
    // Run the heavy (TTS) jobs inline, as the task pool would later.
    node.worker.request = [this](std::unique_ptr<OnnxModels::TokenInferJob> job) {
      if(auto done = OnnxModels::TextToken::worker::work(std::move(job)))
        done(node);
    };
  }

  // Synthesise once, then drain one frame per tick and count the samples.
  int synthesise(std::vector<int> ids, int max_ticks = 1 << 20)
  {
    node.inputs.tokens.value = std::move(ids);
    int n = 0;
    for(int t = 0; t < max_ticks; t++)
    {
      sample = 0.f;
      node(1);
      node.inputs.tokens.value.clear();
      if(sample != 0.f)
        n++;
      else if(t > 0)
        break;
    }
    return n;
  }
};
}

TEST_CASE("Token output roles: rank-4 Piper waveform", "[onnx][texttoken]")
{
  using Onnx::TensorElemType;
  using Onnx::TokenOutputRole;
  using Onnx::classifyTokenOutput;
  CHECK(classifyTokenOutput("output", {-1, -1, 1, -1}, TensorElemType::Float)
        == TokenOutputRole::Waveform);
  CHECK(classifyTokenOutput("output", {1, 1, 1, 113152}, TensorElemType::Float)
        == TokenOutputRole::Waveform);
  // Image-like: several channels on axis 1.
  CHECK(classifyTokenOutput("output", {1, 3, 1, 64}, TensorElemType::Float)
        == TokenOutputRole::Vector);
  // Too short to be audio.
  CHECK(classifyTokenOutput("output", {1, 1, 1, 8}, TensorElemType::Float)
        == TokenOutputRole::Vector);
  // Unchanged ranks.
  CHECK(classifyTokenOutput("output", {1, 1, -1}, TensorElemType::Float)
        == TokenOutputRole::Waveform);
  CHECK(classifyTokenOutput("embeds", {1, 512}, TensorElemType::Float)
        == TokenOutputRole::Vector);
}

TEST_CASE("TTS model rate: metadata, then sidecar, then name guess", "[onnx][texttoken]")
{
  const std::string dir = SCORE_ONNX_TEST_DATA_DIR "/tts";

  // 64 samples at 16 kHz -> 192 at 48 kHz.
  Harness sidecar{dir + "/sidecar/tts.onnx"};
  const int s = sidecar.synthesise({1, 2, 3});
  CHECK(std::abs(s - 192) <= 3); // the resampler holds back a sample or two

  // The metadata (24 kHz) wins over the sidecar (16 kHz): 128 samples.
  Harness metadata{dir + "/metadata/tts.onnx"};
  CHECK(std::abs(metadata.synthesise({1, 2, 3}) - 128) <= 3);

  // Neither: the 22050 fallback, 64 * 48000 / 22050 = 139.3.
  Harness none{dir + "/none/tts.onnx"};
  CHECK(std::abs(none.synthesise({1, 2, 3}) - 139) <= 3);
}

TEST_CASE("Piper voice plays on the audio outlet", "[onnx][texttoken]")
{
  const char* env = std::getenv("ONNX_TEST_MODELS");
  const std::string model
      = std::string(env ? env : "/mnt/win2/models/models-presets/models")
        + "/text-token/piper-en_US-amy-low/en_US-amy-low.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);

  Harness piper{model};
  std::vector<int> ids;
  for(int i = 0; i < 40; i++)
    ids.push_back(1 + (i * 7) % 60);
  // Only the first second: the ring's size is a separate issue (T4).
  CHECK(piper.synthesise(ids, 48000) > 1000);
  CHECK(piper.node.outputs.data.value.empty());
}

// BUG-LEDGER T5: the token and int aux tensors were always int64, and ORT
// rejects an int32 / bool input fed int64. The fixture computes
// 2 * inputs + text_lengths + 100 * flag.
TEST_CASE("Int32 and bool inputs get their declared type", "[onnx][texttoken]")
{
  Harness enc{SCORE_ONNX_TEST_DATA_DIR "/tts/int32/encoder.onnx"};
  enc.node.inputs.tokens.value = {5, 7, 11};
  enc.node(1);
  REQUIRE(!enc.node.inputs.model.current_model_invalid);
  const auto& d = enc.node.outputs.data.value;
  REQUIRE(d.size() == 3);
  // text_lengths is the token count; flag takes Param 1 (0.667, rounds to 1).
  CHECK(d == std::vector<float>{2 * 5 + 3 + 100, 2 * 7 + 3 + 100, 2 * 11 + 3 + 100});
}

TEST_CASE("ct-transformer punctuation (int32, async) runs", "[onnx][texttoken]")
{
  const std::string model = "/mnt/sdd1/models/sherpa/text/"
                            "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/"
                            "model.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);

  Harness punct{model};
  punct.node.inputs.tokens.value = {100, 200, 300, 400};
  punct.node(1);
  // [1, 4, 6] logits: six punctuation classes per token.
  CHECK(punct.node.outputs.data.value.size() == 4 * 6);
}

// BUG-LEDGER T6: a per-token int input (BERT's attention_mask, the punctuation
// model's valid_ids) was fed [1,1] holding round(Param 1). BERT failed in a
// reshape; the punctuation model returned zero rows.
namespace
{
void checkNear(const std::vector<float>& got, std::vector<float> want)
{
  REQUIRE(got.size() >= want.size());
  for(std::size_t i = 0; i < want.size(); i++)
    CHECK(std::abs(got[i] - want[i]) < 1e-3f);
}
}

TEST_CASE("BERT: the attention mask follows the tokens", "[onnx][texttoken]")
{
  const std::string model = "/mnt/sdd1/models/wild2/"
                            "network_intrusion_detection__bert-network-packet-flow-"
                            "header-payload__model.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  Harness bert{model};
  bert.node.inputs.tokens.value = {101, 2000, 3000, 4000, 5000, 102};
  bert.node(1);
  REQUIRE(bert.node.outputs.data.value.size() == 24);
  // Python ORT with attention_mask = ones(1, 6).
  checkNear(bert.node.outputs.data.value, {-14.307504f, -9.751713f, -10.221975f, -8.524228f});
}

TEST_CASE("Punctuation: valid_ids and label_lens follow the tokens", "[onnx][texttoken]")
{
  const std::string model
      = "/mnt/sdd1/models/sherpa/text/sherpa-onnx-online-punct-en-2024-08-06/model.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  Harness punct{model};
  punct.node.inputs.tokens.value = {100, 200, 300, 400};
  punct.node(1);
  // Python ORT with valid_ids = ones(1, 4), label_lens = [4]: a (2, 4) result.
  REQUIRE(punct.node.outputs.data.value.size() == 8);
  checkNear(punct.node.outputs.data.value, {0.75716215f, -1.9144665f, 1.7697821f, -2.6571329f});
}

// BUG-LEDGER T3, T4, T8: TTS re-ran on every tick once a job returned, so the
// output restarted continuously; the utterance went through a ring sized for
// one 22050-sample block, so everything but its last ~1.4 s was lost; and a
// change while a job ran was lost for good.
namespace
{
// Jobs wait until complete() is called, as they would on a busy worker.
struct QueuedHarness : Harness
{
  std::vector<std::unique_ptr<OnnxModels::TokenInferJob>> queue;
  int dispatches = 0;
  explicit QueuedHarness(const std::string& model)
      : Harness{model}
  {
    node.worker.request = [this](std::unique_ptr<OnnxModels::TokenInferJob> job) {
      dispatches++;
      queue.push_back(std::move(job));
    };
  }
  std::vector<int64_t> pendingTokens() const { return queue.front()->tokens; }
  void complete()
  {
    auto job = std::move(queue.front());
    queue.erase(queue.begin());
    if(auto done = OnnxModels::TextToken::worker::work(std::move(job)))
      done(node);
  }
};
}

TEST_CASE("TTS runs once per change, not on every tick", "[onnx][texttoken]")
{
  QueuedHarness h{SCORE_ONNX_TEST_DATA_DIR "/tts/sidecar/tts.onnx"};
  h.node.inputs.tokens.value = {1, 2, 3};
  for(int t = 0; t < 50; t++)
  {
    h.node(1);
    if(!h.queue.empty())
      h.complete();
  }
  CHECK(h.dispatches == 1);

  // Reset speaks again.
  h.node.inputs.reset.value.emplace();
  for(int t = 0; t < 50; t++)
  {
    h.node(1);
    if(!h.queue.empty())
      h.complete();
  }
  CHECK(h.dispatches == 2);
}

TEST_CASE("TTS: changes while a job runs coalesce to the latest", "[onnx][texttoken]")
{
  QueuedHarness h{SCORE_ONNX_TEST_DATA_DIR "/tts/sidecar/tts.onnx"};
  h.node.inputs.tokens.value = {1};
  h.node(1);
  REQUIRE(h.dispatches == 1);
  h.node.inputs.tokens.value = {2};
  h.node(1);
  h.node.inputs.tokens.value = {3};
  h.node(1);
  CHECK(h.dispatches == 1); // busy: nothing more yet

  // The superseded result is dropped: nothing plays.
  h.complete();
  h.sample = 1.f;
  h.node.inputs.tokens.value = {3};
  h.node(1);
  CHECK(h.sample == 0.f);
  // ...and the latest ids were dispatched, once.
  REQUIRE(h.dispatches == 2);
  CHECK(h.pendingTokens() == std::vector<int64_t>{3});
  h.complete();
  CHECK(std::abs(h.synthesise({3}) - 192) <= 3); // plays, no new dispatch
  CHECK(h.dispatches == 2);
}

TEST_CASE("Async text encoder: a change while busy is not lost", "[onnx][texttoken]")
{
  const std::string model = "/mnt/sdd1/models/sherpa/text/"
                            "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/"
                            "model.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  QueuedHarness h{model};
  h.node.inputs.tokens.value = {100, 200, 300, 400};
  h.node(1);
  h.node.inputs.tokens.value = {100, 200, 300, 400, 500};
  h.node(1);
  h.complete(); // the 4-token result, superseded
  h.node(1);
  REQUIRE(h.dispatches == 2);
  h.complete();
  CHECK(h.node.outputs.data.value.size() == 5 * 6);
}

TEST_CASE("Piper: a long utterance plays whole", "[onnx][texttoken]")
{
  const char* env = std::getenv("ONNX_TEST_MODELS");
  const std::string model
      = std::string(env ? env : "/mnt/win2/models/models-presets/models")
        + "/text-token/piper-en_US-amy-low/en_US-amy-low.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);

  Harness piper{model};
  std::vector<int> ids;
  for(int i = 0; i < 250; i++)
    ids.push_back(1 + (i * 7) % 60);
  // Count every frame until the player falls silent for good.
  piper.node.inputs.tokens.value = ids;
  int played = 0, last = 0;
  for(int t = 0; t < 48000 * 20; t++)
  {
    piper.sample = 0.f;
    piper.node(1);
    if(piper.sample != 0.f)
    {
      played++;
      last = t;
    }
    else if(t - last > 48000)
      break;
  }
  // These 250 ids give about 6.5 s. The old ring kept the last 1.4 s at
  // 48 kHz (66666 frames).
  CHECK(last > 4 * 48000);
  CHECK(played > 4 * 48000);
}

// BUG-LEDGER T7: autoregressive decoders whose state is not named like a KV
// cache (Tacotron2's decoder_iter) were run anyway, fed garbage.
TEST_CASE("Tacotron2 decoder step is refused", "[onnx][texttoken]")
{
  const std::string model = "/mnt/sdd1/models/wild/tacotron2__decoder_iter.onnx";
  if(!std::filesystem::exists(model))
    SKIP("model not found: " << model);
  QueuedHarness h{model};
  h.node.inputs.tokens.value = {1, 2, 3};
  h.node(1);
  CHECK(!h.node.inputs.model.current_model_invalid);
  CHECK(h.dispatches == 0);
  CHECK(h.node.outputs.data.value.empty());
}

// BUG-LEDGER T9: Kitten / Kokoro take a speaker style vector from voices.bin;
// they got a constant Param instead (Kitten 10x too quiet, Kokoro clipping).
namespace
{
float rmsOf(Harness& h, int max_ticks, float& peak)
{
  double acc = 0;
  int n = 0;
  peak = 0;
  for(int t = 0; t < max_ticks; t++)
  {
    h.sample = 0.f;
    h.node(1);
    h.node.inputs.tokens.value.clear();
    acc += (double)h.sample * h.sample;
    peak = std::max(peak, std::abs(h.sample));
    n++;
  }
  return (float)std::sqrt(acc / std::max(n, 1));
}
std::vector<int> kittenIds()
{
  std::vector<int> ids;
  for(int i = 0; i < 60; i++)
    ids.push_back(16 + (i * 11) % 60);
  return ids;
}
}

TEST_CASE("Kitten: the style comes from voices.bin", "[onnx][texttoken]")
{
  const char* env = std::getenv("ONNX_TEST_MODELS");
  const std::string dir = std::string(env ? env : "/mnt/win2/models/models-presets/models")
                          + "/text-token/kitten-nano-en";
  const std::string model = dir + "/kitten-nano-en-v0_1-fp16.onnx";
  if(!std::filesystem::exists(model) || !std::filesystem::exists(dir + "/voices.bin"))
    SKIP("model or voices not found");

  Harness h{model};
  h.node.inputs.tokens.value = kittenIds();
  float peak = 0;
  const float rms = rmsOf(h, 48000 * 4, peak);
  INFO("rms " << rms << " peak " << peak);
  CHECK(rms > 0.03f);
  CHECK(peak < 1.f);

  // Without any voices, the model is not run (it would only make noise).
  const auto tmp = std::filesystem::temp_directory_path() / "onnx_kitten_novoices";
  std::filesystem::create_directories(tmp);
  const auto copy = (tmp / "kitten.onnx").string();
  std::filesystem::copy_file(model, copy, std::filesystem::copy_options::overwrite_existing);
  QueuedHarness q{copy};
  q.node.inputs.tokens.value = kittenIds();
  q.node(1);
  CHECK(q.dispatches == 0);
  std::filesystem::remove_all(tmp);
}
