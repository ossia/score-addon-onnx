// Tokenizer node: text -> the token ids of a Hugging Face tokenizer.json, for
// the Text Token Processor (CLIP's text encoder), whose Text field is unused.
#include <tests/TestPaths.hpp>
#include <tests/TestWorker.hpp>
#include <OnnxModels/TextToken.hpp>
#include <OnnxModels/Tokenizer.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
const std::string clipDir = TestPaths::models() + "/text-token/clip-vit-b32";
const std::string clipJson = clipDir + "/tokenizer.json";

std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

struct Harness
{
  OnnxModels::TokenizerNode node;
  std::deque<std::unique_ptr<OnnxModels::TokenizeJob>> jobs;
  Harness()
  {
    node.worker.request = [this](std::unique_ptr<OnnxModels::TokenizeJob> job) {
      jobs.push_back(std::move(job));
    };
  }
  // Runs the queued jobs, as score's thread pool would.
  void complete()
  {
    while(!jobs.empty())
    {
      auto job = std::move(jobs.front());
      jobs.pop_front();
      if(auto done = OnnxModels::TokenizerNode::worker::work(std::move(job)))
        done(node);
    }
  }
  std::vector<int> tokenize(const std::string& text)
  {
    node.inputs.text.value = text;
    node();
    complete();
    return node.outputs.tokens.value;
  }
};
}

// A lone tokenizer.json, no tokenizer_config.json: the class is guessed.
TEST_CASE("Tokenizer: CLIP ids match Hugging Face's", "[onnx][tokenizer]")
{
  if(!std::filesystem::exists(clipJson))
    SKIP("CLIP tokenizer not found: " << clipJson);
  REQUIRE_FALSE(std::filesystem::exists(clipDir + "/tokenizer_config.json"));
  Onnx::TextTokenizer tok{clipJson};
  CHECK(tok.tokenizerClass() == "CLIPTokenizer");
  // CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32").
  const std::vector<int> dog{49406, 320, 1125, 539, 320, 1929, 49407};
  CHECK(tok.encode("a photo of a dog", true) == dog);
  CHECK(tok.encode("A Photo of a DOG", true) == dog); // CLIP lower-cases
  CHECK(tok.encode("a photo of a dog", false)
        == std::vector<int>(dog.begin() + 1, dog.end() - 1));
}

TEST_CASE("Tokenizer: a snapshot's tokenizer_config.json is used", "[onnx][tokenizer]")
{
  const auto dir = TestPaths::models() + "/language-model/Qwen3-0.6B";
  if(!std::filesystem::exists(dir + "/tokenizer_config.json"))
    SKIP("Qwen3 not found");
  Onnx::TextTokenizer tok{dir + "/tokenizer.json"};
  CHECK(tok.tokenizerClass() == "Qwen2Tokenizer");
  CHECK(tok.encode("Hello world", false) == std::vector<int>{9707, 1879});
}

TEST_CASE("Tokenizer: the class of a lone tokenizer.json", "[onnx][tokenizer]")
{
  using T = Onnx::TextTokenizer;
  CHECK(T::guessClass(R"({"added_tokens":[{"id":1,"content":"<|startoftext|>"}],"model":{"type":"BPE"}})")
        == "CLIPTokenizer");
  CHECK(T::guessClass(R"({"model":{"type":"Unigram"}})") == "T5Tokenizer");
  CHECK(T::guessClass(R"({"model":{"type":"BPE"}})") == "GPT2Tokenizer");
  CHECK(T::guessClass("not json") == "GPT2Tokenizer");
}

TEST_CASE("Tokenizer node: text in, ids out, on the worker", "[onnx][tokenizer]")
{
  if(!std::filesystem::exists(clipJson))
    SKIP("CLIP tokenizer not found");
  Harness h;
  h.node.inputs.tokenizer.file.filename = clipJson;

  // Nothing runs on the processing thread: the ids come from the worker.
  h.node.inputs.text.value = "a photo of a dog";
  h.node();
  CHECK(h.node.outputs.tokens.value.empty());
  REQUIRE(h.jobs.size() == 1);
  h.complete();
  CHECK(h.node.outputs.tokens.value.size() == 7);

  // An unchanged input asks for nothing.
  h.node();
  CHECK(h.jobs.empty());

  // A new text reuses the loaded tokenizer.
  CHECK(h.tokenize("a photo of a cat")
        == std::vector<int>{49406, 320, 1125, 539, 320, 2368, 49407});

  h.node.inputs.special.value = false;
  CHECK(h.tokenize("a photo of a cat") == std::vector<int>{320, 1125, 539, 320, 2368});
}

TEST_CASE("Tokenizer node: another file replaces the tokenizer", "[onnx][tokenizer]")
{
  const auto qwen = TestPaths::models() + "/language-model/Qwen3-0.6B/tokenizer.json";
  if(!std::filesystem::exists(clipJson) || !std::filesystem::exists(qwen))
    SKIP("CLIP or Qwen3 tokenizer not found");
  Harness h;
  h.node.inputs.tokenizer.file.filename = clipJson;
  h.node.inputs.special.value = false;
  CHECK(h.tokenize("Hello world") == std::vector<int>{3306, 1002});

  // The old tokenizer goes back to the worker in a job of its own, its last
  // owner, so it is not freed on the processing thread.
  h.node.inputs.tokenizer.file.filename = qwen;
  h.node();
  REQUIRE(h.jobs.size() == 1);
  auto job = std::move(h.jobs.front());
  h.jobs.pop_front();
  if(auto done = OnnxModels::TokenizerNode::worker::work(std::move(job)))
    done(h.node);
  CHECK(h.node.outputs.tokens.value == std::vector<int>{9707, 1879});
  REQUIRE(h.jobs.size() == 1);
  CHECK(h.jobs.front()->path.empty());
  REQUIRE(h.jobs.front()->dispose);
  CHECK(h.jobs.front()->dispose->path() == clipJson);
  CHECK(h.jobs.front()->dispose.use_count() == 1);
  h.complete();
}

TEST_CASE("Tokenizer node: a file it cannot use gives no ids", "[onnx][tokenizer]")
{
  Harness h;
  h.node.inputs.tokenizer.file.filename = SCORE_ONNX_TEST_DATA_DIR "/audio/make_fixtures.py";
  CHECK(h.tokenize("hello").empty());
  // Not retried until an input changes.
  h.node();
  CHECK(h.jobs.empty());
}

// Tokenizer -> Text Token Processor -> CLIP embedding: related texts land
// closer together than unrelated ones.
TEST_CASE("Tokenizer feeds CLIP's text encoder", "[onnx][tokenizer]")
{
  const auto model = clipDir + "/clip-vit-b32-encode-text.onnx";
  if(!std::filesystem::exists(model) || !std::filesystem::exists(clipJson))
    SKIP("CLIP not found");
  REQUIRE(OnnxModels::initOnnxRuntime());

  Harness tok;
  tok.node.inputs.tokenizer.file.filename = clipJson;

  const auto bytes = slurp(model);
  OnnxModels::TextToken enc;
  std::vector<float> out(64);
  float* outs[1]{out.data()};
  enc.prepare({.rate = 48000., .output_channels = 1, .frames = 64});
  enc.inputs.model.file.bytes = bytes;
  enc.inputs.model.file.filename = model;
  enc.outputs.audio.samples = outs;
  enc.outputs.audio.channels = 1;
  inlineWorker(enc);

  auto embed = [&](const std::string& text) {
    enc.inputs.tokens.value = tok.tokenize(text);
    for(int i = 0; i < 4; i++)
      enc(64);
    auto e = enc.outputs.data.value;
    REQUIRE(e.size() == 512);
    return e;
  };
  auto cosine = [](const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0., aa = 0., bb = 0.;
    for(std::size_t i = 0; i < a.size(); i++)
    {
      ab += a[i] * b[i];
      aa += a[i] * a[i];
      bb += b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb);
  };
  const auto dog = embed("a photo of a dog");
  const auto puppy = embed("a picture of a puppy");
  const auto sheet = embed("a spreadsheet of quarterly tax figures");
  INFO("dog/puppy " << cosine(dog, puppy) << ", dog/spreadsheet " << cosine(dog, sheet));
  CHECK(cosine(dog, puppy) > cosine(dog, sheet) + 0.1);
}
