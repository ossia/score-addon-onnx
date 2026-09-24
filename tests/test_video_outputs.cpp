// Video Processor output routing (BUG-LEDGER V1): only the Output Index output
// was decoded, so RVM gave fgr or pha, never both. The other outputs now go to
// the outlets the primary one left free, and the recurrent states to none.
#include <OnnxModels/VideoProcessor.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace
{
std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

struct Harness
{
  OnnxModels::VideoProcessor node;
  QImage image;
  std::string name, bytes;

  Harness(const std::string& img, const std::string& model, int output_index)
      : image{QImage(QString::fromStdString(img)).convertToFormat(QImage::Format_RGBA8888)}
      , name{model}
      , bytes{slurp(model)}
  {
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.inputs.output_index.value = output_index;
    node.worker.request = [this](std::unique_ptr<OnnxModels::VideoInferJob> job) {
      if(auto done = OnnxModels::VideoProcessor::worker::work(std::move(job)))
        done(node);
    };
  }

  void run()
  {
    auto& t = node.inputs.image.texture;
    t.bytes = image.bits();
    t.width = image.width();
    t.height = image.height();
    t.changed = true;
    node();
  }
};

bool anyNonZero(const unsigned char* p, std::size_t n)
{
  return p && std::any_of(p, p + n, [](unsigned char c) { return c != 0; });
}
}

TEST_CASE("RVM: foreground and alpha from one node", "[onnx][video]")
{
  const char* env = std::getenv("ONNX_TEST_MODELS");
  const std::string model
      = std::string(env ? env : "/mnt/win2/models/models-presets/models")
        + "/video-processor/rvm_mobilenetv3_fp32.onnx";
  const std::string img
      = "/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/body.jpg";
  if(!std::filesystem::exists(model) || !std::filesystem::exists(img))
    SKIP("model or image not found");

  // Output Index 0 (fgr) and 1 (pha) both fill Image and Mask.
  for(int primary : {0, 1})
  {
    DYNAMIC_SECTION("Output Index " << primary)
    {
      Harness h{img, model, primary};
      h.run();
      auto& im = h.node.outputs.image.texture;
      auto& mask = h.node.outputs.mask.texture;
      REQUIRE(im.changed);
      REQUIRE(mask.changed);
      CHECK(im.width == mask.width);
      CHECK(im.height == mask.height);
      CHECK(anyNonZero(
          reinterpret_cast<const unsigned char*>(im.bytes),
          (std::size_t)im.width * im.height * 4));
      CHECK(anyNonZero(
          reinterpret_cast<const unsigned char*>(mask.bytes),
          (std::size_t)mask.width * mask.height));
      // The recurrent states r1o..r4o feed the next frame, no outlet.
      CHECK(h.node.outputs.depth.texture.bytes == nullptr);
      CHECK(h.node.outputs.data.value.empty());

      // The chain still runs on the next frame.
      im.changed = mask.changed = false;
      h.run();
      CHECK(im.changed);
      CHECK(mask.changed);
    }
  }
}

namespace
{
std::string rvmModel()
{
  const char* env = std::getenv("ONNX_TEST_MODELS");
  return std::string(env ? env : "/mnt/win2/models/models-presets/models")
         + "/video-processor/rvm_mobilenetv3_fp32.onnx";
}
const std::string bodyImage
    = "/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/body.jpg";

// Jobs wait until complete() is called, as on a busy worker.
struct QueuedVideo : Harness
{
  std::vector<std::unique_ptr<OnnxModels::VideoInferJob>> queue;
  int dispatches = 0;
  QueuedVideo(const std::string& img, const std::string& model)
      : Harness{img, model, 1}
  {
    node.worker.request = [this](std::unique_ptr<OnnxModels::VideoInferJob> job) {
      dispatches++;
      queue.push_back(std::move(job));
    };
  }
  void complete()
  {
    auto job = std::move(queue.front());
    queue.erase(queue.begin());
    if(auto done = OnnxModels::VideoProcessor::worker::work(std::move(job)))
      done(node);
  }
  std::vector<unsigned char> mask() const
  {
    auto& t = node.outputs.mask.texture;
    return {t.bytes, t.bytes + (std::size_t)t.width * t.height};
  }
};
}

// BUG-LEDGER V3: Reset was ignored whenever a job was in flight, which at HD
// is nearly always, and the job's result brought the old recurrent state back.
TEST_CASE("RVM: Reset during a job restarts the recurrence", "[onnx][video]")
{
  if(!std::filesystem::exists(rvmModel()) || !std::filesystem::exists(bodyImage))
    SKIP("model or image not found");

  // Above 512x512: runs on the worker.
  auto setup = [](QueuedVideo& v) { v.node.inputs.resolution.value = {768, 512}; };

  QueuedVideo fresh{bodyImage, rvmModel()};
  setup(fresh);
  fresh.run();
  REQUIRE(fresh.dispatches == 1);
  fresh.complete();
  const auto first = fresh.mask();

  QueuedVideo v{bodyImage, rvmModel()};
  setup(v);
  v.run();
  v.complete(); // frame 1
  v.run();      // frame 2 in flight
  v.node.inputs.reset.value.emplace();
  v.run(); // busy: the reset waits
  v.node.inputs.reset.value.reset(); // an impulse lasts one tick
  v.complete();
  v.run(); // zeroes the states, then dispatches frame 3
  REQUIRE(v.dispatches == 3);
  v.complete();
  CHECK(v.mask() == first);
}

// BUG-LEDGER V4: the heavy test is static (file and pixel size), so RVM at
// 512 px ran on the render thread however long it took.
TEST_CASE("RVM: a slow synchronous run moves the model to the worker", "[onnx][video]")
{
  if(!std::filesystem::exists(rvmModel()) || !std::filesystem::exists(bodyImage))
    SKIP("model or image not found");
  QueuedVideo v{bodyImage, rvmModel()};
  v.node.inputs.resolution.value = {512, 512};
  v.node.inputs.param1.value = 1.0f; // downsample ratio 1: the "fast" preset
  const auto t0 = std::chrono::steady_clock::now();
  v.run();
  const auto first = std::chrono::steady_clock::now() - t0;
  CHECK(v.dispatches == 0); // measured synchronously once
  if(first < std::chrono::milliseconds(8))
    SKIP("this machine runs RVM at 512 px within the budget");
  v.run();
  CHECK(v.dispatches == 1);
}
