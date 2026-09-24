// Image Processor output decoding (BUG-LEDGER V2, I2, I4): the pixel mappings
// and the 2-class segmentation outputs.
#include <OnnxModels/ImageProcessor.hpp>

#include <Onnx/helpers/TensorToTexture.hpp>

#include <QImage>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using Onnx::WriteMode;

namespace
{
std::vector<uint8_t> mask(const std::vector<float>& plane, std::vector<int64_t> shape, WriteMode m)
{
  const auto s = Onnx::makeOutSpec(shape);
  std::vector<uint8_t> out((std::size_t)s.w * s.h);
  Onnx::writeMask(plane.data(), s, m, out.data());
  return out;
}

std::string slurp(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

struct ImageHarness
{
  OnnxModels::ImageProcessor node;
  QImage image;
  std::string name, bytes;
  ImageHarness(const std::string& img, const std::string& model)
      : image{QImage(QString::fromStdString(img)).convertToFormat(QImage::Format_RGBA8888)}
      , name{model}
      , bytes{slurp(model)}
  {
    node.inputs.model.file.bytes = bytes;
    node.inputs.model.file.filename = name;
    node.worker.request = [this](std::unique_ptr<OnnxModels::InferJob> job) {
      if(auto done = OnnxModels::ImageProcessor::worker::work(std::move(job)))
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
  // Mask value at a point given as a fraction of the mask's size.
  int maskAt(float x, float y) const
  {
    auto& t = node.outputs.mask.texture;
    return t.bytes[(int)(y * t.height) * t.width + (int)(x * t.width)];
  }
};
}

// BUG-LEDGER V2: Auto stretched every mask to its min..max, so a nearly empty
// alpha matte (max 0.3) became full white specks.
TEST_CASE("Auto mask mapping keeps a [0,1] matte as it is", "[onnx][image]")
{
  std::vector<float> weak(16, 0.f);
  weak[5] = 0.3f;
  CHECK(mask(weak, {1, 1, 4, 4}, WriteMode::AutoRange)[5] == 76);
  CHECK(mask(weak, {1, 1, 4, 4}, WriteMode::MinMaxNormalize)[5] == 255);
  // Logits are still stretched.
  std::vector<float> logits(16, -5.f);
  logits[3] = 5.f;
  auto l = mask(logits, {1, 1, 4, 4}, WriteMode::AutoRange);
  CHECK(l[3] == 255);
  CHECK(l[0] == 0);
}

// BUG-LEDGER I4: DexiNed's fused output is a logit map; min-max left its
// background grey.
TEST_CASE("Sigmoid mapping", "[onnx][image]")
{
  std::vector<float> v{-8.f, 0.f, 8.f, 0.f};
  auto m = mask(v, {1, 1, 2, 2}, WriteMode::Sigmoid);
  CHECK(m[0] == 0);
  CHECK(m[1] == 127);
  CHECK(m[2] == 254);
}

// BUG-LEDGER I2: Mask read channel 0, the background of a 2-class output, so
// the mask came out inverted; an NHWC [1,H,W,2] output was read as W=2.
TEST_CASE("2-class outputs give the foreground", "[onnx][image]")
{
  // NCHW probabilities [1,2,1,5]: background plane, then foreground.
  std::vector<float> probs{0.9f, 0.2f, 0.5f, 0.5f, 0.5f, 0.1f, 0.8f, 0.5f, 0.5f, 0.5f};
  auto p = mask(probs, {1, 2, 1, 5}, WriteMode::DirectClamp);
  CHECK(p[0] == 25);
  CHECK(p[1] == 204);
  // NCHW logits: softmax of the pair.
  std::vector<float> logits{15.f, -15.f, 0.f, 0.f, 0.f, -15.f, 15.f, 0.f, 0.f, 0.f};
  auto l = mask(logits, {1, 2, 1, 5}, WriteMode::DirectClamp);
  CHECK(l[0] == 0);
  CHECK(l[1] == 255);
  // NHWC.
  const auto s = Onnx::makeOutSpec({1, 512, 512, 2});
  CHECK(s.nhwc);
  CHECK(s.channels == 2);
  CHECK(s.w == 512);
  // NHWC [1,5,1,2]: (background, foreground) per pixel.
  std::vector<float> nhwc{0.9f, 0.1f, 0.2f, 0.8f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  auto n = mask(nhwc, {1, 5, 1, 2}, WriteMode::DirectClamp);
  CHECK(n[0] == 25);
  CHECK(n[1] == 204);
}

TEST_CASE("PP-HumanSeg: the person is the mask", "[onnx][image]")
{
  const std::string model = "/mnt/sdd1/models/pinto/196__human_segmentation_pphumanseg_2021oct.onnx";
  const std::string img = "/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/body.jpg";
  if(!std::filesystem::exists(model) || !std::filesystem::exists(img))
    SKIP("model or image not found");
  ImageHarness h{img, model};
  h.run();
  REQUIRE(!h.node.inputs.model.current_model_invalid);
  REQUIRE(h.node.outputs.mask.texture.bytes);
  CHECK(h.maskAt(0.34f, 0.45f) > 128); // the player on the left
  CHECK(h.maskAt(0.03f, 0.9f) < 128);  // the grass
}

TEST_CASE("Hand segmentation: the hand is the mask", "[onnx][image]")
{
  const std::string model = "/mnt/sdd1/models/wild2/"
                            "hand_recognition__hands_segmentation_pytorch__hands_segmentation_pytorch.onnx";
  const std::string img
      = "/home/jcelerier/projets/oss/ailia-models/hand_recognition/blazehand/person_hand.jpg";
  if(!std::filesystem::exists(model) || !std::filesystem::exists(img))
    SKIP("model or image not found");
  ImageHarness h{img, model};
  h.node.inputs.resolution.value = {256, 256};
  // This export wants ImageNet-normalised input (a preset setting).
  h.node.inputs.normalization.value = OnnxModels::InputNormalization::ImageNet;
  h.run();
  REQUIRE(!h.node.inputs.model.current_model_invalid);
  // Fully symbolic output: it used to decode as an RGB picture.
  REQUIRE(h.node.outputs.mask.texture.bytes);
  CHECK(h.maskAt(0.42f, 0.6f) > 128); // the palm
  CHECK(h.maskAt(0.95f, 0.05f) < 128);
}


// BUG-LEDGER I1: only 1 or 3 input channels were built, so a 4-channel model
// (RGB + prior mask) threw and was disabled.
TEST_CASE("Hair segmenter: a 4-channel input gets a zero 4th channel", "[onnx][image]")
{
  const std::string model = "/mnt/sdd1/models/pinto/060__hair_segmenter.onnx";
  const std::string img
      = "/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/face.png";
  if(!std::filesystem::exists(model) || !std::filesystem::exists(img))
    SKIP("model or image not found");
  ImageHarness h{img, model};
  h.run();
  REQUIRE(!h.node.inputs.model.current_model_invalid);
  auto& t = h.node.outputs.mask.texture;
  REQUIRE(t.bytes);
  CHECK(t.width == 512);
  CHECK(t.height == 512);
  const auto* b = reinterpret_cast<const unsigned char*>(t.bytes);
  CHECK(std::any_of(b, b + 512 * 512, [](unsigned char c) { return c > 128; }));
}

// BUG-LEDGER I3: FILM declares its time [?,1] before the two frames, so it was
// classified from `time` as a latent generator and never got its images.
TEST_CASE("FILM: frame interpolation runs with time on Param 1", "[onnx][image]")
{
  const std::string model = "/mnt/sdd1/models/wild/film__film_net.onnx";
  const std::string img
      = "/mnt/win2/models/libreonnx/ossia-detection-model-pack/test_images/face.png";
  if(!std::filesystem::exists(model) || !std::filesystem::exists(img))
    SKIP("model or image not found");
  ImageHarness h{img, model};
  h.image = h.image.scaled(256, 256);
  h.node.inputs.resolution.value = {256, 256};
  h.node.inputs.resize_mode.value = OnnxModels::ResizeMode::Stretch;
  h.node.inputs.output_index.value = 24; // "image"
  h.node.inputs.param1.value = 0.5f;
  h.run(); // Aux unconnected: both frames are In
  REQUIRE(!h.node.inputs.model.current_model_invalid);
  auto& t = h.node.outputs.image.texture;
  REQUIRE(t.bytes);
  REQUIRE(t.width == 256);
  // Between two identical frames, the frame itself.
  const auto* out = reinterpret_cast<const unsigned char*>(t.bytes);
  const auto* in = h.image.constBits();
  double err = 0;
  for(int i = 0; i < 256 * 256 * 4; i += 4)
    for(int c = 0; c < 3; c++)
      err += std::abs(int(out[i + c]) - int(in[i + c]));
  CHECK(err / (256 * 256 * 3) < 10.);
}
