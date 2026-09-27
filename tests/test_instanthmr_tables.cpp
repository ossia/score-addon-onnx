// InstantHMR: classification of the HF signature, output
// resolution, and the MHR70 -> standard-skeleton remap tables. Header-only
// (ModelRole.hpp / SkeletonFormats.hpp are ossia/Qt/ORT-free), so this builds
// into the plain unit-test executable: no model file, no onnxruntime.
#include <Onnx/helpers/ModelRole.hpp>
#include <Onnx/helpers/SkeletonFormats.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <set>

using Onnx::ModelIO;
using Onnx::ModelKind;
namespace Skel = Onnx::Skel;

namespace
{
// instanthmr.onnx from huggingface.co/momolesang/InstantHMR, as ORT reports
// it: the symbolic "batch" dim comes back as -1 (GetShape()).
ModelIO instantHmrHF()
{
  ModelIO io;
  io.inputs = {{"image", {-1, 3, 224, 224}}, {"cliff_cond", {-1, 3}}};
  io.outputs
      = {{"mhr_params", {-1, 204}},
         {"shape_params", {-1, 45}},
         {"cam_trans", {-1, 3}},
         {"joints_2d", {-1, 70, 2}},
         {"joints_3d", {-1, 70, 3}}};
  return io;
}
// instanthmr_distill_train/train_distill_mhr_only.py HMRDeployWrapper export:
// same inputs, joints_3d dropped (4 outputs).
ModelIO instantHmrMhrOnly()
{
  ModelIO io = instantHmrHF();
  io.outputs.pop_back();
  return io;
}
}

TEST_CASE("InstantHMR: HF signature classifies", "[onnx][pose][instanthmr]")
{
  const auto r = Onnx::classify(instantHmrHF());
  CHECK(r.kind == ModelKind::InstantHmr);
  CHECK(r.stage == Onnx::ModelStage::Landmark);
  CHECK(r.domain == Onnx::ModelDomain::Body);
  CHECK(r.num_keypoints == 70);
  CHECK(r.num_inputs == 2);
  CHECK_FALSE(r.nhwc);
  CHECK(r.input_w == 224);
  CHECK(r.input_h == 224);

  SECTION("static batch export")
  {
    ModelIO io = instantHmrHF();
    for(auto* ports : {&io.inputs, &io.outputs})
      for(auto& p : *ports)
        p.shape[0] = 1;
    CHECK(Onnx::classify(io).kind == ModelKind::InstantHmr);
  }
  SECTION("renamed second input, still [N,3]")
  {
    ModelIO io = instantHmrHF();
    io.inputs[1].name = "bbox_info";
    CHECK(Onnx::classify(io).kind == ModelKind::InstantHmr);
  }
  SECTION("second input named cliff with another shape")
  {
    ModelIO io = instantHmrHF();
    io.inputs[1].shape = {-1, 4};
    io.inputs[1].name = "cliff";
    CHECK(Onnx::classify(io).kind == ModelKind::InstantHmr);
  }
}

TEST_CASE("InstantHMR: rule does not capture other families", "[onnx][pose][instanthmr]")
{
  SECTION("mhr-only 4-output variant stays Unknown but is recognised")
  {
    const auto io = instantHmrMhrOnly();
    CHECK(Onnx::classify(io).kind == ModelKind::Unknown);
    CHECK(Onnx::isInstantHmrMhrOnly(io));
    CHECK_FALSE(Onnx::isInstantHmrMhrOnly(instantHmrHF()));
    const auto h = Onnx::resolveHmrOutputs(io);
    CHECK(h.mhr_only);
    CHECK_FALSE(h.valid());
    CHECK(h.joints_3d == -1);
    CHECK(h.joints_2d == 3);
    CHECK(h.num_keypoints == 70);
  }
  SECTION("[N,2] second input (PINTO with_post bbox) never matches")
  {
    ModelIO io = instantHmrHF();
    io.inputs[1] = {"bbox", {1, 2}};
    CHECK(Onnx::classify(io).kind != ModelKind::InstantHmr);
  }
  SECTION("single input never matches")
  {
    ModelIO io = instantHmrHF();
    io.inputs.pop_back();
    CHECK(Onnx::classify(io).kind != ModelKind::InstantHmr);
  }
  SECTION("K mismatch / too few / dynamic K never matches")
  {
    ModelIO io = instantHmrHF();
    io.outputs[4].shape = {-1, 69, 3};
    CHECK(Onnx::classify(io).kind != ModelKind::InstantHmr);
    io = instantHmrHF();
    io.outputs[3].shape = {-1, 5, 2};
    io.outputs[4].shape = {-1, 5, 3};
    CHECK(Onnx::classify(io).kind != ModelKind::InstantHmr);
    io = instantHmrHF();
    io.outputs[3].shape = {-1, -1, 2};
    io.outputs[4].shape = {-1, -1, 3};
    CHECK(Onnx::classify(io).kind != ModelKind::InstantHmr);
  }
  SECTION("PINTO RTMPose with_post (image + [1,2] bbox -> [1,K,3])")
  {
    ModelIO io;
    io.inputs = {{"input", {1, 3, 256, 192}}, {"bbox_x1y1x2y2", {1, 2}}};
    io.outputs = {{"kpts_xyscore", {1, 17, 3}}};
    CHECK(Onnx::classify(io).kind == ModelKind::XyScoreLandmark);
  }
  SECTION("SimCC with a second input")
  {
    ModelIO io;
    io.inputs = {{"input", {1, 3, 256, 192}}, {"bbox", {1, 2}}};
    io.outputs = {{"simcc_x", {1, 17, 384}}, {"simcc_y", {1, 17, 512}}};
    CHECK(Onnx::classify(io).kind == ModelKind::SimccPose);
  }
  SECTION("RTMO")
  {
    ModelIO io;
    io.inputs = {{"input", {1, 3, 640, 640}}};
    io.outputs = {{"dets", {1, -1, 5}}, {"keypoints", {1, -1, 17, 3}}};
    CHECK(Onnx::classify(io).kind == ModelKind::RtmoPose);
  }
  SECTION("YOLO-pose")
  {
    ModelIO io;
    io.inputs = {{"images", {1, 3, 640, 640}}};
    io.outputs = {{"output0", {1, 56, 8400}}};
    CHECK(Onnx::classify(io).kind == ModelKind::YoloPose);
  }
  SECTION("RetinaFace")
  {
    ModelIO io;
    io.inputs = {{"input", {1, 3, 640, 640}}};
    io.outputs
        = {{"loc", {1, 16800, 4}}, {"conf", {1, 16800, 2}}, {"landms", {1, 16800, 10}}};
    CHECK(Onnx::classify(io).kind == ModelKind::RetinaFaceDetector);
  }
}

TEST_CASE("InstantHMR: output resolution", "[onnx][pose][instanthmr]")
{
  SECTION("by name (HF order)")
  {
    const auto h = Onnx::resolveHmrOutputs(instantHmrHF());
    CHECK(h.valid());
    CHECK_FALSE(h.mhr_only);
    CHECK(h.mhr_params == 0);
    CHECK(h.shape_params == 1);
    CHECK(h.cam_trans == 2);
    CHECK(h.joints_2d == 3);
    CHECK(h.joints_3d == 4);
    CHECK(h.num_keypoints == 70);
  }
  SECTION("by shape when names are generic, any order")
  {
    ModelIO io;
    io.inputs = instantHmrHF().inputs;
    io.outputs
        = {{"out3", {-1, 70, 3}},
           {"out0", {-1, 3}},
           {"out1", {-1, 45}},
           {"out2", {-1, 70, 2}},
           {"out4", {-1, 204}}};
    const auto h = Onnx::resolveHmrOutputs(io);
    CHECK(h.valid());
    CHECK(h.joints_3d == 0);
    CHECK(h.cam_trans == 1);
    CHECK(h.shape_params == 2);
    CHECK(h.joints_2d == 3);
    CHECK(h.mhr_params == 4);
    CHECK(Onnx::classify(io).kind == ModelKind::InstantHmr);
  }
  SECTION("absent outputs are -1")
  {
    ModelIO io;
    io.inputs = instantHmrHF().inputs;
    io.outputs = {{"joints_2d", {-1, 70, 2}}, {"joints_3d", {-1, 70, 3}}};
    const auto h = Onnx::resolveHmrOutputs(io);
    CHECK(h.cam_trans == -1);
    CHECK(h.mhr_params == -1);
    CHECK(h.shape_params == -1);
    CHECK_FALSE(h.valid()); // no camera placement
  }
}

namespace
{
struct KP
{
  float x, y, z, confidence;
};
std::vector<KP> synthMhr70()
{
  std::vector<KP> v;
  for(int i = 0; i < 70; ++i)
    v.push_back({float(i), 2.f * i, 3.f * i, 1.f});
  return v;
}
}

TEST_CASE("MHR70 remap tables", "[onnx][pose][skeleton][instanthmr]")
{
  using S = Skel::SourceSkeleton;
  using T = Skel::TargetSkeleton;
  const std::pair<T, std::size_t> targets[] = {
      {T::Coco17, 17},  {T::OpenPoseCoco18, 18}, {T::OpenPoseBody25, 25},
      {T::Halpe26, 26}, {T::Mpii16, 16},         {T::H36m17, 17},
      {T::Hand21, 21}};
  for(auto [t, n] : targets)
  {
    const auto table = Skel::mappingFor(S::Mhr70, t);
    INFO("target " << int(t));
    REQUIRE(table.size() == n);
    CHECK(Skel::namesFor(t).size() == n);
    for(const auto& m : table)
    {
      CHECK(m.n > 0); // every target joint is fillable from MHR70
      for(int k = 0; k < m.n; ++k)
      {
        CHECK(m.idx[k] >= 0);
        CHECK(m.idx[k] < 70);
      }
    }
  }
  CHECK(Skel::mappingFor(S::Mhr70, T::Dlib68).empty());
  CHECK(Skel::mappingFor(S::Mhr70, T::Native).empty());

  const auto in = synthMhr70();
  std::vector<KP> out;
  auto run = [&](T t) {
    REQUIRE(Skel::remap<KP>(S::Mhr70, t, std::span<const KP>(in), out));
  };

  run(T::Coco17);
  CHECK(out[9].x == 62.f);  // left wrist
  CHECK(out[10].x == 41.f); // right wrist
  CHECK(out[11].x == 9.f);  // left hip
  for(int i = 0; i < 17; ++i)
    CHECK(out[i].confidence == 1.f);

  run(T::OpenPoseBody25);
  CHECK(out[1].x == 69.f);  // Neck is the real joint
  CHECK(out[8].x == 9.5f);  // MidHip
  CHECK(out[8].y == 19.f);
  CHECK(out[4].x == 41.f);  // RWrist
  CHECK(out[7].x == 62.f);  // LWrist
  CHECK(out[19].x == 15.f); // LBigToe
  CHECK(out[21].x == 17.f); // LHeel
  CHECK(out[24].x == 20.f); // RHeel

  run(T::Halpe26);
  CHECK(out[18].x == 69.f); // Neck
  CHECK(out[19].x == 9.5f); // Hip
  CHECK(out[22].x == 16.f); // LSmallToe
  CHECK(out[23].x == 19.f); // RSmallToe

  run(T::H36m17);
  CHECK(out[8].x == 69.f);                     // Thorax
  CHECK(out[7].x == (69.f + 69 + 9 + 10) / 4); // Spine
  CHECK(out[16].x == 41.f);                    // RWrist

  run(T::Hand21);
  CHECK(out[0].x == 41.f); // right wrist
  // Thumb CMC..TIP = third, second, first, tip = 24, 23, 22, 21.
  CHECK(out[1].x == 24.f);
  CHECK(out[4].x == 21.f);
  // Index MCP = third_joint (28), TIP = 25; pinky TIP = 37.
  CHECK(out[5].x == 28.f);
  CHECK(out[8].x == 25.f);
  CHECK(out[20].x == 37.f);
}

TEST_CASE("MHR70 native edges and names", "[onnx][pose][skeleton][instanthmr]")
{
  using S = Skel::SourceSkeleton;
  const auto edges = Skel::nativeEdges(S::Mhr70);
  const auto names = Skel::nativeNames(S::Mhr70);
  REQUIRE(names.size() == 70);
  CHECK(std::strcmp(names[41], "right_wrist") == 0);
  CHECK(std::strcmp(names[62], "left_wrist") == 0);
  CHECK(std::strcmp(names[69], "neck") == 0);
  REQUIRE(edges.size() == std::size_t(Skel::kMhr70BodyBones + 2 * Skel::kMhr70HandBones));
  std::set<int> touched;
  for(const auto& b : edges)
  {
    CHECK(b.a >= 0);
    CHECK(b.a < 70);
    CHECK(b.b >= 0);
    CHECK(b.b < 70);
    CHECK(b.a != b.b);
    CHECK_FALSE((b.a >= 63 && b.a <= 68));
    CHECK_FALSE((b.b >= 63 && b.b <= 68));
    touched.insert(b.a);
    touched.insert(b.b);
  }
  // Every joint except the 63-68 surface landmarks is drawn.
  for(int i = 0; i < 70; ++i)
    CHECK(touched.contains(i) == !(i >= 63 && i <= 68));
  // Hand blocks: each finger chain starts at its wrist.
  for(int f = 0; f < 5; ++f)
  {
    CHECK(edges[Skel::kMhr70BodyBones + 4 * f].a == 41);
    CHECK(edges[Skel::kMhr70BodyBones + Skel::kMhr70HandBones + 4 * f].a == 62);
  }
  CHECK(Skel::nativeEdges(S::Coco17).size() == Skel::edgesFor(Skel::TargetSkeleton::Coco17).size());
  CHECK(Skel::nativeNames(S::Mhr70).size() == 70);
}
