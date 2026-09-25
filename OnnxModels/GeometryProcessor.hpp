#pragma once
#include <OnnxModels/Utils.hpp>

#include <Onnx/helpers/AuxInputs.hpp>
#include <Onnx/helpers/GeometryIO.hpp>
#include <Onnx/helpers/ModelArchetype.hpp>
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/TensorType.hpp>

#include <boost/container/vector.hpp>
#include <halp/controls.hpp>
#include <halp/meta.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Onnx
{
struct OnnxRunContext;
}

namespace OnnxModels
{
struct GeometryProcessor;

// Off-thread inference job (mirrors ImageProcessor::InferJob): a packed input
// cloud plus everything work() needs to run + decode without touching the node.
// ctx is shared so it outlives the node frame; readModelSpec() is re-read inside
// work() for thread-stable name pointers.
//
// Jobs travel through score::TaskPool's 128-byte smallfun queue, so they are
// heap-allocated and recycled through the lock-free JobPool (only the
// unique_ptr crosses the queue; recycled vectors keep their capacity, so the
// steady-state request path does not allocate). See OnnxModels/JobPool.hpp.
// Override the auto-detected output routing (cloud vs data).
enum class GeomTaskMode
{
  Auto, // classifyGeomOutput() decides from the selected output shape
  PointCloud,
  Data,
};

struct GeomInferJob
{
  uint32_t gen = 0; // the node's generation at dispatch
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  boost::container::vector<float> input; // packed tensor (NPC or NCP)
  std::vector<int64_t> ishape;
  Onnx::TensorElemType in_dt = Onnx::TensorElemType::Float;
  Onnx::PointLayout in_layout;
  int output_index = 0;
  GeomTaskMode task = GeomTaskMode::Auto; // user output-routing override
  int cloud_in = 0;              // model input the packed cloud goes to
  std::vector<Onnx::AuxPlan> aux; // the other inputs
  float params[2]{};
};

// How the input xyz payload is normalized before inference. Many point models
// expect a unit-sphere / unit-cube normalized cloud (PointNet centers + scales
// to the unit sphere); raw clouds otherwise pass through.
enum class PointNormalization
{
  None,       // pass coordinates through unchanged
  UnitSphere, // center on centroid, scale so max radius == 1 (PointNet default)
  UnitCube,   // center on bbox center, scale longest bbox edge to [-0.5, 0.5]
};

// Generic ONNX point-set processor. I/O is expressed as flat interleaved-xyz
// value payloads (+ an explicit point count) rather than a first-class halp
// geometry/mesh port: the geometry port in halp models a renderable GPU mesh
// (positions/normals/indices bound to a draw call), whereas point-set ML I/O is
// a raw [N,3] coordinate set with optional per-point features and no topology.
// A flat val_port<vector<float>> + count is the real-time-safe, lossless carrier
// for that and round-trips cleanly through pack/unpack. (See docs/geometry-node-PLAN.md.)
struct GeometryProcessor : OnnxObject
{
public:
  halp_meta(name, "Geometry Processor");
  halp_meta(c_name, "geometry_processor");
  halp_meta(category, "AI/Geometry");
  halp_meta(author, "ossia team");
  halp_meta(
      description,
      "Generic ONNX point-cloud / point-set processor: runs PointNet-style "
      "classification & segmentation, point-cloud completion / upsampling, "
      "implicit/SDF query-point models and depth->cloud models. Auto-detects "
      "the point layout ([1,N,3] vs [1,3,N]), per-point feature channels and "
      "element type. Input/output clouds are interleaved-xyz float payloads with "
      "an explicit point count; labels / segmentation / field values come out on "
      "the Data port.");
  halp_meta(uuid, "b2c3d4e5-f6a7-4890-b1c2-d3e4f5a6b7c8");

  struct
  {
    // Interleaved xyz(+features) coordinates, point_stride floats per point.
    halp::val_port<"In", std::vector<float>> cloud;
    halp::spinbox_i32<"Point Count", halp::range{0, 1 << 22, 0}> point_count;
    halp::spinbox_i32<"Point Stride", halp::range{3, 32, 3}> point_stride;

    ModelPort<"Model"> model;

    halp::enum_t<PointNormalization, "Normalization"> normalization;
    halp::enum_t<GeomTaskMode, "Task"> task;
    halp::spinbox_i32<"Output Index", halp::range{0, 16, 0}> output_index;

    // Two generic float params forwarded as extra scalar model inputs when the
    // model declares additional scalar inputs (e.g. SDF level, time, scale).
    halp::hslider_f32<"Param 1", halp::range{-10., 10., 0.}> param1;
    halp::hslider_f32<"Param 2", halp::range{-10., 10., 1.}> param2;
  } inputs;

  struct
  {
    // Output cloud (interleaved xyz(+features)) + its point count, for
    // completion / upsampling / depth->cloud models.
    halp::val_port<"Out", std::vector<float>> cloud;
    halp::val_port<"Out Count", int> out_count;
    // Labels / per-point segmentation / SDF or field values.
    halp::val_port<"Data", std::vector<float>> data;
  } outputs;

  GeometryProcessor() noexcept;
  ~GeometryProcessor();

  void operator()();

  // Worker-thread inference (avendish pattern): heavy point models (PointNet++
  // grouping, completion) run off the render thread so a slow inference can't
  // freeze the graph. Cheap models stay inline.
  struct worker
  {
    std::function<void(std::unique_ptr<GeomInferJob>)> request;
    static std::function<void(GeometryProcessor&)>
    work(std::unique_ptr<GeomInferJob> job);
  } worker;

private:
  std::shared_ptr<Onnx::OnnxRunContext> ctx;
  Onnx::ModelSpec spec;
  Onnx::ModelArchetype arch;
  Onnx::PointLayout in_layout;
  int cloudIn = 0;                // model input the packed cloud goes to
  std::vector<Onnx::AuxPlan> aux; // the other inputs: scalars take Param 1 / 2
  std::string lastModelPath;
  int lastOutputIndex = -1;
  bool inferenceInProgress = false;
  // Bumped by a model reload: a job of the previous model finishing later
  // must not publish its result (nor, for Auto, its mapping).
  uint32_t gen = 0;

  // Preprocessing buffers, reused across frames (grow-only).
  std::vector<float> normalized;          // interleaved xyz(+f) after normalize
  boost::container::vector<float> packed; // packed tensor in model layout
  std::vector<uint16_t> half_buf;         // fp16-input staging
  std::vector<float> out_scratch;         // output dtype->float scratch

  void reloadModel();
  void runCloud();
  void dispatchInfer(int64_t npoints, bool force_async);
};

}
