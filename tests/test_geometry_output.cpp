// Output routing of the Geometry Processor in Task=Auto (BUG-LEDGER G1): the
// input layout detector (xyz + up to 13 features) was reused for outputs, so
// PointNet part-segmentation logits and its 3x3 transform went to the cloud.
#include <Onnx/helpers/GeometryIO.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace Onnx;

TEST_CASE("geometry output: coordinates are a cloud", "[onnx][geometry]")
{
  CHECK(classifyGeomOutput({1, 2048, 3}) == GeomOutputKind::PointCloud);
  CHECK(classifyGeomOutput({1, 3, 2048}) == GeomOutputKind::PointCloud);
  CHECK(classifyGeomOutput({1, -1, 3}) == GeomOutputKind::PointCloud);
  CHECK(classifyGeomOutput({1, 8192, 3}) == GeomOutputKind::PointCloud);
}

TEST_CASE("geometry output: per-point logits and transforms are data", "[onnx][geometry]")
{
  CHECK(classifyGeomOutput({1, 2048, 4}) == GeomOutputKind::Data);  // PointNet airplane pred
  CHECK(classifyGeomOutput({1, 2048, 13}) == GeomOutputKind::Data);
  CHECK(classifyGeomOutput({1, 2048, 50}) == GeomOutputKind::Data); // ShapeNet parts
  CHECK(classifyGeomOutput({1, 3, 3}) == GeomOutputKind::Data);     // PointNet T-Net
  CHECK(classifyGeomOutput({1, 40}) == GeomOutputKind::Data);       // classification
}

TEST_CASE("geometry output: same features in and out is a cloud", "[onnx][geometry]")
{
  // xyz + normals in, xyz + normals out (e.g. a denoiser)
  CHECK(classifyGeomOutput({1, 2048, 6}, 6) == GeomOutputKind::PointCloud);
  CHECK(classifyGeomOutput({1, 2048, 6}, 3) == GeomOutputKind::Data);
  CHECK(classifyGeomOutput({1, 2048, 6}) == GeomOutputKind::Data);
}
