#pragma once
// Per-frame mesh kernels for the PoseDetector's body mesh (MHR, see
// MhrModel.hpp): placing a rig-local mesh in camera space, flattening it into
// the Geometry outlets' vertex-array layouts, and a tiny z-buffered software
// rasterizer for the "Draw Mesh" overlay.
//
// These tight loops over every vertex / face / pixel live in the -O3 helper
// lib (SCORE_ONNX_IMAGEOPS_SRC in CMakeLists.txt) next to ImageOps, because the
// node's own translation units run at -O0/-Og in a developer build.
//
// Conventions: positions are packed xyz floats. "Camera" space is the OpenCV /
// InstantHMR vision frame (metres, X right, Y down, Z forward, camera at the
// origin); "OpenGL" space is the same frame rotated 180 deg about X (X right,
// Y up, Z backward: y -> -y, z -> -z), which is what score's 3D objects (Y-up
// cameras) expect. A rotation keeps the handedness, so faces that are
// counter-clockwise seen from outside stay so in both spaces.
//
// Real-time contract: nothing here allocates except Raster::begin() when the
// frame size grows, and Raster::draw() the first time a mesh larger than any
// before it is drawn (its per-vertex projection scratch).
#include <cstdint>
#include <span>
#include <vector>

namespace Onnx::MeshOps
{
// out[i] = local[i] + origin (per xyz triple). out.size() == local.size();
// out may be local (in place).
void translate(
    std::span<const float> local, const float origin[3], std::span<float> out) noexcept;

// Unique vertices, V*3 floats, optionally converted to OpenGL space.
// Returns the end of what was written.
float* writeVertices(std::span<const float> verts, bool opengl, float* out) noexcept;

// Unindexed triangle soup, face by face in `faces` order, corners in the
// face's (counter-clockwise) order: 9 floats per face (xyz of the 3 corners).
// Returns the end of what was written.
float* writeTriangles(
    std::span<const float> verts, std::span<const int32_t> faces, bool opengl,
    float* out) noexcept;

// Z-buffered, back-face culled, flat-shaded software rasterizer: every mesh of
// a frame goes into one depth + colour layer (so people occlude each other
// correctly), which composite() then blends onto the output image once. The
// buffers are members, reused frame to frame; only the dirty rectangle is
// cleared and composited, so a small person on a large frame costs little.
struct Raster
{
  // Start a frame of size w x h (allocates only when the size grows).
  void begin(int w, int h);
  // Draw one mesh, camera space (OpenCV, metres), pinhole camera with focal
  // f (px) and principal point (cx, cy). rgb: 0..255. Faces with a corner
  // closer than 5 cm (or behind the camera) are skipped. Shading: Lambert
  // with a headlight, so a face seen edge-on is darker than one facing the
  // camera; the flat colour of each face keeps the body's form readable.
  void draw(
      std::span<const float> verts, std::span<const int32_t> faces, float f,
      float cx, float cy, const uint8_t rgb[3]);
  // Blend the drawn layer onto an RGBA8888 image (the size given to begin())
  // with opacity `alpha` in [0,1].
  void composite(uint8_t* rgba, int stride_bytes, float alpha) const noexcept;
  bool empty() const noexcept { return x1 <= x0 || y1 <= y0; }

  std::vector<float> depth;    // 1/z per pixel, 0 = empty
  std::vector<uint32_t> color; // packed 0x00BBGGRR per pixel
  std::vector<float> screen;   // per vertex: u, v, 1/z (scratch)
  // Per row, the columns [span_lo, span_hi) this frame's triangles may have
  // touched (conservative): what begin() clears and composite() blends.
  std::vector<int> span_lo, span_hi;
  int w = 0, h = 0;
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0; // dirty rectangle [x0,x1) x [y0,y1)
};
}
