#pragma once
// Classify an image/generative ONNX model purely from its declared input/output
// shapes & names — the image-processing counterpart to ModelRole.hpp's
// classify() (which only covers the pose/detection families). Dependency-free
// (reuses Onnx::ModelIO + Onnx::detail), so it stays free of Qt/ossia/ORT and is
// standalone-compilable.
//
// Covers the three structural cases the generic ImageProcessor targets:
//   image -> image   (style / super-res / restoration / deblur / low-light)
//   image -> data    (depth, segmentation mask, matte, scalar)
//   data  -> image   (latent vector -> image; single-network GANs)
//
// See OnnxModels/ImageProcessor-ANALYSIS.md for the grounded I/O taxonomy.
#include <Onnx/helpers/ModelRole.hpp> // ModelIO, detail::nameContains

#include <cstdint>

namespace Onnx
{
enum class ImageModelKind : uint8_t
{
  Unknown = 0,
  ImageToImage,  // RGB in -> RGB(A) out
  ImageToMask,   // RGB in -> single-channel mask / matte / segmentation
  ImageToDepth,  // RGB in -> single-channel (relative) depth
  LatentToImage, // latent/data vector in -> RGB out (GAN)
  ImageToData,   // RGB in -> scalar / vector (e.g. JPEG quality factor)
};

// Spatial tensor layout of an image-shaped port. Maps 1:1 to ImageOps'
// TensorLayout for the RGB cases; NchwGray is the 1-channel extension.
enum class ImgLayout : uint8_t
{
  Unknown = 0,
  NchwRgb, // planar [N,C,H,W], C in {3,4}
  NhwcRgb, // interleaved [N,H,W,C], C in {3,4}
  NchwGray // planar single channel [N,1,H,W]
};

struct ImageModelRole
{
  ImageModelKind kind = ImageModelKind::Unknown;

  // --- input ---
  ImgLayout in_layout = ImgLayout::Unknown;
  int in_w = 0, in_h = 0;     // 0 => dynamic (caller supplies a resolution)
  int in_channels = 3;        // declared channel count (3 assumed if symbolic)
  bool in_dynamic_hw = false; // a spatial dim was symbolic / <= 0
  int in_stride = 32;         // snap dynamic H/W to this multiple (14 for ViT depth)
  int latent_dim = 0;         // LatentToImage: flattened non-batch input size

  // --- output (the selected index) ---
  int out_index = 0;
  int out_rank = 0;
  ImgLayout out_layout = ImgLayout::Unknown;
  int out_w = 0, out_h = 0, out_channels = 0;
  bool out_dynamic = false; // a spatial output dim was symbolic / <= 0

  bool valid() const noexcept { return kind != ImageModelKind::Unknown; }
};

namespace detail
{
// Resolve (layout, channels, w, h, dynamic) for a 3-D/4-D image-shaped tensor.
// Channel detection mirrors classify(): a literal {1,3,4} in the channel slot
// wins; a symbolic/other channel dim (e.g. depth_pro's num_channels) is assumed
// NCHW-RGB. Returns false if the shape is not image-shaped (rank < 3).
struct LayoutInfo
{
  ImgLayout layout = ImgLayout::Unknown;
  int channels = 0, w = 0, h = 0;
  bool dynamic = false;
};
inline bool isChan(int64_t d) noexcept
{
  return d == 1 || d == 3 || d == 4;
}
inline LayoutInfo resolveLayout(const std::vector<int64_t>& s)
{
  LayoutInfo li;
  const int rank = static_cast<int>(s.size());
  int ci = -1, hi = -1, wi = -1;
  bool nhwc = false;
  if(rank == 4)
  {
    if(isChan(s[1])) { ci = 1; hi = 2; wi = 3; }       // NCHW
    else if(isChan(s[3])) { ci = 3; hi = 1; wi = 2; nhwc = true; } // NHWC
    else { ci = 1; hi = 2; wi = 3; }                   // symbolic channel -> NCHW
  }
  else if(rank == 3)
  {
    // [C,H,W] (no batch) or [N,H,W] (squeezed single channel).
    if(isChan(s[0])) { ci = 0; hi = 1; wi = 2; }       // CHW
    else if(isChan(s[2])) { ci = 2; hi = 0; wi = 1; nhwc = true; } // HWC
    else { ci = -1; hi = 1; wi = 2; }                  // [N,H,W] -> 1 channel
  }
  else
  {
    return li; // not image-shaped
  }

  const int64_t c = (ci >= 0) ? s[ci] : 1;
  li.channels = (c > 0) ? static_cast<int>(c) : 3;
  li.h = (s[hi] > 0) ? static_cast<int>(s[hi]) : 0;
  li.w = (s[wi] > 0) ? static_cast<int>(s[wi]) : 0;
  li.dynamic = (li.w == 0 || li.h == 0);
  if(li.channels == 1)
    li.layout = ImgLayout::NchwGray; // single channel: planar regardless
  else
    li.layout = nhwc ? ImgLayout::NhwcRgb : ImgLayout::NchwRgb;
  return li;
}
} // namespace detail

inline ImageModelRole classifyImage(const ModelIO& io, int out_index = 0)
{
  ImageModelRole r;
  if(io.inputs.empty())
    return r;

  const auto& in = io.inputs[0].shape;
  const int in_rank = static_cast<int>(in.size());

  // --- latent / data input (rank <= 2) ---------------------------------------
  bool latent_in = false;
  if(in_rank <= 2)
  {
    latent_in = true;
    int64_t d = 1;
    for(int i = (in_rank == 2 ? 1 : 0); i < in_rank; ++i)
      if(in[i] > 0)
        d *= in[i];
    r.latent_dim = static_cast<int>(d);
  }
  else
  {
    const auto li = detail::resolveLayout(in);
    r.in_layout = li.layout;
    r.in_channels = li.channels;
    r.in_w = li.w;
    r.in_h = li.h;
    r.in_dynamic_hw = li.dynamic;
  }

  // --- output selection ------------------------------------------------------
  if(io.outputs.empty())
    return r;
  if(out_index < 0 || out_index >= static_cast<int>(io.outputs.size()))
    out_index = 0;
  r.out_index = out_index;
  const auto& out = io.outputs[out_index];
  r.out_rank = static_cast<int>(out.shape.size());

  // depth vs mask + stride hint: depth-patch ViTs (DINOv2 / Depth-Anything) need
  // multiples of 14; UNet/segmentation tolerate 32. Detect by output/input name.
  const bool depthName = detail::nameContains(out.name, "depth")
                         || detail::nameContains(out.name, "disp")
                         || detail::nameContains(io.inputs[0].name, "depth")
                         || detail::nameContains(io.inputs[0].name, "dinov2");
  r.in_stride = depthName ? 14 : 32;

  // Classify the output tensor.
  bool out_spatial = false;
  if(r.out_rank >= 3)
  {
    const auto lo = detail::resolveLayout(out.shape);
    if(lo.layout != ImgLayout::Unknown)
    {
      out_spatial = true;
      r.out_layout = lo.layout;
      r.out_channels = lo.channels;
      r.out_w = lo.w;
      r.out_h = lo.h;
      r.out_dynamic = lo.dynamic;
    }
  }

  // --- decide kind -----------------------------------------------------------
  if(latent_in)
  {
    // data -> image only makes sense if the output is a spatial RGB image.
    r.kind = (out_spatial && r.out_channels >= 3) ? ImageModelKind::LatentToImage
                                                   : ImageModelKind::Unknown;
    return r;
  }

  if(!out_spatial)
  {
    r.kind = ImageModelKind::ImageToData; // scalar / vector (fbcnn QF, focal len)
    return r;
  }

  if(r.out_channels >= 3)
    r.kind = ImageModelKind::ImageToImage;
  else // single channel (or >4 logits treated as visualizable mask)
    r.kind = depthName ? ImageModelKind::ImageToDepth : ImageModelKind::ImageToMask;

  return r;
}
} // namespace Onnx
