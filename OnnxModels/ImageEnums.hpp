#pragma once

// Shared image/video preprocessing + output enums, defined ONCE here and
// included by both ImageProcessor and VideoProcessor. They used to be defined
// separately (and identically) in each node's header, which put two definitions
// of OnnxModels::OutputMode / TaskMode / ... in the same namespace — an ODR
// hazard. A single definition removes it.
namespace OnnxModels
{

enum class InputNormalization
{
  DivBy255, // (x) / 255          -> [0, 1]
  ImageNet, // ImageNet mean/std  -> standardized
  Centered, // (x - 127.5)/127.5  -> [-1, 1]
  None,     // raw [0, 255]
};

enum class ChannelOrder
{
  RGB,
  BGR, // OpenCV-trained nets (YOLOX/RTMDet families)
};

enum class ResizeMode
{
  Crop,      // keep aspect, fill + center-crop overflow (default; legacy behavior)
  Stretch,   // ignore aspect, scale to model size
  Letterbox, // keep aspect, fit + pad
};

// Override the auto-detected model task, or pick a specific output target.
enum class TaskMode
{
  Auto, // classifyImage() decides
  Image,
  Mask,
  Depth,
  Data,
};

// Output value -> pixel mapping. Auto picks from the task (image->DirectClamp,
// depth/mask->MinMaxNormalize).
enum class OutputMode
{
  Auto,
  DirectClamp,
  MinMaxNormalize,
  Denormalize,
  Passthrough,
  Half255,
};

}
