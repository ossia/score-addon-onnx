//Copied from BlazePose.cpp

#include "RTMPose.hpp"

#include <Onnx/helpers/Images.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/RTMPose.hpp>

namespace OnnxModels
{
RTMPoseDetector::RTMPoseDetector() noexcept
{
  inputs.image.request_height = 256;
  inputs.image.request_width = 256;
}
RTMPoseDetector::~RTMPoseDetector() = default;

void RTMPoseDetector::operator()() { }

}
