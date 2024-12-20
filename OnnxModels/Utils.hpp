#pragma once

#include <halp/file_port.hpp>
#include <halp/meta.hpp>

namespace OnnxModels
{
struct ModelPort : halp::file_port<"Model", halp::mmap_file_view> {
  halp_meta(extensions, "*.onnx");
};
}
