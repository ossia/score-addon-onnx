#pragma once
#include <Process/Dataflow/PortType.hpp>

#include <ossia/network/common/parameter_properties.hpp>

#include <QString>

#include <vector>

namespace Onnx
{
struct ModelSpec
{
  struct Port
  {
    QString name;
    Process::PortType port_type{};
    ossia::val_type data_type{};
    std::vector<int64_t> shape;
  };
  std::vector<Port> inputs;
  std::vector<Port> outputs;

  std::vector<std::string> input_names, output_names;
  std::vector<const char*> input_names_char;
  std::vector<const char*> output_names_char;
};
}
