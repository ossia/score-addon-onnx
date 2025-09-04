#pragma once
#include <Process/Dataflow/PortType.hpp>

#include <ossia/network/common/parameter_properties.hpp>

#include <QDebug>
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

inline QDebug operator<<(QDebug s, const ModelSpec& spec)
{
  s << "Model: " << spec.input_names.size() << spec.output_names.size()
    << "\n";
  for (auto& port : spec.input_names)
    s << " - i: " << port.c_str() << "\n";
  for (auto& port : spec.inputs)
    s << "   => " << port.name << (int)port.port_type << (int)port.data_type
      << port.shape << "\n";
  for (auto& port : spec.output_names)
    s << " - o: " << port.c_str() << "\n";
  for (auto& port : spec.outputs)
    s << "   => " << port.name << (int)port.port_type << (int)port.data_type
      << port.shape << "\n";

  return s;
}
}
