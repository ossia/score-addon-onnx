#pragma once
#include <Onnx/helpers/TensorType.hpp>

#include <string>
#include <vector>

namespace Onnx
{
struct ModelSpec
{
  struct Port
  {
    std::string name;
    std::vector<int64_t> shape;
    TensorElemType elem_type{TensorElemType::Float};
  };
  std::vector<Port> inputs;
  std::vector<Port> outputs;

  std::vector<std::string> input_names, output_names;
  // These alias the *_names strings above (ORT's Run() wants const char**).
  // Because they alias, a naive copy/move would leave them pointing at the
  // SOURCE's string buffers (and SSO strings relocate on move), so we rebuild
  // them after every copy/move to point into THIS object's strings.
  std::vector<const char*> input_names_char;
  std::vector<const char*> output_names_char;

  void rebuildCharPointers()
  {
    input_names_char.resize(input_names.size());
    output_names_char.resize(output_names.size());
    for(std::size_t i = 0; i < input_names.size(); ++i)
      input_names_char[i] = input_names[i].c_str();
    for(std::size_t i = 0; i < output_names.size(); ++i)
      output_names_char[i] = output_names[i].c_str();
  }

  ModelSpec() = default;
  ModelSpec(const ModelSpec& o)
      : inputs(o.inputs), outputs(o.outputs), input_names(o.input_names)
      , output_names(o.output_names)
  {
    rebuildCharPointers();
  }
  ModelSpec(ModelSpec&& o) noexcept
      : inputs(std::move(o.inputs)), outputs(std::move(o.outputs))
      , input_names(std::move(o.input_names))
      , output_names(std::move(o.output_names))
  {
    rebuildCharPointers();
  }
  ModelSpec& operator=(const ModelSpec& o)
  {
    inputs = o.inputs;
    outputs = o.outputs;
    input_names = o.input_names;
    output_names = o.output_names;
    rebuildCharPointers();
    return *this;
  }
  ModelSpec& operator=(ModelSpec&& o) noexcept
  {
    inputs = std::move(o.inputs);
    outputs = std::move(o.outputs);
    input_names = std::move(o.input_names);
    output_names = std::move(o.output_names);
    rebuildCharPointers();
    return *this;
  }
};
}
