#pragma once
#include <Process/Execution/ProcessComponent.hpp>

#include <ossia/dataflow/node_process.hpp>

namespace Onnx
{
class Model;
class ProcessExecutorComponent final
    : public Execution::
          ProcessComponent_T<Onnx::Model, ossia::node_process>
{
  COMPONENT_METADATA("77262c15-17c7-4d69-ac13-ff858aa9fd8b")
public:
  ProcessExecutorComponent(
      Model& element,
      const Execution::Context& ctx,
      QObject* parent);
};

using ProcessExecutorComponentFactory
    = Execution::ProcessComponentFactory_T<ProcessExecutorComponent>;
}
