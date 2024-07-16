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
  COMPONENT_METADATA("642fde65-33ad-4f23-a225-5e94eb9f3ede")
public:
  ProcessExecutorComponent(
      Model& element,
      const Execution::Context& ctx,
      QObject* parent);
};

using ProcessExecutorComponentFactory
    = Execution::ProcessComponentFactory_T<ProcessExecutorComponent>;
}
