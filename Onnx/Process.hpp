#pragma once
#include <Process/GenericProcessFactory.hpp>
#include <Process/Process.hpp>

#include <Process/Execution/ProcessComponent.hpp>
#include <Process/GenericProcessFactory.hpp>
#include <Process/Process.hpp>
#include <Process/ProcessMetadata.hpp>
#include <Process/Script/ScriptEditor.hpp>

#include <Scenario/Commands/ScriptEditCommand.hpp>

#include <Control/DefaultEffectItem.hpp>
#include <Effect/EffectFactory.hpp>

#include <ossia/dataflow/execution_state.hpp>
#include <ossia/dataflow/graph_node.hpp>
#include <ossia/dataflow/node_process.hpp>

#include <verdigris>

#include <Onnx/CommandFactory.hpp>
#include <Onnx/Metadata.hpp>

#include <score/command/Command.hpp>

namespace Onnx
{
class Model final : public Process::ProcessModel
{
  SCORE_SERIALIZE_FRIENDS
  PROCESS_METADATA_IMPL(Onnx::Model)
  W_OBJECT(Model)

public:
    Model(const TimeVal &duration,
          const QString &name,
          const Id<Process::ProcessModel> &id,
          QObject *parent);

    template<typename Impl>
    Model(Impl &vis, QObject *parent)
        : Process::ProcessModel{vis, parent}
    {
        vis.writeTo(*this);
  }

  ~Model() override;

  static constexpr bool hasExternalUI() noexcept { return true; }

  bool validate(const QString &txt) const noexcept;

  void errorMessage(int line, const QString &e) W_SIGNAL(errorMessage, line, e);

  private:
  QString prettyName() const noexcept override;
  QString m_text;
};

using ProcessFactory = Process::ProcessFactory_T<Onnx::Model>;
}

namespace Process {
template<>
QString EffectProcessFactory_T<Onnx::Model>::customConstructionData() const noexcept;

template<>
Process::Descriptor EffectProcessFactory_T<Onnx::Model>::descriptor(QString d) const noexcept;
} // namespace Process
