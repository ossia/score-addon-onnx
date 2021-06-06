#pragma once
#include <Process/Inspector/ProcessInspectorWidgetDelegate.hpp>
#include <Process/Inspector/ProcessInspectorWidgetDelegateFactory.hpp>

#include <score/command/Dispatchers/CommandDispatcher.hpp>

#include <MyProcess/Process.hpp>

namespace MyProcess
{
class InspectorWidget final
    : public Process::InspectorWidgetDelegate_T<MyProcess::Model>
{
public:
  explicit InspectorWidget(
      const MyProcess::Model& object,
      const score::DocumentContext& context,
      QWidget* parent);
  ~InspectorWidget() override;

private:
  CommandDispatcher<> m_dispatcher;
};

class InspectorFactory final
    : public Process::InspectorWidgetDelegateFactory_T<Model, InspectorWidget>
{
  SCORE_CONCRETE("00000000-0000-0000-0000-000000000000")
};
}
