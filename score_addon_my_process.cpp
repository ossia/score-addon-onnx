#include "score_addon_my_process.hpp"

#include <score/plugins/FactorySetup.hpp>

#include <MyProcess/CommandFactory.hpp>
#include <MyProcess/Executor.hpp>
#include <MyProcess/Inspector.hpp>
#include <MyProcess/Layer.hpp>
#include <MyProcess/LocalTree.hpp>
#include <MyProcess/Process.hpp>
#include <score_addon_my_process_commands_files.hpp>

score_addon_my_process::score_addon_my_process() { }

score_addon_my_process::~score_addon_my_process() { }

std::vector<std::unique_ptr<score::InterfaceBase>>
score_addon_my_process::factories(
    const score::ApplicationContext& ctx,
    const score::InterfaceKey& key) const
{
  return instantiate_factories<
      score::ApplicationContext,
      FW<Process::ProcessModelFactory, MyProcess::ProcessFactory>,
      FW<Process::LayerFactory, MyProcess::LayerFactory>,
      FW<Process::InspectorWidgetDelegateFactory, MyProcess::InspectorFactory>,
      FW<Execution::ProcessComponentFactory,
         MyProcess::ProcessExecutorComponentFactory>,
      FW<LocalTree::ProcessComponentFactory,
         MyProcess::LocalTreeProcessComponentFactory>>(ctx, key);
}

std::pair<const CommandGroupKey, CommandGeneratorMap>
score_addon_my_process::make_commands()
{
  using namespace MyProcess;
  std::pair<const CommandGroupKey, CommandGeneratorMap> cmds{
      CommandFactoryName(), CommandGeneratorMap{}};

  ossia::for_each_type<
#include <score_addon_my_process_commands.hpp>
      >(score::commands::FactoryInserter{cmds.second});

  return cmds;
}

#include <score/plugins/PluginInstances.hpp>
SCORE_EXPORT_PLUGIN(score_addon_my_process)
