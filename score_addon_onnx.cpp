#include "score_addon_onnx.hpp"

#include <score/plugins/FactorySetup.hpp>

#include <QFileInfo>

#include <Onnx/CommandFactory.hpp>
#include <Onnx/Executor.hpp>
#include <Onnx/Library.hpp>
#include <Onnx/Process.hpp>
#include <score_addon_onnx_commands_files.hpp>

score_addon_onnx::score_addon_onnx() { }
score_addon_onnx::~score_addon_onnx() { }

std::vector<score::InterfaceBase*>
score_addon_onnx::factories(
    const score::ApplicationContext& ctx,
    const score::InterfaceKey& key) const
{
  return instantiate_factories<
      score::ApplicationContext,
      FW<Process::ProcessModelFactory, Onnx::ProcessFactory>,
      FW<Library::LibraryInterface, Onnx::LibraryHandler>,
      FW<Process::ProcessDropHandler, Onnx::DropHandler>,
      FW<Execution::ProcessComponentFactory,
         Onnx::ProcessExecutorComponentFactory>>(ctx, key);
}

std::pair<const CommandGroupKey, CommandGeneratorMap>
score_addon_onnx::make_commands()
{
  using namespace Onnx;
  std::pair<const CommandGroupKey, CommandGeneratorMap> cmds{
      CommandFactoryName(), CommandGeneratorMap{}};

  ossia::for_each_type<
#include <score_addon_onnx_commands.hpp>
      >(score::commands::FactoryInserter{cmds.second});

  return cmds;
}

#include <score/plugins/PluginInstances.hpp>
SCORE_EXPORT_PLUGIN(score_addon_onnx)
