#include "score_addon_onnx.hpp"

#include <score/plugins/FactorySetup.hpp>

#include <QFileInfo>
#include <Library/LibraryInterface.hpp>
#include <Library/ProcessesItemModel.hpp>
#include <Onnx/CommandFactory.hpp>
#include <Onnx/Executor.hpp>
#include <Onnx/Layer.hpp>
#include <Onnx/Process.hpp>
#include <Process/Drop/ProcessDropHandler.hpp>
#include <score_addon_onnx_commands_files.hpp>

namespace Onnx {
class LibraryHandler final : public QObject, public Library::LibraryInterface
{
    SCORE_CONCRETE("6e980c42-323a-40c4-b983-5c188bc19e4e")

    QSet<QString> acceptedFiles() const noexcept override { return {"onnx"}; }

    Library::Subcategories categories;

    void setup(Library::ProcessesItemModel &model, const score::GUIApplicationContext &ctx) override
    {
        // TODO relaunch whenever library path changes...
        const auto &key = Metadata<ConcreteKey_k, Onnx::Model>::get();
        QModelIndex node = model.find(key);
        if (node == QModelIndex{})
            return;

        categories.init(node, ctx);
    }

    void addPath(std::string_view path) override
    {
        QFileInfo file{QString::fromUtf8(path.data(), path.length())};
        Library::ProcessData pdata;
        pdata.prettyName = file.completeBaseName();

        pdata.key = Metadata<ConcreteKey_k, Onnx::Model>::get();
        pdata.customData = file.absoluteFilePath();
        categories.add(file, std::move(pdata));
    }
};

class DropHandler final : public Process::ProcessDropHandler
{
    SCORE_CONCRETE("fc93cc04-bdbd-43e7-bc6d-ebdafe4b5426")

    QSet<QString> fileExtensions() const noexcept override { return {"onnx"}; }

    void dropPath(std::vector<ProcessDrop> &vec,
                  const score::FilePath &filename,
                  const score::DocumentContext &ctx) const noexcept override
    {
        Process::ProcessDropHandler::ProcessDrop p;
        p.creation.key = Metadata<ConcreteKey_k, Onnx::Model>::get();
        p.creation.prettyName = filename.basename;
        p.creation.customData = filename.relative;

        vec.push_back(std::move(p));
    }
};

} // namespace Onnx

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
        FW<Execution::ProcessComponentFactory, Onnx::ProcessExecutorComponentFactory>>(ctx, key);
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
