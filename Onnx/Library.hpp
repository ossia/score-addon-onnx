#pragma once

#include <Library/LibraryInterface.hpp>
#include <Library/ProcessesItemModel.hpp>
#include <Process/Drop/ProcessDropHandler.hpp>

#include <Onnx/Process.hpp>

namespace Onnx
{
class LibraryHandler final
    : public QObject
    , public Library::LibraryInterface
{
  SCORE_CONCRETE("11f3cc12-540b-467b-828f-aeb4a26d0348")

  QSet<QString> acceptedFiles() const noexcept override { return {"onnx"}; }

  Library::Subcategories categories;

  void setup(
      Library::ProcessesItemModel& model,
      const score::GUIApplicationContext& ctx) override
  {
    // TODO relaunch whenever library path changes...
    const auto& key = Metadata<ConcreteKey_k, Onnx::Model>::get();
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
  SCORE_CONCRETE("f8799a8f-bb82-4dcd-b02e-a45b63dc87c6")

  QSet<QString> fileExtensions() const noexcept override { return {"onnx"}; }

  void dropPath(
      std::vector<ProcessDrop>& vec,
      const score::FilePath& filename,
      const score::DocumentContext& ctx) const noexcept override
  {
    Process::ProcessDropHandler::ProcessDrop p;
    p.creation.key = Metadata<ConcreteKey_k, Onnx::Model>::get();
    p.creation.prettyName = filename.basename;
    p.creation.customData = filename.relative;

    vec.push_back(std::move(p));
  }
};

} // namespace Onnx
