#pragma once
#include <score/command/Command.hpp>
#include <score/model/path/Path.hpp>

namespace Onnx
{
inline const CommandGroupKey& CommandFactoryName()
{
  static const CommandGroupKey key{"Onnx"};
  return key;
}

class Model;
class MyUndoRedoCommand final : public score::Command
{
  SCORE_COMMAND_DECL(
      CommandFactoryName(),
      MyUndoRedoCommand,
      "MyUndoRedoCommand")
public:
  explicit MyUndoRedoCommand(const Model& process);

public:
  void undo(const score::DocumentContext& ctx) const override;
  void redo(const score::DocumentContext& ctx) const override;

protected:
  void serializeImpl(DataStreamInput&) const override;
  void deserializeImpl(DataStreamOutput&) override;

private:
  Path<Model> m_path;
};
}
