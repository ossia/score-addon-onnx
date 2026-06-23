#include "CommandFactory.hpp"
#include <Onnx/Process.hpp>

#include <score/model/path/PathSerialization.hpp>

#include <cstdio>

namespace Onnx
{
MyUndoRedoCommand::MyUndoRedoCommand(const Model& process)
    : m_path{process}
{
}

void MyUndoRedoCommand::undo(const score::DocumentContext& ctx) const
{
  auto& process = m_path.find(ctx);
  std::fprintf(stderr, "MyUndoRedoCommand: undo\n");
  // process.setSomeProperty(oldValue);
}

void MyUndoRedoCommand::redo(const score::DocumentContext& ctx) const
{
  auto& process = m_path.find(ctx);
  std::fprintf(stderr, "MyUndoRedoCommand: redo\n");
  // process.setSomeProperty(newValue);
}

void MyUndoRedoCommand::serializeImpl(DataStreamInput& s) const
{
  std::fprintf(stderr, "MyUndoRedoCommand: save\n");
  s << m_path;
}

void MyUndoRedoCommand::deserializeImpl(DataStreamOutput& s)
{
  std::fprintf(stderr, "MyUndoRedoCommand: load\n");
  s >> m_path;
}
}