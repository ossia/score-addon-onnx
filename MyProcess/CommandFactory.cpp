#include "CommandFactory.hpp"
#include <QDebug>

namespace MyProcess
{
MyUndoRedoCommand::MyUndoRedoCommand(const Model& process)
    : m_path{process}
{
}

void MyUndoRedoCommand::undo(const score::DocumentContext& ctx) const
{
  auto& process = m_path.find(ctx);
  qDebug() << "MyUndoRedoCommand: undo";
  // process.setSomeProperty(oldValue);
}

void MyUndoRedoCommand::redo(const score::DocumentContext& ctx) const
{
  auto& process = m_path.find(ctx);
  qDebug() << "MyUndoRedoCommand: redo";
  // process.setSomeProperty(newValue);
}

void MyUndoRedoCommand::serializeImpl(DataStreamInput& s) const
{
  qDebug() << "MyUndoRedoCommand: save";
  s << m_path;
}

void MyUndoRedoCommand::deserializeImpl(DataStreamOutput& s)
{
  qDebug() << "MyUndoRedoCommand: load";
  s >> m_path;
}
}