#include "Process.hpp"

#include <wobjectimpl.h>

W_OBJECT_IMPL(Onnx::Model)
namespace Onnx
{

Model::Model(
    const TimeVal& duration,
    const Id<Process::ProcessModel>& id,
    QObject* parent)
    : Process::ProcessModel{duration, id, "OnnxProcess", parent}
{
  metadata().setInstanceName(*this);
}

Model::~Model() { }

QString Model::prettyName() const noexcept
{
  return tr("Onnx Process");
}

void Model::setDurationAndScale(const TimeVal& newDuration) noexcept { }

void Model::setDurationAndGrow(const TimeVal& newDuration) noexcept { }

void Model::setDurationAndShrink(const TimeVal& newDuration) noexcept { }
}
template <>
void DataStreamReader::read(const Onnx::Model& proc)
{
  insertDelimiter();
}

template <>
void DataStreamWriter::write(Onnx::Model& proc)
{
  checkDelimiter();
}

template <>
void JSONReader::read(const Onnx::Model& proc)
{
}

template <>
void JSONWriter::write(Onnx::Model& proc)
{
}
