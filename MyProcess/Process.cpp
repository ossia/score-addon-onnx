#include "Process.hpp"

#include <wobjectimpl.h>

W_OBJECT_IMPL(MyProcess::Model)
namespace MyProcess
{

Model::Model(
    const TimeVal& duration,
    const Id<Process::ProcessModel>& id,
    QObject* parent)
    : Process::ProcessModel{duration, id, "MyProcessProcess", parent}
{
  metadata().setInstanceName(*this);
}

Model::~Model() { }

QString Model::prettyName() const noexcept
{
  return tr("MyProcess Process");
}

void Model::setDurationAndScale(const TimeVal& newDuration) noexcept { }

void Model::setDurationAndGrow(const TimeVal& newDuration) noexcept { }

void Model::setDurationAndShrink(const TimeVal& newDuration) noexcept { }
}
template <>
void DataStreamReader::read(const MyProcess::Model& proc)
{
  insertDelimiter();
}

template <>
void DataStreamWriter::write(MyProcess::Model& proc)
{
  checkDelimiter();
}

template <>
void JSONReader::read(const MyProcess::Model& proc)
{
}

template <>
void JSONWriter::write(MyProcess::Model& proc)
{
}
