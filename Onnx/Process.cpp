#include "Process.hpp"

#include <Process/Dataflow/WidgetInlets.hpp>

#include <score/tools/File.hpp>

#include <Onnx/helpers/OnnxBase.hpp>
#include <wobjectimpl.h>

#include <cassert>

W_OBJECT_IMPL(Onnx::Model)

/*
auto x = []
{
  // OnnxRunContext ctx(
  //     "/opt/models/models/validated/vision/classification/resnet/model/"
  //     "resnet101-v1-7.onnx");

  //OnnxRunContext ctx(
  //    "/opt/models/PINTO_model_zoo/307_YOLOv7/yolov7_640x640.onnx");
  //OnnxRunContext ctx("/opt/models/yolov7/yolov7.onnx");
  OnnxRunContext ctx(
      "/home/jcelerier/Documents/ossia/score/packages/yolov7.onnx");

  auto spec = ctx.readModelSpec();
  if (spec.inputs.empty())
  {
    qDebug() << "No input port";
    return 0;
  }

  if (std::ranges::any_of(spec.inputs[0].shape, [](int x) { return x <= 0; }))
  {
    qDebug() << "invalid shape" << spec.inputs[0].shape;
    return 0;
  }

  qDebug() << "Infering...";

  auto t = ctx.tensorFromImage(spec.inputs[0], 224, 224, false);

  Ort::Value input_tensors[1]{std::move(t.value)};
  qDebug() << input_tensors[0].GetTypeInfo();
  assert(
      input_tensors[0].IsTensor()
      && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()
             == spec.inputs[0].shape);
  qDebug() << "Input_tensor shape: "
           << input_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

  ctx.infer(spec, input_tensors);
  std::exit(0);
  return 0;
}();
*/

namespace Onnx
{

Model::Model(const TimeVal &duration,
             const QString &name,
             const Id<Process::ProcessModel> &id,
             QObject *parent)
    : Process::ProcessModel{duration, id, "OnnxProcess", parent}
    , m_text{name}
{
  qDebug() << "Loading...";

  metadata().setInstanceName(*this);
  /*
  OnnxRunContext ctx(
      score::locateFilePath(name, score::IDocument::documentContext(*parent)));
  auto spec = ctx.readModelSpec();
  if (spec.inputs.empty())
  {
    qDebug() << "No input port";
    return;
  }
*/
}

Model::~Model() { }

QString Model::prettyName() const noexcept
{
  return tr("Onnx Process");
}
} // namespace Onnx
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
{}

template <>
void JSONWriter::write(Onnx::Model& proc)
{}
