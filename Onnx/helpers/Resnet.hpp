#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

namespace OnnxModels
{
struct Resnet
{
  // Output format: 1000 float values representing the Imagenet classes.
  QList<QByteArray> classes;
  Resnet()
  {
    QFile f("/opt/models/imagenet_classes.txt");
    f.open(QIODevice::ReadOnly);
    classes = f.readAll().split('\n');
  }

  struct recognition_type
  {
    std::string name;
    float probability{};
  };

  void processOutput(
      const Onnx::ModelSpec& spec,
      std::span<Ort::Value> output_tensors,
      std::vector<recognition_type>& out) const
  {
    for (const Ort::Value& ot : output_tensors)
    {
      const int N = ot.GetTensorTypeAndShapeInfo().GetElementCount();
      std::span<const float> res = std::span(ot.GetTensorData<float>(), N);

      thread_local std::vector<float> recog;
      recog.clear();
      Onnx::softmax(res, recog);

      thread_local std::vector<int> idx(N);
      std::iota(idx.begin(), idx.end(), 0);

      std::stable_sort(
          idx.begin(),
          idx.end(),
          [&](int i1, int i2) { return recog[i1] > recog[i2]; });

      for (int i = 0; i < 5; i++)
      {
        int the_class = idx[i];
        float value = recog[the_class];

        out.push_back({classes[the_class].toStdString(), value});
      }
    }
  }
};
}
