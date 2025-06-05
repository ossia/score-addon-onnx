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
  Resnet() { }

  void loadClasses(std::string_view filePath)
  {
    QFile f(QString::fromUtf8(filePath.data(), filePath.size()));
    if (f.open(QIODevice::ReadOnly))
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
    if (classes.empty())
    {
      [[unlikely]];
      return;
    }

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
        if (the_class >= 0 && the_class < classes.size())
        {
          float value = recog[the_class];

          out.push_back({classes[the_class].toStdString(), value});
        }
      }
    }
  }
};

struct EmotionNet
{
  // Input format: float32[batch_size,3,224,224]
  // Output format: float values representing the emotion classes.
  // tensor: float32[batch_size,8] or ,10 if is_mtl
  QList<QByteArray> classes_7;
  QList<QByteArray> classes_8;
  QList<QByteArray> classes_10;
  EmotionNet()
  {
    classes_7 = {
        "Anger",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    };
    classes_8 = {
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Happiness",
        "Neutral",
        "Sadness",
        "Surprise",
    };
    classes_10 = classes_8;
    classes_10.push_back("class_8");
    classes_10.push_back("class_9");
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
      if (N != 7 && N != 8 && N != 10)
        return;

      std::span<const float> res = std::span(ot.GetTensorData<float>(), N);

      thread_local std::vector<float> recog;
      recog.clear();

      // See post processing here:
      // https://github.com/sb-ai-lab/EmotiEffLib/blob/90690cda1644819d7c83b914db46a6d7e7efbd91/emotieffcpplib/src/facial_analysis.cpp#L119
      Onnx::softmax(res, recog);

      auto& classes = N == 7 ? classes_7 : (N == 8 ? classes_8 : classes_10);
      for (int i = 0; i < std::min(N, 8); i++)
      {
        out.push_back({classes[i].toStdString(), recog[i]});
      }
    }
  }
};
}
