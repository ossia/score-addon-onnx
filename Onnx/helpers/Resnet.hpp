#pragma once
#include <Onnx/helpers/ModelSpec.hpp>
#include <Onnx/helpers/OnnxContext.hpp>
#include <Onnx/helpers/Utilities.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <span>
#include <string>

namespace OnnxModels
{
// Softmax, unless the model already outputs probabilities (it ends in a
// Softmax): a second one flattens them towards 1/N.
inline void toProbabilities(std::span<const float> in, std::vector<float>& out)
{
  float sum = 0.f, mn = 0.f;
  if(!in.empty())
    mn = *std::min_element(in.begin(), in.end());
  for(float v : in)
    sum += v;
  if(mn >= 0.f && std::abs(sum - 1.f) < 1e-3f)
    out.assign(in.begin(), in.end());
  else
    Onnx::softmax(in, out);
}

struct Resnet
{
  // Output format: 1000 float values representing the Imagenet classes.
  std::vector<std::string> classes;
  Resnet() { }

  void loadClasses(std::string_view filePath)
  {
    classes.clear();
    std::ifstream f{std::string(filePath)};
    if (!f)
      return;
    for (std::string line; std::getline(f, line);)
      classes.push_back(line);
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
      toProbabilities(res, recog);

      // Resized on every call: the buffer is shared by every Resnet node on
      // this thread, and sizing it only once let a model with another class
      // count sort a stale range or read past the end.
      thread_local std::vector<int> idx;
      idx.resize(N);
      std::iota(idx.begin(), idx.end(), 0);

      std::stable_sort(
          idx.begin(),
          idx.end(),
          [&](int i1, int i2) { return recog[i1] > recog[i2]; });

      for (int i = 0, n = std::min(5, N); i < n; i++)
      {
        int the_class = idx[i];
        if (the_class >= 0 && the_class < (int)classes.size())
        {
          float value = recog[the_class];

          out.push_back({classes[the_class], value});
        }
      }
    }
  }
};

struct EmotionNet
{
  // EmotiEffLib / HSEmotion: float32[batch,3,224,224] -> 7 or 8 emotion
  // logits, followed by valence and arousal on the multi-task models (9 or 10
  // outputs). FER+: float32[batch,1,64,64] -> 8 emotions in its own order.
  std::vector<std::string> classes_7{
      "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"};
  std::vector<std::string> classes_8{
      "Anger",   "Contempt", "Disgust", "Fear",
      "Happiness", "Neutral", "Sadness", "Surprise"};
  std::vector<std::string> classes_ferplus{
      "Neutral", "Happiness", "Surprise", "Sadness",
      "Anger",   "Disgust",   "Fear",     "Contempt"};

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
    const bool ferplus = !spec.inputs.empty() && spec.inputs[0].shape.size() == 4
                         && spec.inputs[0].shape[1] == 1;
    for (const Ort::Value& ot : output_tensors)
    {
      const int N = ot.GetTensorTypeAndShapeInfo().GetElementCount();
      // Valence and arousal are not part of the emotion distribution: the
      // softmax covers the emotions only, and they are passed through.
      const bool va = (N == 9 || N == 10);
      const int emotions = va ? N - 2 : N;
      if (emotions != 7 && emotions != 8)
        return;
      std::span<const float> res(ot.GetTensorData<float>(), N);

      // See post processing here:
      // https://github.com/sb-ai-lab/EmotiEffLib/blob/90690cda1644819d7c83b914db46a6d7e7efbd91/emotieffcpplib/src/facial_analysis.cpp#L119
      thread_local std::vector<float> recog;
      recog.clear();
      toProbabilities(res.first(emotions), recog);

      const auto& classes
          = emotions == 7 ? classes_7 : (ferplus ? classes_ferplus : classes_8);
      for (int i = 0; i < emotions; i++)
        out.push_back({classes[i], recog[i]});
      if (va)
      {
        out.push_back({"Valence", res[N - 2]});
        out.push_back({"Arousal", res[N - 1]});
      }
    }
  }
};
}
