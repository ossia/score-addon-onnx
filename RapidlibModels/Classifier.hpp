#pragma once
#include "src/classification.h"

#include <halp/controls.hpp>
#include <halp/dynamic_port.hpp>
#include <halp/meta.hpp>

#include <cstddef>
#include <functional>
#include <vector>

namespace RapidlibModels
{

struct Classifier
{
public:
  halp_meta(name, "Classifier");
  halp_meta(c_name, "classifier");
  halp_meta(category, "AI/Data processing");
  halp_meta(author, "RapidLib authors");
  halp_meta(
      description,
      "k-nearest-neighbour classifier: record examples of the input for each "
      "class, train, then recognize the class of new inputs "
      "(Wekinator-style interactive machine learning).");
  halp_meta(uuid, "039763f8-ea15-4900-9400-8c7f6a1c56cd");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-processing.html#classifier");
  enum Mode
  {
    Test,
    Infer
  };
  struct
  {
    halp::val_port<"Input", std::vector<double>> input;
    halp::enum_t<Mode, "Mode"> mode;
    halp::impulse_button<"Record"> record;
    halp::impulse_button<"Undo"> undo;
    halp::impulse_button<"Reset"> reset;
    halp::impulse_button<"Train"> train;

    // FIXME if set to 1 by default it's not created
    struct : halp::spinbox_i32<"Param. count", halp::range{0, 1024, 0}>
    {
      static std::function<void(Classifier&, int)> on_controller_interaction()
      {
        return [](Classifier& object, int value)
        { object.inputs.parameters_i.request_port_resize(value); };
      }
    } controller;
    // The recorded value of each port is a class label; rapidLib's kNN
    // rounds labels to the nearest integer, so these are int spinboxes
    // (a 0..1 float knob could only ever express classes 0 and 1).
    halp::dynamic_port<halp::spinbox_i32<"Class {}", halp::range{0, 1000, 0}>>
        parameters_i;
  } inputs;

  struct
  {
    halp::val_port<"Output", std::vector<double>> output;
    halp::val_port<"Examples", int> examples;
  } outputs;

  Classifier() noexcept;
  ~Classifier();

  void operator()();

private:
  std::vector<rapidLib::trainingExample> m_trainingSet;

  rapidLib::classification m_model;
  std::size_t m_numInputs{};
  bool m_trained{};
};
}
