#pragma once
#include "src/classification.h"

#include <halp/controls.hpp>
#include <halp/dynamic_port.hpp>
#include <halp/meta.hpp>

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
  halp_meta(description, "Linear regression across a set of parameters.");
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
    halp::dynamic_port<halp::knob_f32<"Param. {}">> parameters_i;
  } inputs;

  struct
  {
    halp::val_port<"Output", std::vector<double>> output;
  } outputs;

  Classifier() noexcept;
  ~Classifier();

  void operator()();

private:
  std::vector<rapidLib::trainingExample> m_trainingSet;

  rapidLib::classification m_model;
  bool m_trained{};
};
}
