#pragma once
#include "src/regression.h"

#include <halp/controls.hpp>
#include <halp/dynamic_port.hpp>
#include <halp/meta.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace RapidlibModels
{

struct Regressor
{
public:
  halp_meta(name, "Regressor");
  halp_meta(c_name, "regressor");
  halp_meta(category, "AI/Data processing");
  halp_meta(author, "RapidLib authors");
  halp_meta(
      description,
      "Neural-network regression: record example mappings from the input to "
      "the parameters, train, then generate interpolated parameters from new "
      "inputs (Wekinator-style interactive machine learning).");
  halp_meta(uuid, "c9613fba-6318-463c-91b0-cab4c6c7ab2b");
  halp_meta(
      manual_url,
      "https://ossia.io/score-docs/processes/ai-processing.html#regressor");
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
      static std::function<void(Regressor&, int)> on_controller_interaction()
      {
        return [](Regressor& object, int value)
        { object.inputs.parameters_i.request_port_resize(value); };
      }
    } controller;
    halp::dynamic_port<halp::knob_f32<"Param. {}">> parameters_i;
  } inputs;

  struct
  {
    halp::val_port<"Output", std::vector<double>> output;
    halp::val_port<"Examples", int> examples;
  } outputs;

  // Training a multilayer perceptron (500 epochs over the whole training set,
  // one thread per output parameter) is far too slow for the processing
  // thread; run it in the host's worker thread pool and swap the finished
  // model in on the processing thread. Inference keeps using the previous
  // model (if any) until then.
  struct worker
  {
    std::function<void(std::vector<rapidLib::trainingExample>)> request;

    // Called in a worker thread; the returned function is then invoked in
    // this object's processing thread.
    static std::function<void(Regressor&)>
    work(std::vector<rapidLib::trainingExample> trainingSet);
  } worker;

  Regressor() noexcept;
  ~Regressor();

  void operator()();

private:
  std::vector<rapidLib::trainingExample> m_trainingSet;

  std::shared_ptr<rapidLib::regression> m_model;
  std::size_t m_numInputs{};
  bool m_trained{};
};
}
