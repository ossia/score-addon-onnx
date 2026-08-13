// Tests for the Wekinator-style interactive machine learning objects:
// Classifier (RapidLib kNN) and Regressor (RapidLib multilayer perceptron).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <RapidlibModels/Classifier.hpp>
#include <RapidlibModels/Regressor.hpp>

#include <functional>
#include <utility>
#include <vector>

using Catch::Approx;
using RapidlibModels::Classifier;
using RapidlibModels::Regressor;

namespace
{
// Run one tick, then clear the impulse buttons like the host bindings do
// (avnd/binding/ossia/port_run_postprocess.hpp resets optional-valued
// inlets after every tick).
template <typename T>
void tick(T& obj)
{
  obj();
  obj.inputs.record.value.reset();
  obj.inputs.undo.value.reset();
  obj.inputs.reset.value.reset();
  obj.inputs.train.value.reset();
}

template <typename T>
void set_params(T& obj, const std::vector<double>& params)
{
  REQUIRE(obj.inputs.parameters_i.ports.size() == params.size());
  for (std::size_t i = 0; i < params.size(); i++)
  {
    auto& port = obj.inputs.parameters_i.ports[i];
    port.value = static_cast<std::decay_t<decltype(port.value)>>(params[i]);
  }
}

template <typename T>
void record(T& obj, std::vector<double> in, const std::vector<double>& params)
{
  obj.inputs.input.value = std::move(in);
  set_params(obj, params);
  obj.inputs.record.value.emplace();
  tick(obj);
}

template <typename T>
void train(T& obj)
{
  obj.inputs.train.value.emplace();
  tick(obj);
}

template <typename T>
const std::vector<double>& infer(T& obj, std::vector<double> in)
{
  obj.inputs.mode.value = T::Mode::Infer;
  obj.inputs.input.value = std::move(in);
  tick(obj);
  return obj.outputs.output.value;
}
}

TEST_CASE("Classifier: Test mode passes the parameters through")
{
  Classifier c;
  c.inputs.parameters_i.ports.resize(2);
  set_params(c, {3, 7});
  c.inputs.mode.value = Classifier::Mode::Test;
  tick(c);

  REQUIRE(c.outputs.output.value == std::vector<double>{3, 7});
  REQUIRE(c.outputs.examples.value == 0);
}

TEST_CASE("Classifier: records, trains and recognizes several classes")
{
  Classifier c;
  c.inputs.parameters_i.ports.resize(1);

  // Three classes at three corners of the feature space. Class labels beyond
  // 1 exercise the int spinbox ports: kNN rounds labels to integers, so the
  // former 0..1 float knobs could never express a third class.
  record(c, {0., 0.}, {0});
  record(c, {1., 0.}, {1});
  record(c, {0., 1.}, {2});
  REQUIRE(c.outputs.examples.value == 3);

  train(c);

  CHECK(infer(c, {0.1, 0.1}) == std::vector<double>{0});
  CHECK(infer(c, {0.9, 0.1}) == std::vector<double>{1});
  CHECK(infer(c, {0.1, 0.9}) == std::vector<double>{2});
}

TEST_CASE("Classifier: unusable examples are not recorded")
{
  Classifier c;
  c.inputs.parameters_i.ports.resize(1);

  // Empty feature vector: nothing connected to the input yet.
  record(c, {}, {0});
  REQUIRE(c.outputs.examples.value == 0);

  record(c, {0., 0.}, {0});
  REQUIRE(c.outputs.examples.value == 1);

  // Input arity changed upstream: rapidLib's train() would throw on it.
  record(c, {1., 2., 3.}, {1});
  REQUIRE(c.outputs.examples.value == 1);

  // Parameter count changed between recordings: same problem.
  c.inputs.parameters_i.ports.resize(2);
  record(c, {1., 1.}, {1, 1});
  REQUIRE(c.outputs.examples.value == 1);
}

TEST_CASE("Classifier: undo removes the last example, reset removes them all")
{
  Classifier c;
  c.inputs.parameters_i.ports.resize(1);

  // Undo with nothing recorded must not crash.
  c.inputs.undo.value.emplace();
  tick(c);
  REQUIRE(c.outputs.examples.value == 0);

  record(c, {0., 0.}, {0});
  record(c, {1., 1.}, {1});
  REQUIRE(c.outputs.examples.value == 2);

  c.inputs.undo.value.emplace();
  tick(c);
  REQUIRE(c.outputs.examples.value == 1);

  // Only the class-0 example remains: everything classifies as 0.
  train(c);
  CHECK(infer(c, {1., 1.}) == std::vector<double>{0});

  // Reset clears the examples but keeps the trained model, so the object
  // keeps working until the next training.
  c.inputs.reset.value.emplace();
  tick(c);
  REQUIRE(c.outputs.examples.value == 0);
  CHECK(infer(c, {1., 1.}) == std::vector<double>{0});
}

TEST_CASE("Classifier: training without examples is a safe no-op")
{
  Classifier c;
  c.inputs.parameters_i.ports.resize(1);
  train(c);

  CHECK(infer(c, {0.5, 0.5}).empty());
}

TEST_CASE("Classifier: inference with a mismatched input arity is skipped")
{
  Classifier c;
  c.inputs.parameters_i.ports.resize(1);
  record(c, {0., 0.}, {0});
  record(c, {1., 1.}, {1});
  train(c);

  REQUIRE(infer(c, {1., 1.}) == std::vector<double>{1});

  // Trained on two features; feeding three used to throw std::length_error
  // through the processing thread. The output keeps its previous value.
  CHECK(infer(c, {1., 1., 1.}) == std::vector<double>{1});
}

TEST_CASE("Regressor: Test mode passes the parameters through")
{
  Regressor r;
  r.inputs.parameters_i.ports.resize(2);
  set_params(r, {0.25, 0.75});
  r.inputs.mode.value = Regressor::Mode::Test;
  tick(r);

  REQUIRE(r.outputs.output.value.size() == 2);
  CHECK(r.outputs.output.value[0] == Approx(0.25));
  CHECK(r.outputs.output.value[1] == Approx(0.75));
}

TEST_CASE("Regressor: trains synchronously when no worker is bound")
{
  Regressor r;
  r.inputs.parameters_i.ports.resize(1);

  record(r, {0.}, {0.});
  record(r, {1.}, {1.});
  train(r);

  REQUIRE(infer(r, {0.}).size() == 1);
  CHECK(infer(r, {0.})[0] == Approx(0.).margin(0.2));
  CHECK(infer(r, {1.})[0] == Approx(1.).margin(0.2));

  // The MLP interpolates in-between: strictly between the two endpoints.
  const double mid = infer(r, {0.5})[0];
  CHECK(mid > 0.05);
  CHECK(mid < 0.95);
}

TEST_CASE("Regressor: trains through the worker protocol")
{
  Regressor r;
  std::vector<std::function<void(Regressor&)>> jobs;
  r.worker.request = [&](std::vector<rapidLib::trainingExample> set)
  { jobs.push_back(Regressor::worker::work(std::move(set))); };

  r.inputs.parameters_i.ports.resize(1);
  record(r, {0.}, {0.});
  record(r, {1.}, {1.});
  train(r);

  // The training request went to the worker; until its result is applied on
  // the processing thread, there is no model and inference outputs nothing.
  REQUIRE(jobs.size() == 1);
  CHECK(infer(r, {1.}).empty());

  jobs.front()(r); // what the host does after work() completes
  CHECK(infer(r, {1.})[0] == Approx(1.).margin(0.2));
}

TEST_CASE("Regressor: training without examples is a safe no-op")
{
  Regressor r;
  r.inputs.parameters_i.ports.resize(1);

  // This used to reach rapidLib::regression::train, which throws
  // std::length_error("empty training set.").
  train(r);
  CHECK(infer(r, {0.5}).empty());
}

TEST_CASE("Regressor: a constant input feature still trains to finite output")
{
  Regressor r;
  r.inputs.parameters_i.ports.resize(1);

  // First feature never changes: its normalization range is zero. RapidLib's
  // divide-by-zero guard mutated a by-value loop variable, so inference
  // returned NaN and the object silently output nothing forever.
  record(r, {5., 0.}, {0.});
  record(r, {5., 1.}, {1.});
  train(r);

  const auto& out = infer(r, {5., 0.5});
  REQUIRE(out.size() == 1);
  CHECK(out[0] > -1.);
  CHECK(out[0] < 2.);
}

TEST_CASE("Regressor: several output parameters")
{
  Regressor r;
  r.inputs.parameters_i.ports.resize(2);

  record(r, {0.}, {0., 1.});
  record(r, {1.}, {1., 0.});
  train(r);

  const auto& out = infer(r, {0.});
  REQUIRE(out.size() == 2);
  CHECK(out[0] == Approx(0.).margin(0.2));
  CHECK(out[1] == Approx(1.).margin(0.2));
}

TEST_CASE("Regressor: inference with a mismatched input arity is skipped")
{
  Regressor r;
  r.inputs.parameters_i.ports.resize(1);
  record(r, {0., 0.}, {0.});
  record(r, {1., 1.}, {1.});
  train(r);

  const auto before = infer(r, {1., 1.});
  REQUIRE(before.size() == 1);

  // Used to throw std::length_error through the processing thread.
  CHECK(infer(r, {1.}) == before);
  CHECK(infer(r, {1., 1., 1.}) == before);
}
