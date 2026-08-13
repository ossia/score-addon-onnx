#include "Regressor.hpp"

#include "src/regression.h"

#include <Onnx/helpers/compat/safe_math.hpp>

#include <algorithm>

namespace RapidlibModels
{
Regressor::Regressor() noexcept { }
Regressor::~Regressor() { }

std::function<void(Regressor&)>
Regressor::worker::work(std::vector<rapidLib::trainingExample> trainingSet)
{
  auto model = std::make_shared<rapidLib::regression>();
  bool ok = false;
  try
  {
    ok = model->train(trainingSet);
  }
  catch (...)
  {
    // rapidLib throws std::length_error on empty sets and on examples with
    // mismatched arity; treat any of those as a failed training.
    ok = false;
  }

  const std::size_t numInputs
      = trainingSet.empty() ? 0 : trainingSet[0].input.size();
  return [model = std::move(model), ok, numInputs](Regressor& self)
  {
    self.m_model = std::move(model);
    self.m_trained = ok;
    self.m_numInputs = numInputs;
  };
}

void Regressor::operator()()
{
  if (inputs.undo)
  {
    if (!m_trainingSet.empty())
      m_trainingSet.pop_back();
  }

  if (inputs.reset)
  {
    m_trainingSet.clear();
  }

  if (inputs.record)
  {
    // An example needs a non-empty feature vector, and every example must
    // agree on input & output arity: rapidLib's train() throws otherwise.
    auto& in = inputs.input.value;
    const bool arity_ok
        = !in.empty()
          && (m_trainingSet.empty()
              || (in.size() == m_trainingSet[0].input.size()
                  && inputs.parameters_i.ports.size()
                         == m_trainingSet[0].output.size()));
    if (arity_ok)
    {
      rapidLib::trainingExample ex;
      ex.input = in;
      ex.output.reserve(inputs.parameters_i.ports.size());
      for (auto& port : inputs.parameters_i.ports)
      {
        ex.output.push_back(port.value);
      }
      m_trainingSet.push_back(std::move(ex));
    }
  }

  if (inputs.train && !m_trainingSet.empty())
  {
    if (worker.request)
      worker.request(m_trainingSet);
    else // hosts without a worker thread pool: train synchronously
      worker.work(m_trainingSet)(*this);
  }

  outputs.examples.value = m_trainingSet.size();

  if (inputs.mode == Mode::Test)
  {
    outputs.output.value.clear();
    for (auto& port : inputs.parameters_i.ports)
    {
      outputs.output.value.push_back(port.value);
    }
  }
  else
  {
    // rapidLib::modelSet::run() throws on input arity mismatch, so only
    // run the model on inputs shaped like the ones it was trained with;
    // a transient arity mismatch keeps the last good prediction.
    if (!m_trained || !m_model)
    {
      outputs.output.value.clear();
    }
    else if (inputs.input.value.size() == m_numInputs)
    {
      outputs.output.value = m_model->run(inputs.input.value);
      if (std::any_of(
              outputs.output.value.begin(),
              outputs.output.value.end(),
              [](auto x)
              { return ossia::safe_isnan(x) || ossia::safe_isinf(x); }))
        outputs.output.value.clear();
    }
  }
}
}
