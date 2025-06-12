#include "Classifier.hpp"

#include "src/classification.h"

#include <ossia/math/safe_math.hpp>

namespace RapidlibModels
{
Classifier::Classifier() noexcept { }
Classifier::~Classifier() { }
void Classifier::operator()()
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
    rapidLib::trainingExample ex;
    ex.input = inputs.input.value;
    ex.output.reserve(inputs.parameters_i.ports.size());
    for (auto& port : inputs.parameters_i.ports)
    {
      ex.output.push_back(port.value);
    }
    m_trainingSet.push_back(std::move(ex));
  }

  if (inputs.train)
  {
    m_trained = m_model.train(m_trainingSet);
  }

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
    if (m_trained)
    {
      outputs.output.value = m_model.run(inputs.input.value);
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
