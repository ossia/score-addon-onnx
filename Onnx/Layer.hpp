#pragma once
#include <Process/GenericProcessFactory.hpp>

#include <Onnx/Presenter.hpp>
#include <Onnx/Process.hpp>
#include <Onnx/View.hpp>

namespace Onnx
{
using LayerFactory = Process::
    LayerFactory_T<Onnx::Model, Onnx::Presenter, Onnx::View>;
}
