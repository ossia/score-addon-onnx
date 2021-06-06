#pragma once
#include <Process/GenericProcessFactory.hpp>

#include <MyProcess/Presenter.hpp>
#include <MyProcess/Process.hpp>
#include <MyProcess/View.hpp>

namespace MyProcess
{
using LayerFactory = Process::
    LayerFactory_T<MyProcess::Model, MyProcess::Presenter, MyProcess::View>;
}
