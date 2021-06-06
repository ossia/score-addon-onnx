#include "LocalTree.hpp"

#include <LocalTree/Property.hpp>

#include <MyProcess/Process.hpp>

namespace MyProcess
{
LocalTreeProcessComponent::LocalTreeProcessComponent(
    ossia::net::node_base& parent,
    MyProcess::Model& proc,
    const score::DocumentContext& sys,
    QObject* parent_obj)
    : LocalTree::ProcessComponent_T<MyProcess::Model>{
        parent,
        proc,
        sys,
        "MyProcessComponent",
        parent_obj}
{
}

LocalTreeProcessComponent::~LocalTreeProcessComponent() { }
}
