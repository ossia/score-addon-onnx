#include "View.hpp"

#include <Process/Style/ScenarioStyle.hpp>

#include <QPainter>
namespace Onnx
{

View::View(QGraphicsItem* parent)
    : LayerView{parent}
{
}

View::~View() { }

void View::paint_impl(QPainter* painter) const
{
  painter->drawText(boundingRect(), "Change me");
}
}
