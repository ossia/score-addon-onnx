#pragma once
#include <Process/LayerView.hpp>

namespace Onnx
{
class View final : public Process::LayerView
{
public:
  explicit View(QGraphicsItem* parent);
  ~View() override;

private:
  void paint_impl(QPainter*) const override;
};
}
