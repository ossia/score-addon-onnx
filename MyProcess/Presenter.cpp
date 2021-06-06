#include <score/command/Dispatchers/CommandDispatcher.hpp>

#include <MyProcess/Presenter.hpp>
#include <MyProcess/Process.hpp>
#include <MyProcess/View.hpp>

namespace MyProcess
{
Presenter::Presenter(
    const Model& layer,
    View* view,
    const Process::Context& ctx,
    QObject* parent)
    : Process::LayerPresenter{layer, view, ctx, parent}
    , m_model{layer}
    , m_view{view}
{
}

void Presenter::setWidth(qreal val, qreal defaultWidth)
{
  m_view->setWidth(val);
}

void Presenter::setHeight(qreal val)
{
  m_view->setHeight(val);
}

void Presenter::putToFront()
{
  m_view->setOpacity(1);
}

void Presenter::putBehind()
{
  m_view->setOpacity(0.2);
}

void Presenter::on_zoomRatioChanged(ZoomRatio) { }

void Presenter::parentGeometryChanged() { }

}
