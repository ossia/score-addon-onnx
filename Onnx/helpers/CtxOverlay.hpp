#pragma once
/* Thin C++ RAII wrapper over the ctx overlay shim (ctx_overlay.h). Keeps the
 * heavy ctx.h confined to ctx_overlay.c; here we only depend on the tiny C API
 * plus Qt's color/geometry value types (kept for now — full Qt-type removal is
 * a later step). Renders straight into an RGBA8888 buffer; no QPainter/GPU. */
#include <Onnx/helpers/CoreTypes.hpp>
#include <Onnx/helpers/ctx_overlay.h>

#include <QPointF>
#include <QRectF>
#include <QString>

namespace OnnxModels
{
struct Overlay
{
  void* c{};

  Overlay(unsigned char* pixels, int w, int h)
      : c(onnx_overlay_new(pixels, w, h, w * 4))
  {
  }
  ~Overlay() { onnx_overlay_free(c); } // rasterizes into the buffer
  Overlay(const Overlay&) = delete;
  Overlay& operator=(const Overlay&) = delete;

  void color(Onnx::Rgba q) { onnx_overlay_color(c, q.r, q.g, q.b, q.a); }
  void lineWidth(float w) { onnx_overlay_line_width(c, w); }
  void line(QPointF a, QPointF b)
  {
    onnx_overlay_line(
        c, static_cast<float>(a.x()), static_cast<float>(a.y()),
        static_cast<float>(b.x()), static_cast<float>(b.y()));
  }
  void fillCircle(QPointF p, float r)
  {
    onnx_overlay_fill_circle(
        c, static_cast<float>(p.x()), static_cast<float>(p.y()), r);
  }
  void strokeRect(QRectF r)
  {
    onnx_overlay_stroke_rect(
        c, static_cast<float>(r.x()), static_cast<float>(r.y()),
        static_cast<float>(r.width()), static_cast<float>(r.height()));
  }
  void text(float size, QPointF p, const QString& s)
  {
    onnx_overlay_text(
        c, size, static_cast<float>(p.x()), static_cast<float>(p.y()),
        s.toUtf8().constData());
  }
};
} // namespace OnnxModels
