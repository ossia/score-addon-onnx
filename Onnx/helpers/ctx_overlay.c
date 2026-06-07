/* ctx-backed overlay shim. Compiled as C (ctx.h does not build as C++) with
 * CTX_RASTERIZER=1 so the in-memory framebuffer rasterizer is enabled (it is
 * otherwise gated behind a windowing backend). See CMakeLists.txt. */
#define CTX_IMPLEMENTATION
#include "ctx.h"

#include "ctx_overlay.h"

void* onnx_overlay_new(unsigned char* pixels, int w, int h, int stride)
{
  return ctx_new_for_framebuffer(pixels, w, h, stride, CTX_FORMAT_RGBA8);
}

void onnx_overlay_free(void* o)
{
  if(o)
    ctx_free((Ctx*)o);
}

void onnx_overlay_color(void* o, float r, float g, float b, float a)
{
  ctx_rgba((Ctx*)o, r, g, b, a);
}

void onnx_overlay_line_width(void* o, float width)
{
  ctx_line_width((Ctx*)o, width);
}

void onnx_overlay_line(void* o, float x1, float y1, float x2, float y2)
{
  Ctx* c = (Ctx*)o;
  ctx_begin_path(c);
  ctx_move_to(c, x1, y1);
  ctx_line_to(c, x2, y2);
  ctx_stroke(c);
}

void onnx_overlay_fill_circle(void* o, float x, float y, float radius)
{
  Ctx* c = (Ctx*)o;
  ctx_begin_path(c);
  ctx_arc(c, x, y, radius, 0.0f, 6.2831853f, 0);
  ctx_fill(c);
}

void onnx_overlay_stroke_rect(void* o, float x, float y, float w, float h)
{
  Ctx* c = (Ctx*)o;
  ctx_begin_path(c);
  ctx_rectangle(c, x, y, w, h);
  ctx_stroke(c);
}

void onnx_overlay_text(void* o, float size, float x, float y, const char* s)
{
  Ctx* c = (Ctx*)o;
  ctx_font_size(c, size);
  ctx_move_to(c, x, y);
  ctx_text(c, s);
}
