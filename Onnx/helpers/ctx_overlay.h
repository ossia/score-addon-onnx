#pragma once
/* Minimal software-canvas overlay backed by ctx (https://ctx.graphics).
 * Renders 2D primitives directly into an RGBA8888 pixel buffer — no Qt, no GPU.
 * The implementation lives in ctx_overlay.c (compiled as C; ctx.h is a C lib).
 * C++ callers use the thin RAII wrapper in CtxOverlay.hpp. */
#ifdef __cplusplus
extern "C"
{
#endif

  /* Bind a context to an RGBA8888 framebuffer. stride is bytes per row. */
  void* onnx_overlay_new(unsigned char* pixels, int w, int h, int stride);
  /* Rasterize the queued drawing into the buffer and release the context. */
  void onnx_overlay_free(void* ovl);

  /* Current source color for subsequent stroke/fill/text (components 0..1). */
  void onnx_overlay_color(void* ovl, float r, float g, float b, float a);
  void onnx_overlay_line_width(void* ovl, float width);

  /* Each primitive begins its own path, so per-call color changes are honored. */
  void onnx_overlay_line(void* ovl, float x1, float y1, float x2, float y2);
  void onnx_overlay_fill_circle(void* ovl, float x, float y, float radius);
  void onnx_overlay_stroke_rect(void* ovl, float x, float y, float w, float h);
  /* Text baseline at (x,y); uses ctx's built-in font (no external asset). */
  void onnx_overlay_text(void* ovl, float size, float x, float y, const char* s);

#ifdef __cplusplus
}
#endif
