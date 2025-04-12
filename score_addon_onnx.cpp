#include "score_addon_onnx.hpp"

score_addon_onnx::score_addon_onnx() { }

score_addon_onnx::~score_addon_onnx() { }

#include <score/plugins/PluginInstances.hpp>
SCORE_EXPORT_PLUGIN(score_addon_onnx)
