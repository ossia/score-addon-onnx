#pragma once
#include <score/application/ApplicationContext.hpp>
#include <score/command/Command.hpp>
#include <score/command/CommandGeneratorMap.hpp>
#include <score/plugins/InterfaceList.hpp>
#include <score/plugins/qt_interfaces/CommandFactory_QtInterface.hpp>
#include <score/plugins/qt_interfaces/FactoryFamily_QtInterface.hpp>
#include <score/plugins/qt_interfaces/FactoryInterface_QtInterface.hpp>
#include <score/plugins/qt_interfaces/GUIApplicationPlugin_QtInterface.hpp>
#include <score/plugins/qt_interfaces/PluginRequirements_QtInterface.hpp>

class score_addon_onnx final : public score::Plugin_QtInterface
{
  SCORE_PLUGIN_METADATA(1, "993b0d0f-dc5c-410a-ac3c-17edc791fde0")

public:
  score_addon_onnx();
  ~score_addon_onnx() override;
};
