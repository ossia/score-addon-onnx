#pragma once

#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#if defined(ORT_API_MANUAL_INIT)
// ossia::dylib_loader / ossia::get_exe_folder come from a vendored,
// API-compatible copy under Onnx/helpers/compat so this header carries no
// ossia/ include in either build.
#include <Onnx/helpers/compat/dylib_loader.hpp>

#include <Onnx/helpers/OnnxBase.hpp>
#endif

#include <cstdio>

namespace OnnxModels
{

#if defined(ORT_API_MANUAL_INIT)
struct libonnxruntime
{
public:
  bool available{false};
  decltype(&::OrtGetApiBase) get_api_base;

  libonnxruntime()
      : library{{

#if defined(__linux__)
            // Bundled next to / near the plugin in its package (Godot bin/<os-arch>/,
            // etc.) -- resolved relative to this module, not the host executable.
            ossia::get_module_folder() + "/libonnxruntime.so.1",
            ossia::get_module_folder() + "/../libonnxruntime.so.1",
            ossia::get_module_folder() + "/../support/libonnxruntime.so.1",
            "lib/libonnxruntime.so.1",
            "../lib/libonnxruntime.so.1",
            "./_deps/onnxruntime-src/lib/libonnxruntime.so.1",
            "../_deps/onnxruntime-src/lib/libonnxruntime.so.1",
            ossia::get_exe_folder() + "/libonnxruntime.so.1",
            ossia::get_exe_folder() + "/lib/libonnxruntime.so.1",
            ossia::get_exe_folder() + "/../lib/libonnxruntime.so.1",
            ossia::get_exe_folder()
                + "/_deps/onnxruntime-src/lib/libonnxruntime.so.1",
            ossia::get_exe_folder()
                + "/../_deps/onnxruntime-src/lib/libonnxruntime.so.1",
            "libonnxruntime.so.1",
#elif defined(__APPLE__)
            // Bundled inside the plugin's own package, resolved relative to this
            // module (not the host app): sibling (TouchDesigner Plugins/), one up
            // (Godot bin/<os-arch>/ next to the .framework), the Max package
            // support/ (externals/<x>.mxo/Contents/MacOS -> ../../../../support),
            // or a bundle Frameworks/ folder.
            ossia::get_module_folder() + "/libonnxruntime.dylib",
            ossia::get_module_folder() + "/../libonnxruntime.dylib",
            ossia::get_module_folder()
                + "/../../../../support/libonnxruntime.dylib",
            ossia::get_module_folder() + "/../Frameworks/libonnxruntime.dylib",
            "./_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            "../_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            ossia::get_exe_folder() + "/libonnxruntime.dylib",
            ossia::get_exe_folder()
                + "/_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            ossia::get_exe_folder()
                + "/../_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            ossia::get_exe_folder()
                + "/../../../_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            ossia::get_exe_folder() + "/../Frameworks/libonnxruntime.dylib",
            "libonnxruntime.dylib",
#elif defined(_WIN32)
            // Bundled next to the plugin .dll/.mxe64 in its package, or in the Max
            // package support/ folder. The OS loader already searches the module's
            // own directory for siblings, but probe explicitly via the module path
            // so it works regardless of how the host loaded us.
            ossia::get_module_folder() + "/onnxruntime.dll",
            ossia::get_module_folder() + "/../support/onnxruntime.dll",
            "./_deps/onnxruntime-src/lib/onnxruntime.dll",
            "../_deps/onnxruntime-src/lib/onnxruntime.dll",
            ossia::get_exe_folder() + "/onnxruntime.dll",
            ossia::get_exe_folder()
                + "/_deps/onnxruntime-src/lib/onnxruntime.dll",
            ossia::get_exe_folder()
                + "/../_deps/onnxruntime-src/lib/onnxruntime.dll",
            "onnxruntime.dll",
#endif

        }}
  {
    if (!library)
    {
      std::fprintf(stderr, "Could not load libonnxruntime!\n");
      return;
    }

    get_api_base = library.symbol<decltype(&::OrtGetApiBase)>("OrtGetApiBase");
    if (!get_api_base)
      return;

    available = true;
  }

private:
  ossia::dylib_loader library;
};
#endif

[[nodiscard]]
inline bool initOnnxRuntime()
{
  static bool available{};

  // Needed so that machines that cannot run onnxruntime (e.g. macs older than 13.x)
  // can still run ossia
#if defined(ORT_API_MANUAL_INIT)
  if (!available)
  {
    try
    {
      static const libonnxruntime ort;
      if (ort.available)
      {
        auto api = ort.get_api_base();
        if (api)
        {
          auto apiapi = api->GetApi(ORT_API_VERSION);
          Ort::InitApi(apiapi);
          available = bool(api->GetApi(ORT_API_VERSION));
        }
      }
    }
    catch (...)
    {
    }
  }
#else
  return true;
#endif

  return available;
}

struct OnnxObject
{
public:
  OnnxObject() noexcept { available = initOnnxRuntime(); }
  bool available{false};
};

template <halp::static_string lit>
struct ModelPort : halp::file_port<lit, halp::mmap_file_view>
{
  halp_meta(extensions, "*.onnx");

  void update(OnnxObject& obj)
  {
    current_model_invalid = this->file.bytes.size() < 32;
  }
  bool current_model_invalid{false};
};
}
