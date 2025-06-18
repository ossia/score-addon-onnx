#pragma once

#include <ossia/detail/thread.hpp>

#include <halp/file_port.hpp>
#include <halp/meta.hpp>

#if defined(ORT_API_MANUAL_INIT)
#include <ossia/detail/dylib_loader.hpp>

#include <onnxruntime_cxx_api.h>
#endif

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
            "libonnxruntime.so.1",
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
#elif defined(__APPLE__)
            "libonnxruntime.dylib",
            "./_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            "../_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            ossia::get_exe_folder() + "/libonnxruntime.dylib",
            ossia::get_exe_folder()
                + "/_deps/onnxruntime-src/lib/libonnxruntime.dylib",
            ossia::get_exe_folder()
                + "/../_deps/onnxruntime-src/lib/libonnxruntime.dylib",
#elif defined(_WIN32)
            "libonnxruntime.dll",
            "./_deps/onnxruntime-src/lib/libonnxruntime.dll",
            "../_deps/onnxruntime-src/lib/libonnxruntime.dll",
            ossia::get_exe_folder() + "/libonnxruntime.dll",
            ossia::get_exe_folder()
                + "/_deps/onnxruntime-src/lib/libonnxruntime.dll",
            ossia::get_exe_folder()
                + "/../_deps/onnxruntime-src/lib/libonnxruntime.dll",
#endif

        }}
  {
    if (!library)
    {
      qDebug("Could not load libonnxruntime!");
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
  // Needed so that machines that cannot run onnxruntime (e.g. macs older than 13.x)
  // can still run ossia
#if defined(ORT_API_MANUAL_INIT)
  if (!Ort::Global<void>::api_)
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
          return Ort::Global<void>::api_;
        }
      }
    }
    catch (...)
    {
    }
    return false;
  }
#endif
  return true;
}

struct OnnxObject
{
public:
  OnnxObject() noexcept { available = initOnnxRuntime(); }
  bool available{false};
};

struct ModelPort : halp::file_port<"Model", halp::mmap_file_view> {
  halp_meta(extensions, "*.onnx");
};
}
