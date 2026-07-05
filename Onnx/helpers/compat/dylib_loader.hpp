#pragma once
// Vendored, self-contained copy of ossia::dylib_loader / ossia::get_exe_folder
// (from libossia, ossia/detail/dylib_loader.hpp + ossia/detail/thread.hpp).
//
// When libossia is on the include path (the score build) we MUST use the real
// headers: ossia's get_exe_folder() has external linkage (OSSIA_EXPORT, defined
// in thread.cpp), so redefining it here -- even inline -- yields a
// "duplicate symbol: ossia::get_exe_folder()" at link time. The vendored copies
// below are only for standalone back-ends where ossia is unavailable.
#if __has_include(<ossia/detail/dylib_loader.hpp>)
#define OSSIA_ONNX_HAS_REAL_OSSIA 1
#include <ossia/detail/dylib_loader.hpp>
#include <ossia/detail/thread.hpp>
#else
#define OSSIA_ONNX_HAS_REAL_OSSIA 0
#endif

#if !OSSIA_ONNX_HAS_REAL_OSSIA
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
// Declares _NSGetExecutablePath with C linkage. Declaring it ourselves needs a
// linkage-specification (extern "C"), which is only legal at namespace scope --
// not inside the get_exe_folder() body -- so use the system header instead.
#include <mach-o/dyld.h>
#endif
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace ossia
{
class dylib_loader
{
public:
  explicit dylib_loader(const char* const so)
  {
#ifdef _WIN32
    impl = (void*)LoadLibraryA(so);
#else
    impl = dlopen(so, RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
#endif
    if(!impl)
    {
      throw std::runtime_error(std::string(so) + ": not found. ");
    }
  }

  explicit dylib_loader(std::vector<std::string_view> sos)
  {
    if(sos.empty())
      throw std::runtime_error("No shared object specified");

    for(const auto so : sos)
    {
#ifdef _WIN32
      impl = (void*)LoadLibraryA(so.data());
#else
      impl = dlopen(so.data(), RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
#endif
      if(impl)
        return;
    }

    throw std::runtime_error(std::string(sos[0]) + ": not found. ");
  }

  dylib_loader(const dylib_loader&) noexcept = delete;
  dylib_loader& operator=(const dylib_loader&) noexcept = delete;
  dylib_loader(dylib_loader&& other) noexcept
  {
    impl = other.impl;
    other.impl = nullptr;
  }

  dylib_loader& operator=(dylib_loader&& other) noexcept
  {
    impl = other.impl;
    other.impl = nullptr;
    return *this;
  }

  ~dylib_loader()
  {
    if(impl)
    {
#ifdef _WIN32
      FreeLibrary((HMODULE)impl);
#else
      dlclose(impl);
#endif
    }
  }

  template <typename T>
  T symbol(const char* const sym) const noexcept
  {
#ifdef _WIN32
    return (T)GetProcAddress((HMODULE)impl, sym);
#else
    return (T)dlsym(impl, sym);
#endif
  }

  operator bool() const { return bool(impl); }

private:
  void* impl{};
};

// ossia::get_exe_folder() — the directory of the running executable. ossia's
// real implementation lives in libossia's .cpp; the standalone back-ends only
// need it to probe a handful of relative paths next to the binary, so provide
// a small header-only implementation here.
inline std::string get_exe_folder()
{
  std::string path;
#if defined(_WIN32)
  char buf[32768];
  DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
  path.assign(buf, n);
#elif defined(__APPLE__)
  char buf[16384];
  uint32_t size = sizeof(buf);
  if(_NSGetExecutablePath(buf, &size) == 0)
    path = buf;
#else
  char buf[16384];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf));
  if(n > 0)
    path.assign(buf, (std::size_t)n);
#endif
  auto pos = path.find_last_of("/\\");
  if(pos != std::string::npos)
    path.resize(pos);
  return path;
}
} // namespace ossia
#endif // !OSSIA_ONNX_HAS_REAL_OSSIA

// ossia::get_module_folder() — the directory of the shared library / plugin that
// contains THIS code, as opposed to the host executable. When loaded as a Max /
// TouchDesigner / Godot plugin, get_exe_folder() returns the host app's folder,
// not the plugin's -- so resources bundled inside the plugin's own package (e.g.
// libonnxruntime in a support/ or sibling folder) can't be found relative to it.
// dladdr / GetModuleHandleEx resolve the address of a function back to the module
// file it lives in. Falls back to get_exe_folder() if resolution fails (e.g. a
// fully static link, where this code is in the main executable anyway).
//
// ossia has no equivalent, so this addon-only helper is defined unconditionally
// (it never collides with libossia).
namespace ossia
{
inline std::string get_module_folder()
{
  std::string path;
#if defined(_WIN32)
  HMODULE hm = nullptr;
  if(GetModuleHandleExA(
         GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
             | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
         reinterpret_cast<LPCSTR>(&get_module_folder), &hm)
     != 0)
  {
    char buf[32768];
    DWORD n = GetModuleFileNameA(hm, buf, sizeof(buf));
    path.assign(buf, n);
  }
#else
  Dl_info info{};
  if(dladdr(reinterpret_cast<const void*>(&get_module_folder), &info)
     && info.dli_fname)
    path = info.dli_fname;
#endif
  if(path.empty())
    return get_exe_folder();
  auto pos = path.find_last_of("/\\");
  if(pos != std::string::npos)
    path.resize(pos);
  return path;
}
}
