#pragma once
// Vendored, self-contained copy of ossia::dylib_loader (from libossia,
// ossia/detail/dylib_loader.hpp). Used only for standalone builds, where
// libossia is not on the include path; the score build picks the real header
// via __has_include in Utils.hpp. Keep API-compatible with ossia.
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
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
  extern int _NSGetExecutablePath(char*, uint32_t*);
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
}
