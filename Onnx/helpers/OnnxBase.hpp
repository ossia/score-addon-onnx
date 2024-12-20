#pragma once

#include <cstdlib>
#include <version>

#if !defined(_MSC_VER)

#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif
#endif

#include <onnxruntime_cxx_api.h>
#if __APPLE__
#include <coreml_provider_factory.h>
#endif
