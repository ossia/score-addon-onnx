#pragma once

#include <cstdlib>
#include <version>

#if !defined(_MSC_VER)

#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#define _In_
#define _In_z_
#define _In_opt_
#define _In_reads_(x)
#define _In_reads_opt_(x)
#define _Inout_
#define _Inout_bytecount_(x)
#define _Inout_opt_
#define _Inout_updates_opt_(x)
#define _Out_
#define _Out_opt_
#define _Out_writes_(x)
#define _Out_writes_bytes_opt_(x)
#define _Out_writes_opt_(x)
#define _Out_writes_to_(x, y)
#define _When_(x, y)
#define _Frees_ptr_opt_
#define _Post_ptr_invalid_
#define _Check_return_opt_
#define _Printf_format_string_
#define _Success_(x)
#endif
#endif

#include <onnxruntime_cxx_api.h>
#if __APPLE__
#include <coreml_provider_factory.h>
#endif
