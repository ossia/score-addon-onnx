# Resolves the CUDA runtime libraries that onnxruntime's prebuilt CUDA execution
# provider needs at load time, so they can be shipped next to it.
#
# Microsoft's onnxruntime-win-x64-gpu archive contains the provider but none of
# the libraries it links, so the CUDA EP only ever loads on a machine that
# already has a matching CUDA install on PATH. Everywhere else onnxruntime logs
#
#   Error loading "onnxruntime_providers_cuda.dll" which depends on
#   "cublasLt64_13.dll" which is missing. (Error 126)
#
# and silently falls back to the CPU provider.
#
# The set below is the import table of onnxruntime_providers_cuda.dll, not a
# guess: for 1.27.1 (the gpu_cuda13 archive) it is cublas64_13, cublasLt64_13,
# cufft64_12 and cudnn64_9. The CUDA runtime proper (cudart) is *not* in it --
# the CUDA 13 math libraries and the provider link it statically -- so shipping
# cudart, nvrtc, nvJitLink, npp or curand would be dead weight. Re-derive the
# list with `objdump -p onnxruntime_providers_cuda.dll` when bumping the pin.
#
# cudnn64_9 is only a dispatch shim: it LoadLibrary's cudnn_graph64_9,
# cudnn_ops64_9, cudnn_cnn64_9, cudnn_adv64_9, cudnn_ext64_9 and the engine
# libraries on demand, which is why the whole cuDNN bin/ directory ships rather
# than the one DLL the import table names.
#
# cuDNN must be the build for the same CUDA major as the onnxruntime archive
# (cuda13 here); NVIDIA ships cuda11/cuda12/cuda13 variants of every cuDNN 9.x.
#
# Sets ONNXRUNTIME_CUDA_RUNTIME_FILES (absolute paths) and
# ONNXRUNTIME_CUDA_RUNTIME_MISSING (DLL names that could not be found).

set(ONNXRUNTIME_CUDA_RUNTIME_FILES "")
set(ONNXRUNTIME_CUDA_RUNTIME_MISSING "")

if(WIN32)
  set(_ort_cuda_libs cublas64_13 cublasLt64_13 cufft64_12)
  set(_ort_cuda_suffix ".dll")
  set(_ort_cudnn_glob "cudnn*.dll")
else()
  set(_ort_cuda_libs libcublas.so.13 libcublasLt.so.13 libcufft.so.12)
  set(_ort_cuda_suffix "")
  set(_ort_cudnn_glob "libcudnn*.so.9*")
endif()

foreach(_lib IN LISTS _ort_cuda_libs)
  unset(_ort_found CACHE)
  find_file(_ort_found "${_lib}${_ort_cuda_suffix}"
    HINTS ${ONNXRUNTIME_CUDA_RUNTIME_DIRS}
    PATH_SUFFIXES bin bin/x64 lib lib64 targets/x86_64-linux/lib)
  if(_ort_found)
    list(APPEND ONNXRUNTIME_CUDA_RUNTIME_FILES "${_ort_found}")
  else()
    list(APPEND ONNXRUNTIME_CUDA_RUNTIME_MISSING "${_lib}${_ort_cuda_suffix}")
  endif()
endforeach()

# Take every sibling of the cuDNN dispatch shim: the backends it opens by name
# are not in anyone's import table and their set changes between cuDNN releases.
if(WIN32)
  set(_ort_cudnn_shim "cudnn64_9.dll")
else()
  set(_ort_cudnn_shim "libcudnn.so.9")
endif()

unset(_ort_cudnn CACHE)
find_file(_ort_cudnn "${_ort_cudnn_shim}"
  HINTS ${ONNXRUNTIME_CUDA_RUNTIME_DIRS}
  PATH_SUFFIXES bin bin/x64 lib lib64 targets/x86_64-linux/lib)
if(_ort_cudnn)
  get_filename_component(_ort_cudnn_dir "${_ort_cudnn}" DIRECTORY)
  file(GLOB _ort_cudnn_all "${_ort_cudnn_dir}/${_ort_cudnn_glob}")
  list(APPEND ONNXRUNTIME_CUDA_RUNTIME_FILES ${_ort_cudnn_all})
else()
  list(APPEND ONNXRUNTIME_CUDA_RUNTIME_MISSING "${_ort_cudnn_shim}")
endif()

list(REMOVE_DUPLICATES ONNXRUNTIME_CUDA_RUNTIME_FILES)
