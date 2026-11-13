# Onnx
A new and wonderful [ossia score](https://ossia.io) add-on

## Build dependencies

`OSSIA_USE_SYSTEM_LIBRARIES=ON` uses the installed ONNX Runtime headers and
library, including for the tokenizer extension; it does not download another
runtime or require the official archive's directory layout. If either dependency
is missing, the addon is not built. Without this option, the addon downloads the
platform's prebuilt ONNX Runtime package.

`SCORE_ONNX_ENABLE_EXTENSIONS` defaults to `ON` on native builds and preserves the
QwenLLM and FastVLM objects. The extension's `token_api_only` preset requires
Microsoft GSL 4.0.0 and nlohmann JSON 3.11.3. For network-isolated builds, prefetch
their source trees and pass `FETCHCONTENT_SOURCE_DIR_GSL` and
`FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON` to CMake. The score Flatpak manifest does
this with checksum-pinned archives and shares the single `/app` ONNX Runtime
installation between this addon and sherpa-plugins.
