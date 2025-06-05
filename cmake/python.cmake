find_program(UV_PROGRAM uv REQUIRED)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/.venv")
execute_process(
    COMMAND "${UV_PROGRAM}" venv
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
endif()

set(Python_ROOT_DIR  "${CMAKE_BINARY_DIR}/.venv")
add_subdirectory("${OSSIA_3RDPARTY_FOLDER}/pybind11" pybind11)
