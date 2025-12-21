cmake_minimum_required(VERSION 3.20)
project(forge_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# M4-specific optimizations
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.5-a -mcpu=apple-m4")
    message(STATUS "Building with M4 optimizations")
endif()

# Find required packages
# Find pybind11 - it should be installed via pip first
# pybind11 provides a CMake config when installed via pip
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Try to find uv first, then regular python
    find_program(UV_EXECUTABLE uv)
    
    if(UV_EXECUTABLE)
        # Try with uv run python (run from server directory where dependencies are installed)
        get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
        set(SERVER_DIR "${PROJECT_ROOT}/server")
        if(EXISTS "${SERVER_DIR}/requirements.txt")
            execute_process(
                COMMAND ${UV_EXECUTABLE} run python -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE pybind11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                WORKING_DIRECTORY ${SERVER_DIR}
            )
        endif()
    endif()
    
    if(NOT pybind11_CMAKE_DIR)
        # Fall back to regular python3
        execute_process(
            COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    
    if(pybind11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKE_DIR}")
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR 
            "pybind11 not found. Please install it first:\n"
            "  cd ../server && uv pip install -r requirements.txt\n"
            "  or\n"
            "  pip install pybind11\n\n"
            "Then run cmake again."
        )
    endif()
endif()

# Accelerate is an Apple framework, not a CMake package
# We'll link it directly
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "Found Accelerate framework: ${ACCELERATE_FRAMEWORK}")
    else()
        message(FATAL_ERROR "Accelerate framework not found")
    endif()
else()
    message(FATAL_ERROR "Accelerate framework is only available on macOS")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Source files
set(SOURCES
    src/FrameProcessor.cpp
    src/Model3D.cpp
    src/pybind_module.cpp
)

# Header files
set(HEADERS
    include/forge_engine/FrameProcessor.hpp
    include/forge_engine/CircularBuffer.hpp
    include/forge_engine/Model3D.hpp
)

# Create shared library
pybind11_add_module(forge_engine ${SOURCES} ${HEADERS})

# Link Accelerate framework
if(APPLE)
    target_link_libraries(forge_engine PRIVATE
        ${ACCELERATE_FRAMEWORK}
    )
endif()

# Compiler-specific options
target_compile_options(forge_engine PRIVATE
    -Wall
    -Wextra
    -O3
    -ffast-math
)

# Set output directory
set_target_properties(forge_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

