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

# Find Python executable - prefer uv run python if available
# This ensures we build for the same Python version that runs the server
get_filename_component(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(SERVER_DIR "${PROJECT_ROOT}/server")

find_program(UV_EXECUTABLE uv)
if(UV_EXECUTABLE AND EXISTS "${SERVER_DIR}/requirements.txt")
    # Get Python executable from uv run python
    execute_process(
        COMMAND ${UV_EXECUTABLE} run python -c "import sys; print(sys.executable)"
        OUTPUT_VARIABLE UV_PYTHON_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        WORKING_DIRECTORY ${SERVER_DIR}
    )
    if(UV_PYTHON_EXECUTABLE)
        set(Python3_EXECUTABLE "${UV_PYTHON_EXECUTABLE}")
        message(STATUS "Using Python from uv: ${Python3_EXECUTABLE}")
    endif()
endif()

# Find Python3 with all components needed for pybind11
# Use the executable we found if available
find_package(Python3 COMPONENTS Interpreter Development NumPy)
if(Python3_FOUND)
    message(STATUS "Found Python ${Python3_VERSION} at ${Python3_EXECUTABLE}")
    message(STATUS "Python libraries: ${Python3_LIBRARIES}")
else()
    message(WARNING "Python3 not found, pybind11 may use wrong Python version")
endif()

# Tell pybind11 to use FindPython instead of deprecated FindPythonInterp
set(PYBIND11_FINDPYTHON ON)

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

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR 
        "OpenCV not found. Please install it first:\n"
        "  brew install opencv\n"
        "  or\n"
        "  Download from https://opencv.org/releases/\n\n"
        "Then run cmake again."
    )
endif()

# Find Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3 ${EIGEN3_VERSION_STRING}")
    message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR 
        "Eigen3 not found. Please install it first:\n"
        "  brew install eigen\n"
        "  or\n"
        "  Download from https://eigen.tuxfamily.org/\n\n"
        "Then run cmake again."
    )
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
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# Source files
set(SOURCES
    src/FrameProcessor.cpp
    src/Model3D.cpp
    src/DepthBuffer.cpp
    src/pybind_module.cpp
)

# Header files
set(HEADERS
    include/forge_engine/FrameProcessor.hpp
    include/forge_engine/CircularBuffer.hpp
    include/forge_engine/Model3D.hpp
    include/forge_engine/DepthBuffer.hpp
)

# Create shared library
pybind11_add_module(forge_engine ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(forge_engine PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

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

