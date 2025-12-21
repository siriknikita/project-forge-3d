#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "forge_engine/FrameProcessor.hpp"
#include "forge_engine/Model3D.hpp"
#include <memory>

namespace py = pybind11;
using namespace forge_engine;

PYBIND11_MODULE(forge_engine, m) {
    m.doc() = "Forge Engine - High-performance 3D reconstruction pipeline";

    // FrameConfig
    py::class_<FrameConfig>(m, "FrameConfig")
        .def(py::init<>())
        .def_readwrite("width", &FrameConfig::width)
        .def_readwrite("height", &FrameConfig::height)
        .def_readwrite("channels", &FrameConfig::channels);

    // Vertex
    py::class_<Vertex>(m, "Vertex")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vertex::x)
        .def_readwrite("y", &Vertex::y)
        .def_readwrite("z", &Vertex::z)
        .def_readwrite("nx", &Vertex::nx)
        .def_readwrite("ny", &Vertex::ny)
        .def_readwrite("nz", &Vertex::nz)
        .def_readwrite("u", &Vertex::u)
        .def_readwrite("v", &Vertex::v)
        .def_readwrite("r", &Vertex::r)
        .def_readwrite("g", &Vertex::g)
        .def_readwrite("b", &Vertex::b)
        .def_readwrite("a", &Vertex::a);

    // Model3D
    py::class_<Model3D, std::shared_ptr<Model3D>>(m, "Model3D")
        .def(py::init<>())
        .def("addVertex", &Model3D::addVertex)
        .def("addFace", &Model3D::addFace)
        .def("updateVertexPosition", &Model3D::updateVertexPosition)
        .def("updateVertexNormal", &Model3D::updateVertexNormal)
        .def("getVertexCount", &Model3D::getVertexCount)
        .def("getFaceCount", &Model3D::getFaceCount)
        .def("clear", &Model3D::clear)
        .def("exportPLY", &Model3D::exportPLY, 
             py::arg("filename"), py::arg("binary") = true)
        .def("exportOBJ", &Model3D::exportOBJ)
        .def("getStatistics", &Model3D::getStatistics);

    // Model3D::Statistics
    py::class_<Model3D::Statistics>(m, "ModelStatistics")
        .def_readonly("vertex_count", &Model3D::Statistics::vertex_count)
        .def_readonly("face_count", &Model3D::Statistics::face_count)
        .def_readonly("min_x", &Model3D::Statistics::min_x)
        .def_readonly("min_y", &Model3D::Statistics::min_y)
        .def_readonly("min_z", &Model3D::Statistics::min_z)
        .def_readonly("max_x", &Model3D::Statistics::max_x)
        .def_readonly("max_y", &Model3D::Statistics::max_y)
        .def_readonly("max_z", &Model3D::Statistics::max_z);

    // FrameProcessor::Stats
    py::class_<FrameProcessor::Stats>(m, "FrameProcessorStats")
        .def_readonly("frames_processed", &FrameProcessor::Stats::frames_processed)
        .def_readonly("avg_processing_time_ms", &FrameProcessor::Stats::avg_processing_time_ms);

    // FrameProcessor
    py::class_<FrameProcessor, std::shared_ptr<FrameProcessor>>(m, "FrameProcessor")
        .def(py::init<const FrameConfig&>())
        .def("processFrame", [](FrameProcessor& self, py::array_t<uint8_t> frame) {
            // Zero-copy access to numpy array
            py::buffer_info buf = frame.request();
            
            if (buf.ndim != 1) {
                throw std::runtime_error("Frame must be 1D array (flattened)");
            }
            
            // Get raw pointer - zero-copy!
            const uint8_t* data = static_cast<const uint8_t*>(buf.ptr);
            size_t size = buf.size;
            
            self.processFrame(data, size);
        }, py::arg("frame"), "Process frame with zero-copy from numpy array")
        .def("getModel", &FrameProcessor::getModel)
        .def("getStats", &FrameProcessor::getStats)
        .def("reset", &FrameProcessor::reset);
}

