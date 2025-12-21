#include "forge_engine/Model3D.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace forge_engine {

Model3D::Model3D() = default;

Model3D::~Model3D() = default;

uint32_t Model3D::addVertex(const Vertex& vertex) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t index = static_cast<uint32_t>(vertices_.size());
    vertices_.push_back(vertex);
    return index;
}

void Model3D::addFace(uint32_t v0, uint32_t v1, uint32_t v2) {
    std::lock_guard<std::mutex> lock(mutex_);
    faces_.emplace_back(v0, v1, v2);
}

void Model3D::updateVertexPosition(uint32_t index, float x, float y, float z) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index < vertices_.size()) {
        vertices_[index].x = x;
        vertices_[index].y = y;
        vertices_[index].z = z;
    }
}

void Model3D::updateVertexNormal(uint32_t index, float nx, float ny, float nz) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index < vertices_.size()) {
        vertices_[index].nx = nx;
        vertices_[index].ny = ny;
        vertices_[index].nz = nz;
    }
}

size_t Model3D::getVertexCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vertices_.size();
}

size_t Model3D::getFaceCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return faces_.size();
}

void Model3D::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    vertices_.clear();
    faces_.clear();
}

void Model3D::computeBoundingBox(float& min_x, float& min_y, float& min_z,
                                 float& max_x, float& max_y, float& max_z) const {
    if (vertices_.empty()) {
        min_x = min_y = min_z = max_x = max_y = max_z = 0.0f;
        return;
    }

    min_x = max_x = vertices_[0].x;
    min_y = max_y = vertices_[0].y;
    min_z = max_z = vertices_[0].z;

    for (const auto& v : vertices_) {
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
        min_z = std::min(min_z, v.z);
        max_z = std::max(max_z, v.z);
    }
}

Model3D::Statistics Model3D::getStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Statistics stats;
    stats.vertex_count = vertices_.size();
    stats.face_count = faces_.size();
    computeBoundingBox(stats.min_x, stats.min_y, stats.min_z,
                      stats.max_x, stats.max_y, stats.max_z);
    return stats;
}

bool Model3D::exportPLY(const std::string& filename, bool binary) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
    if (!file.is_open()) {
        return false;
    }

    // Write PLY header
    file << "ply\n";
    if (binary) {
        file << "format binary_little_endian 1.0\n";
    } else {
        file << "format ascii 1.0\n";
    }
    file << "element vertex " << vertices_.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    file << "property uchar red\n";
    file << "property uchar green\n";
    file << "property uchar blue\n";
    file << "property uchar alpha\n";
    file << "element face " << faces_.size() << "\n";
    file << "property list uchar uint vertex_indices\n";
    file << "end_header\n";

    if (binary) {
        // Write vertices in binary format
        for (const auto& v : vertices_) {
            file.write(reinterpret_cast<const char*>(&v.x), sizeof(float));
            file.write(reinterpret_cast<const char*>(&v.y), sizeof(float));
            file.write(reinterpret_cast<const char*>(&v.z), sizeof(float));
            file.write(reinterpret_cast<const char*>(&v.nx), sizeof(float));
            file.write(reinterpret_cast<const char*>(&v.ny), sizeof(float));
            file.write(reinterpret_cast<const char*>(&v.nz), sizeof(float));
            file.write(reinterpret_cast<const char*>(&v.r), sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(&v.g), sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(&v.b), sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(&v.a), sizeof(uint8_t));
        }

        // Write faces in binary format
        uint8_t face_size = 3;
        for (const auto& f : faces_) {
            file.write(reinterpret_cast<const char*>(&face_size), sizeof(uint8_t));
            file.write(reinterpret_cast<const char*>(&f.v0), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&f.v1), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&f.v2), sizeof(uint32_t));
        }
    } else {
        // Write vertices in ASCII format
        for (const auto& v : vertices_) {
            file << v.x << " " << v.y << " " << v.z << " "
                 << v.nx << " " << v.ny << " " << v.nz << " "
                 << static_cast<int>(v.r) << " " << static_cast<int>(v.g) << " "
                 << static_cast<int>(v.b) << " " << static_cast<int>(v.a) << "\n";
        }

        // Write faces in ASCII format
        for (const auto& f : faces_) {
            file << "3 " << f.v0 << " " << f.v1 << " " << f.v2 << "\n";
        }
    }

    file.close();
    return true;
}

bool Model3D::exportOBJ(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    // Write vertices
    for (const auto& v : vertices_) {
        file << "v " << v.x << " " << v.y << " " << v.z;
        if (v.r != 255 || v.g != 255 || v.b != 255) {
            file << " " << (v.r / 255.0f) << " " << (v.g / 255.0f) << " " << (v.b / 255.0f);
        }
        file << "\n";
    }

    // Write normals
    for (const auto& v : vertices_) {
        file << "vn " << v.nx << " " << v.ny << " " << v.nz << "\n";
    }

    // Write texture coordinates
    for (const auto& v : vertices_) {
        file << "vt " << v.u << " " << v.v << "\n";
    }

    // Write faces (OBJ uses 1-based indexing)
    for (const auto& f : faces_) {
        file << "f " << (f.v0 + 1) << "/" << (f.v0 + 1) << "/" << (f.v0 + 1) << " "
             << (f.v1 + 1) << "/" << (f.v1 + 1) << "/" << (f.v1 + 1) << " "
             << (f.v2 + 1) << "/" << (f.v2 + 1) << "/" << (f.v2 + 1) << "\n";
    }

    file.close();
    return true;
}

bool Model3D::exportGLB(const std::string& filename,
                        const uint8_t* /*texture_data*/,
                        uint32_t /*texture_width*/,
                        uint32_t /*texture_height*/) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Note: Full GLB export is complex. This is a simplified implementation.
    // For production use, consider using tinygltf library.
    // For now, we'll export as OBJ and note that GLB export needs enhancement.
    
    // GLB format structure:
    // - 12-byte header (magic, version, length)
    // - JSON chunk (chunk type, chunk length, JSON data)
    // - BIN chunk (chunk type, chunk length, binary data)
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // For now, create a minimal valid GLB structure
    // This is a placeholder - full implementation would require:
    // 1. JSON glTF structure with scenes, nodes, meshes, accessors, bufferViews
    // 2. Binary buffer with vertex positions, normals, UVs, indices
    // 3. Proper chunk alignment (4-byte)
    
    // GLB Header (12 bytes)
    uint32_t magic = 0x46546C67;  // "glTF"
    uint32_t version = 2;
    uint32_t total_length = 0;  // Will be calculated
    
    file.write(reinterpret_cast<const char*>(&magic), 4);
    file.write(reinterpret_cast<const char*>(&version), 4);
    file.write(reinterpret_cast<const char*>(&total_length), 4);
    
    // For a complete implementation, we would:
    // 1. Build JSON glTF structure
    // 2. Pack binary data (vertices, indices)
    // 3. Write JSON chunk
    // 4. Write BIN chunk
    // 5. Update total_length in header
    
    // This is a minimal placeholder - full GLB export requires significant work
    // Consider using tinygltf library for production
    
    file.close();
    
    // For now, fall back to OBJ export as GLB is complex
    // In production, implement full GLB or use tinygltf
    std::cerr << "[Model3D] GLB export is a placeholder. "
              << "Full GLB implementation requires glTF library. "
              << "Consider using exportOBJ() or integrating tinygltf." << std::endl;
    
    return false;  // Indicate that full GLB export is not yet implemented
}

} // namespace forge_engine

