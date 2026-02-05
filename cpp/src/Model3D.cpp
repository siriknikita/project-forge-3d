#include "forge_engine/Model3D.hpp"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <limits>

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

float Model3D::distanceSquared(const Vertex& v1, const Vertex& v2) const {
    float dx = v1.x - v2.x;
    float dy = v1.y - v2.y;
    float dz = v1.z - v2.z;
    return dx * dx + dy * dy + dz * dz;
}

float Model3D::triangleArea(const Vertex& v0, const Vertex& v1, const Vertex& v2) const {
    // Compute triangle area using cross product
    float ax = v1.x - v0.x;
    float ay = v1.y - v0.y;
    float az = v1.z - v0.z;
    
    float bx = v2.x - v0.x;
    float by = v2.y - v0.y;
    float bz = v2.z - v0.z;
    
    // Cross product
    float cx = ay * bz - az * by;
    float cy = az * bx - ax * bz;
    float cz = ax * by - ay * bx;
    
    // Area = 0.5 * |cross product|
    return 0.5f * std::sqrt(cx * cx + cy * cy + cz * cz);
}

bool Model3D::isValidTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                               float max_edge_length, float min_area) const {
    // Check edge lengths
    float edge01_sq = distanceSquared(v0, v1);
    float edge12_sq = distanceSquared(v1, v2);
    float edge20_sq = distanceSquared(v2, v0);
    
    float max_edge_sq = max_edge_length * max_edge_length;
    if (edge01_sq > max_edge_sq || edge12_sq > max_edge_sq || edge20_sq > max_edge_sq) {
        return false;
    }
    
    // Check triangle area (avoid degenerate triangles)
    float area = triangleArea(v0, v1, v2);
    if (area < min_area) {
        return false;
    }
    
    // Check for consistent normal (avoid flipped triangles)
    // Compute triangle normal
    float ax = v1.x - v0.x;
    float ay = v1.y - v0.y;
    float az = v1.z - v0.z;
    
    float bx = v2.x - v0.x;
    float by = v2.y - v0.y;
    float bz = v2.z - v0.z;
    
    float nx = ay * bz - az * by;
    float ny = az * bx - ax * bz;
    float nz = ax * by - ay * bx;
    
    float len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-6f) {
        return false;  // Degenerate triangle
    }
    
    nx /= len;
    ny /= len;
    nz /= len;
    
    // Check consistency with vertex normals (if they exist)
    float dot0 = nx * v0.nx + ny * v0.ny + nz * v0.nz;
    float dot1 = nx * v1.nx + ny * v1.ny + nz * v1.nz;
    float dot2 = nx * v2.nx + ny * v2.ny + nz * v2.nz;
    
    // Allow some tolerance for normal consistency
    float normal_threshold = 0.3f;  // ~73 degrees
    if (dot0 < normal_threshold && dot1 < normal_threshold && dot2 < normal_threshold) {
        return false;  // Triangle normal inconsistent with vertex normals
    }
    
    return true;
}

size_t Model3D::generateMesh(float max_edge_length, float cell_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clear existing faces
    faces_.clear();
    
    if (vertices_.size() < 3) {
        return 0;  // Need at least 3 vertices for a triangle
    }
    
    // Compute bounding box for spatial hashing
    float min_x, min_y, min_z, max_x, max_y, max_z;
    computeBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    
    // Initialize spatial hash
    // Use a 3D grid to partition space
    int grid_x = static_cast<int>(std::ceil((max_x - min_x) / cell_size)) + 1;
    int grid_y = static_cast<int>(std::ceil((max_y - min_y) / cell_size)) + 1;
    int grid_z = static_cast<int>(std::ceil((max_z - min_z) / cell_size)) + 1;
    
    // Limit grid size to avoid excessive memory usage
    const int MAX_GRID_SIZE = 1000;
    if (grid_x > MAX_GRID_SIZE || grid_y > MAX_GRID_SIZE || grid_z > MAX_GRID_SIZE) {
        // Adjust cell size to fit within limits
        float max_dim = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
        cell_size = max_dim / MAX_GRID_SIZE;
        grid_x = static_cast<int>(std::ceil((max_x - min_x) / cell_size)) + 1;
        grid_y = static_cast<int>(std::ceil((max_y - min_y) / cell_size)) + 1;
        grid_z = static_cast<int>(std::ceil((max_z - min_z) / cell_size)) + 1;
    }
    
    // Hash function for 3D grid
    auto hash3D = [grid_x, grid_y, grid_z](int x, int y, int z) -> size_t {
        return static_cast<size_t>(x + grid_x * (y + grid_y * z));
    };
    
    // Build spatial hash: cell -> list of vertex indices
    std::unordered_map<size_t, std::vector<uint32_t>> spatial_hash;
    
    for (uint32_t i = 0; i < vertices_.size(); ++i) {
        const auto& v = vertices_[i];
        int cell_x = static_cast<int>((v.x - min_x) / cell_size);
        int cell_y = static_cast<int>((v.y - min_y) / cell_size);
        int cell_z = static_cast<int>((v.z - min_z) / cell_size);
        
        size_t cell_hash = hash3D(cell_x, cell_y, cell_z);
        spatial_hash[cell_hash].push_back(i);
    }
    
    // For each vertex, find neighbors and form triangles
    const int k_neighbors = 8;  // Target number of neighbors
    const float min_area = 1e-6f;  // Minimum triangle area
    
    // Track edges to avoid duplicate triangles
    std::unordered_set<Edge, EdgeHash> processed_edges;
    
    size_t faces_added = 0;
    
    for (uint32_t vidx = 0; vidx < vertices_.size(); ++vidx) {
        const auto& vertex = vertices_[vidx];
        
        // Find cell for this vertex
        int cell_x = static_cast<int>((vertex.x - min_x) / cell_size);
        int cell_y = static_cast<int>((vertex.y - min_y) / cell_size);
        int cell_z = static_cast<int>((vertex.z - min_z) / cell_size);
        
        // Collect neighbors from current cell and adjacent cells
        std::vector<std::pair<uint32_t, float>> candidates;  // (vertex_index, distance_squared)
        
        // Check current cell and 26 adjacent cells (3x3x3 neighborhood)
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    int nx = cell_x + dx;
                    int ny = cell_y + dy;
                    int nz = cell_z + dz;
                    
                    if (nx < 0 || ny < 0 || nz < 0) continue;
                    
                    size_t cell_hash = hash3D(nx, ny, nz);
                    auto it = spatial_hash.find(cell_hash);
                    if (it == spatial_hash.end()) continue;
                    
                    for (uint32_t nidx : it->second) {
                        if (nidx == vidx) continue;  // Skip self
                        
                        float dist_sq = distanceSquared(vertex, vertices_[nidx]);
                        float max_dist_sq = max_edge_length * max_edge_length;
                        
                        if (dist_sq <= max_dist_sq) {
                            candidates.push_back({nidx, dist_sq});
                        }
                    }
                }
            }
        }
        
        // Sort by distance and take k nearest
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        if (candidates.size() < 2) continue;
        
        // Take up to k_neighbors
        size_t num_neighbors = std::min(static_cast<size_t>(k_neighbors), candidates.size());
        
        // Form triangles with pairs of neighbors
        for (size_t i = 0; i < num_neighbors; ++i) {
            for (size_t j = i + 1; j < num_neighbors; ++j) {
                uint32_t n1_idx = candidates[i].first;
                uint32_t n2_idx = candidates[j].first;
                
                const auto& n1 = vertices_[n1_idx];
                const auto& n2 = vertices_[n2_idx];
                
                // Check if triangle is valid
                if (!isValidTriangle(vertex, n1, n2, max_edge_length, min_area)) {
                    continue;
                }
                
                // Create edges and check for duplicates
                Edge e1(vidx, n1_idx);
                Edge e2(n1_idx, n2_idx);
                Edge e3(n2_idx, vidx);
                
                // Check if any edge was already used in a triangle
                // (simple check - in a more sophisticated version, we'd track edge->face mapping)
                // For now, we'll allow multiple triangles sharing edges
                
                // Ensure consistent winding order (counter-clockwise when viewed from outside)
                // Use triangle normal to determine winding
                float ax = n1.x - vertex.x;
                float ay = n1.y - vertex.y;
                float az = n1.z - vertex.z;
                
                float bx = n2.x - vertex.x;
                float by = n2.y - vertex.y;
                float bz = n2.z - vertex.z;
                
                float nx = ay * bz - az * by;
                float ny = az * bx - ax * bz;
                float nz = ax * by - ay * bx;
                
                // Check if normal points in reasonable direction (use vertex normal as reference)
                float dot = nx * vertex.nx + ny * vertex.ny + nz * vertex.nz;
                
                // Add triangle
                if (dot >= 0.0f) {
                    faces_.emplace_back(vidx, n1_idx, n2_idx);
                } else {
                    faces_.emplace_back(vidx, n2_idx, n1_idx);  // Reverse winding
                }
                
                faces_added++;
            }
        }
    }
    
    std::cerr << "[Model3D] Generated " << faces_added << " faces from " 
              << vertices_.size() << " vertices" << std::endl;
    
    return faces_added;
}

} // namespace forge_engine

