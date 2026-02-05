#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace forge_engine {

/**
 * 3D vertex with position, normal, and texture coordinates.
 */
struct Vertex {
    float x, y, z;           // Position
    float nx, ny, nz;        // Normal
    float u, v;              // Texture coordinates
    uint8_t r, g, b, a;      // Color (RGBA)

    Vertex() : x(0.0f), y(0.0f), z(0.0f), nx(0.0f), ny(0.0f), nz(0.0f), 
              u(0.0f), v(0.0f), r(255), g(255), b(255), a(255) {}

    Vertex(float x, float y, float z) 
        : x(x), y(y), z(z), nx(0.0f), ny(0.0f), nz(0.0f), 
          u(0.0f), v(0.0f), r(255), g(255), b(255), a(255) {}
};

/**
 * Face index (triangle).
 */
struct Face {
    uint32_t v0, v1, v2;

    Face(uint32_t v0, uint32_t v1, uint32_t v2)
        : v0(v0), v1(v1), v2(v2)
    {}
};

/**
 * High-quality 3D model builder with progressive refinement.
 * Thread-safe for concurrent access from multiple processing threads.
 */
class Model3D {
public:
    Model3D();
    ~Model3D();

    // Non-copyable
    Model3D(const Model3D&) = delete;
    Model3D& operator=(const Model3D&) = delete;

    /**
     * Add a vertex to the model.
     * Thread-safe.
     */
    uint32_t addVertex(const Vertex& vertex);

    /**
     * Add a face (triangle) to the model.
     * Thread-safe.
     */
    void addFace(uint32_t v0, uint32_t v1, uint32_t v2);

    /**
     * Update vertex position (for progressive refinement).
     * Thread-safe.
     */
    void updateVertexPosition(uint32_t index, float x, float y, float z);

    /**
     * Update vertex normal.
     * Thread-safe.
     */
    void updateVertexNormal(uint32_t index, float nx, float ny, float nz);

    /**
     * Get vertex count.
     */
    size_t getVertexCount() const;

    /**
     * Get face count.
     */
    size_t getFaceCount() const;

    /**
     * Clear the model.
     */
    void clear();

    /**
     * Generate mesh from point cloud using spatial neighbor analysis.
     * @param max_edge_length Maximum edge length for triangles (default: 0.1)
     * @param cell_size Spatial hash cell size (default: 0.01)
     * @return Number of faces generated
     */
    size_t generateMesh(float max_edge_length = 0.1f, float cell_size = 0.01f);

    /**
     * Export to PLY format (binary or ASCII).
     */
    bool exportPLY(const std::string& filename, bool binary = true) const;

    /**
     * Export to OBJ format.
     */
    bool exportOBJ(const std::string& filename) const;

    /**
     * Export to GLB (glTF Binary) format with UV mapping.
     * @param filename Output filename
     * @param texture_data Optional texture image data (RGBA)
     * @param texture_width Texture width
     * @param texture_height Texture height
     * @return True if successful
     */
    bool exportGLB(const std::string& filename, 
                   const uint8_t* texture_data = nullptr,
                   uint32_t texture_width = 0,
                   uint32_t texture_height = 0) const;

    /**
     * Get model statistics.
     */
    struct Statistics {
        size_t vertex_count;
        size_t face_count;
        float min_x, min_y, min_z;
        float max_x, max_y, max_z;
    };

    Statistics getStatistics() const;

private:
    mutable std::mutex mutex_;
    std::vector<Vertex> vertices_;
    std::vector<Face> faces_;
    
    void computeBoundingBox(float& min_x, float& min_y, float& min_z,
                           float& max_x, float& max_y, float& max_z) const;
    
    // Helper structures for mesh generation
    struct Edge {
        uint32_t v0, v1;
        Edge(uint32_t a, uint32_t b) : v0(std::min(a, b)), v1(std::max(a, b)) {}
        bool operator==(const Edge& other) const {
            return v0 == other.v0 && v1 == other.v1;
        }
    };
    
    struct EdgeHash {
        size_t operator()(const Edge& e) const {
            // Combine hashes with better distribution
            size_t h1 = std::hash<uint32_t>()(e.v0);
            size_t h2 = std::hash<uint32_t>()(e.v1);
            return h1 ^ (h2 << 1);
        }
    };
    
    float distanceSquared(const Vertex& v1, const Vertex& v2) const;
    float triangleArea(const Vertex& v0, const Vertex& v1, const Vertex& v2) const;
    bool isValidTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2, 
                         float max_edge_length, float min_area) const;
};

} // namespace forge_engine

