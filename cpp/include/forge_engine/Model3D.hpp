#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <mutex>

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
};

} // namespace forge_engine

