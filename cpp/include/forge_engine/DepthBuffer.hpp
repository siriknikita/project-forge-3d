#pragma once

#include "forge_engine/CircularBuffer.hpp"
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <mutex>

// Forward declaration to avoid circular dependency
namespace forge_engine {
    struct CameraPose;
}

// Include the full definition when needed (in .cpp file)

namespace forge_engine {

/**
 * Depth frame with associated gradient and pose information.
 */
    struct DepthFrame {
        cv::Mat depth_map;      // Z values
        cv::Mat gradient_map;   // ∇I values
        uint64_t timestamp;
        
        // Store pose data separately to avoid incomplete type issues
        // We'll store the transform matrix and other needed data
        cv::Mat pose_transform;  // 4x4 transformation matrix
        double pose_confidence;
        
        DepthFrame() : timestamp(0), pose_confidence(0.0) {}
        
        DepthFrame(const cv::Mat& depth, const cv::Mat& gradient, uint64_t ts, 
                   const cv::Mat& transform, double confidence)
            : depth_map(depth.clone())
            , gradient_map(gradient.clone())
            , timestamp(ts)
            , pose_transform(transform.clone())
            , pose_confidence(confidence)
        {}
    };

/**
 * Thread-safe depth buffer for storing and composing depth maps across frames.
 */
class DepthBuffer {
public:
    explicit DepthBuffer(size_t capacity = 10);
    ~DepthBuffer() = default;
    
    // Non-copyable
    DepthBuffer(const DepthBuffer&) = delete;
    DepthBuffer& operator=(const DepthBuffer&) = delete;
    
    /**
     * Append a depth frame to the buffer.
     * @param depth Depth map
     * @param gradient Gradient magnitude map
     * @param pose_transform 4x4 transformation matrix
     * @param timestamp Frame timestamp
     * @param confidence Pose confidence
     */
    void append(const cv::Mat& depth, const cv::Mat& gradient, 
                const cv::Mat& pose_transform, uint64_t timestamp, double confidence);
    
    /**
     * Compose depth maps using min-max composition with median filter.
     * @return Composed depth map
     */
    cv::Mat compose();
    
    /**
     * Get current size.
     */
    size_t size() const;
    
    /**
     * Check if buffer is empty.
     */
    bool empty() const;
    
    /**
     * Clear the buffer.
     */
    void clear();
    
    /**
     * Get all depth frames (for debugging).
     */
    std::vector<DepthFrame> getAll() const;

private:
    mutable std::mutex mutex_;
    CircularBuffer<DepthFrame> frames_;
    
    /**
     * Compose depth for a single pixel using median filter and min-max bounds.
     * @param depths Vector of depth values across frames
     * @param gradients Vector of gradient values across frames
     * @return Composed depth value
     */
    float composeDepthPixel(const std::vector<float>& depths, const std::vector<float>& gradients);
};

} // namespace forge_engine

