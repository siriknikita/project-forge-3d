#pragma once

#include "forge_engine/CircularBuffer.hpp"
#include "forge_engine/Model3D.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <mutex>
#include <dispatch/dispatch.h>

namespace forge_engine {

/**
 * Frame processing configuration.
 */
struct FrameConfig {
    uint32_t width;
    uint32_t height;
    uint32_t channels;  // Typically 3 (RGB) or 4 (RGBA)
};

/**
 * High-performance frame processor using Accelerate Framework and GCD.
 * Processes frames in parallel and builds 3D models progressively.
 */
class FrameProcessor {
public:
    explicit FrameProcessor(const FrameConfig& config);
    ~FrameProcessor();

    // Non-copyable
    FrameProcessor(const FrameProcessor&) = delete;
    FrameProcessor& operator=(const FrameProcessor&) = delete;

    /**
     * Process a frame (zero-copy from Python via pybind11).
     * This method is thread-safe and uses GCD for parallel processing.
     * 
     * @param frame_data Raw frame data (uint8_t array)
     * @param size Size of frame data in bytes
     */
    void processFrame(const uint8_t* frame_data, size_t size);

    /**
     * Get the 3D model (thread-safe).
     */
    std::shared_ptr<Model3D> getModel() const;

    /**
     * Get processing statistics.
     */
    struct Stats {
        uint64_t frames_processed;
        double avg_processing_time_ms;
    };

    Stats getStats() const;

    /**
     * Reset processor state.
     */
    void reset();

    /**
     * Placeholder for 3D reconstruction algorithm.
     * This should be replaced with actual reconstruction logic.
     * Made public for GCD dispatch callback.
     */
    void performReconstruction(const uint8_t* frame_data, size_t size);

private:
    FrameConfig config_;
    std::shared_ptr<Model3D> model_;
    CircularBuffer<std::vector<uint8_t>> frame_buffer_;
    
    // GCD queue for parallel processing
    dispatch_queue_t processing_queue_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    uint64_t frames_processed_;
    double total_processing_time_ms_;

    /**
     * Internal frame processing (runs on GCD queue).
     */
    void processFrameInternal(const uint8_t* frame_data, size_t size);
};

} // namespace forge_engine

