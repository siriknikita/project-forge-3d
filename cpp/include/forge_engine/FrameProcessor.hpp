#pragma once

#include "forge_engine/CircularBuffer.hpp"
#include "forge_engine/Model3D.hpp"
#include "forge_engine/DepthBuffer.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <mutex>
#include <vector>
#include <dispatch/dispatch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

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
 * Camera pose with rotation and translation.
 */
struct CameraPose {
    Eigen::Vector3f rotation;           // θ (axis-angle representation)
    Eigen::Vector3f translation;       // T
    Eigen::Matrix3f rotation_matrix;  // R (computed from rotation)
    Eigen::Matrix4f transform;         // [R|T; 0|1]
    double confidence;                  // Based on match quality (0.0 to 1.0)
    uint64_t timestamp;
    
    CameraPose() : rotation(Eigen::Vector3f::Zero()), 
                   translation(Eigen::Vector3f::Zero()),
                   rotation_matrix(Eigen::Matrix3f::Identity()),
                   transform(Eigen::Matrix4f::Identity()),
                   confidence(0.0), timestamp(0) {}
    
    void updateTransform() {
        // Convert axis-angle to rotation matrix
        float angle = rotation.norm();
        if (angle > 1e-6f) {
            Eigen::Vector3f axis = rotation / angle;
            Eigen::AngleAxisf aa(angle, axis);
            rotation_matrix = aa.toRotationMatrix();
        } else {
            rotation_matrix = Eigen::Matrix3f::Identity();
        }
        
        // Build transform matrix
        transform.setIdentity();
        transform.block<3, 3>(0, 0) = rotation_matrix;
        transform.block<3, 1>(0, 3) = translation;
    }
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
        uint64_t frames_rejected;
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

    /**
     * Detect anchor points (keypoints) in the frame.
     * @param frame OpenCV Mat containing the frame
     * @return Vector of keypoints and their descriptors
     */
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> detectAnchorPoints(const cv::Mat& frame);

    /**
     * Estimate camera pose from keypoint matches.
     * @param matches Keypoint matches between previous and current frame
     * @param prev_keypoints Previous frame keypoints
     * @param curr_keypoints Current frame keypoints
     * @return Camera pose (rotation and translation)
     */
    CameraPose estimateCameraPose(
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& prev_keypoints,
        const std::vector<cv::KeyPoint>& curr_keypoints
    );

    /**
     * Convert RGB image to LAB color space.
     * @param rgb_frame Input RGB frame
     * @return LAB color space image
     */
    cv::Mat convertRGBtoLAB(const cv::Mat& rgb_frame);

    /**
     * Extract luminance channel from LAB image.
     * @param lab_frame LAB color space image
     * @return Luminance channel (L)
     */
    cv::Mat extractLuminance(const cv::Mat& lab_frame);

    /**
     * Compute image gradient magnitude.
     * @param frame Input frame (grayscale or RGB)
     * @return Gradient magnitude map
     */
    cv::Mat computeImageGradient(const cv::Mat& frame);

    /**
     * Auto-tune depth parameters based on frame statistics.
     * @param L Luminance channel
     * @param gradI Gradient magnitude
     * @return Pair of (alpha, beta) parameters
     */
    std::pair<float, float> autoTuneDepthParams(const cv::Mat& L, const cv::Mat& gradI);

    /**
     * Compute depth map using color-depth heuristic.
     * @param frame Input RGB frame
     * @return Depth map (Z values)
     */
    cv::Mat computeDepthMap(const cv::Mat& frame);

    /**
     * Compose depth maps from buffer using min-max composition with median filter.
     * @return Composed depth map
     */
    cv::Mat composeDepthMaps();

    /**
     * Transform depth map to 3D point cloud and stitch with existing model.
     * @param depth_map Composed depth map
     * @param pose Camera pose
     * @param frame Original frame for UV mapping
     */
    void stitchHyperplanes(const cv::Mat& depth_map, const CameraPose& pose, const cv::Mat& frame);

private:
    FrameConfig config_;
    std::shared_ptr<Model3D> model_;
    CircularBuffer<std::vector<uint8_t>> frame_buffer_;
    
    // GCD queue for parallel processing
    dispatch_queue_t processing_queue_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    uint64_t frames_processed_;
    uint64_t frames_rejected_;
    double total_processing_time_ms_;
    
    // Feature detection and tracking
    cv::Ptr<cv::FeatureDetector> feature_detector_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
    
    // Previous frame data for tracking
    mutable std::mutex tracking_mutex_;
    std::vector<cv::KeyPoint> previous_keypoints_;
    cv::Mat previous_descriptors_;
    
    // Camera pose history
    CircularBuffer<CameraPose> pose_history_;
    CameraPose current_pose_;
    
    // Depth buffer for composition
    DepthBuffer depth_buffer_;
    
    // Camera intrinsics (default values, should be calibrated)
    cv::Mat camera_matrix_;
    cv::Mat distortion_coeffs_;

    /**
     * Internal frame processing (runs on GCD queue).
     */
    void processFrameInternal(const uint8_t* frame_data, size_t size);
    
    /**
     * Initialize camera intrinsics with default values.
     */
    void initializeCameraIntrinsics();
};

} // namespace forge_engine

