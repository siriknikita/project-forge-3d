#include "forge_engine/FrameProcessor.hpp"
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <iostream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

namespace forge_engine {

// Helper structure for async frame processing
struct FrameData {
    FrameProcessor* processor;
    std::vector<uint8_t> frame_data;
    
    FrameData(FrameProcessor* proc, const uint8_t* data, size_t size)
        : processor(proc), frame_data(data, data + size) {}
};

// Static function for GCD dispatch (C-compatible)
static void processFrameDispatch(void* context) {
    auto* frame_data = static_cast<FrameData*>(context);
    // Process frame directly (inline the processing logic)
    frame_data->processor->performReconstruction(frame_data->frame_data.data(), frame_data->frame_data.size());
    delete frame_data;
}

// C-compatible wrapper for GCD
extern "C" void processFrameDispatchC(void* context) {
    forge_engine::processFrameDispatch(context);
}

FrameProcessor::FrameProcessor(const FrameConfig& config)
    : config_(config)
    , model_(std::make_shared<Model3D>())
    , frame_buffer_(100)  // Buffer up to 100 frames
    , processing_queue_(dispatch_queue_create("com.forge_engine.processing", DISPATCH_QUEUE_CONCURRENT))
    , frames_processed_(0)
    , frames_rejected_(0)
    , total_processing_time_ms_(0.0)
    , pose_history_(30)  // Store last 30 poses
    , depth_buffer_(10)  // Store last 10 depth frames for composition
{
    // Initialize feature detector (ORB is more robust than FAST)
    feature_detector_ = cv::ORB::create(1000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    descriptor_extractor_ = feature_detector_;  // ORB includes descriptor extraction
    descriptor_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    
    // Initialize camera intrinsics
    initializeCameraIntrinsics();
    
    std::cerr << "[FrameProcessor] Initialized with config: "
              << config.width << "x" << config.height 
              << " (" << config.channels << " channels)" << std::endl;
}

FrameProcessor::~FrameProcessor() {
    if (processing_queue_) {
        dispatch_release(processing_queue_);
    }
}

void FrameProcessor::processFrame(const uint8_t* frame_data, size_t size) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Expected frame size
    size_t expected_size = config_.width * config_.height * config_.channels;
    if (size != expected_size) {
        // Frame size mismatch - skip or handle error
        std::lock_guard<std::mutex> lock(stats_mutex_);
        frames_rejected_++;
        
        // Log rejection (only log first few and then every 100th to avoid spam)
        if (frames_rejected_ <= 5 || frames_rejected_ % 100 == 0) {
            std::cerr << "[FrameProcessor] Frame rejected: size mismatch. "
                      << "Expected: " << expected_size << " bytes ("
                      << config_.width << "x" << config_.height 
                      << "x" << config_.channels << "), "
                      << "Received: " << size << " bytes. "
                      << "Total rejected: " << frames_rejected_ << std::endl;
        }
        return;
    }

    // Create frame data wrapper for async processing
    auto* frame_wrapper = new forge_engine::FrameData(this, frame_data, size);

    // Dispatch to GCD queue for parallel processing (using C-compatible function)
    dispatch_async_f(processing_queue_, frame_wrapper, processFrameDispatchC);

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    frames_processed_++;
    total_processing_time_ms_ += duration.count() / 1000.0;
}

void FrameProcessor::processFrameInternal(const uint8_t* frame_data, size_t size) {
    // Perform reconstruction
    performReconstruction(frame_data, size);
}

void FrameProcessor::performReconstruction(const uint8_t* frame_data, size_t size) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const uint32_t width = config_.width;
    const uint32_t height = config_.height;
    const uint32_t channels = config_.channels;
    
    // Convert frame data to OpenCV Mat
    cv::Mat frame(height, width, CV_8UC3);
    if (channels == 3) {
        memcpy(frame.data, frame_data, size);
    } else if (channels == 4) {
        // Convert RGBA to RGB
        cv::Mat rgba_frame(height, width, CV_8UC4, const_cast<uint8_t*>(frame_data));
        cv::cvtColor(rgba_frame, frame, cv::COLOR_RGBA2BGR);
    } else {
        std::cerr << "[FrameProcessor] Unsupported channel count: " << channels << std::endl;
        return;
    }
    
    // Step 1: Feature Detection
    auto [keypoints, descriptors] = detectAnchorPoints(frame);
    
    // Step 2: Camera Pose Estimation
    CameraPose pose;
    {
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        
        if (!previous_keypoints_.empty() && !descriptors.empty() && !previous_descriptors_.empty()) {
            // Match keypoints
            std::vector<cv::DMatch> matches;
            descriptor_matcher_->match(previous_descriptors_, descriptors, matches);
            
            // Filter matches using ratio test
            std::vector<cv::DMatch> good_matches;
            for (const auto& match : matches) {
                if (match.distance < 50.0) {  // Threshold for ORB
                    good_matches.push_back(match);
                }
            }
            
            if (good_matches.size() >= 4) {
                pose = estimateCameraPose(good_matches, previous_keypoints_, keypoints);
            } else {
                pose = current_pose_;  // Use previous pose
                pose.confidence = 0.0;
            }
        } else {
            // First frame - initialize pose
            pose = CameraPose();
            pose.confidence = 1.0;
            auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            pose.timestamp = static_cast<uint64_t>(now);
        }
        
        // Update previous frame data
        previous_keypoints_ = keypoints;
        previous_descriptors_ = descriptors;
    }
    
    // Step 3: Color-Depth Calculation
    cv::Mat depth_map = computeDepthMap(frame);
    cv::Mat gradient_map = computeImageGradient(frame);
    
    // Step 4: Add to depth buffer
    // Convert Eigen matrix to OpenCV Mat for storage
    cv::Mat pose_transform_cv;
    cv::eigen2cv(pose.transform, pose_transform_cv);
    depth_buffer_.append(depth_map, gradient_map, pose_transform_cv, pose.timestamp, pose.confidence);
    
    // Step 5: Compose depth maps (periodically, not every frame for performance)
    static uint64_t frame_count = 0;
    frame_count++;
    
    cv::Mat composed_depth;
    if (frame_count % 5 == 0 || depth_buffer_.size() >= 5) {
        // Compose every 5th frame or when buffer has enough frames
        composed_depth = composeDepthMaps();
    } else {
        // Use current depth map if composition not ready
        composed_depth = depth_map;
    }
    
    // Step 6: Hyperplane Stitching (periodically)
    if (frame_count % 10 == 0 || !composed_depth.empty()) {
        stitchHyperplanes(composed_depth, pose, frame);
    }
    
    // Log processing time and statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    static uint64_t frames_logged = 0;
    if (++frames_logged % 30 == 0) {
        std::cerr << "[FrameProcessor] Processed frame: "
                  << "keypoints=" << keypoints.size() 
                  << ", pose_confidence=" << pose.confidence
                  << ", vertices=" << model_->getVertexCount()
                  << ", processing_time=" << duration.count() << "ms" << std::endl;
    }
}

std::shared_ptr<Model3D> FrameProcessor::getModel() const {
    return model_;
}

FrameProcessor::Stats FrameProcessor::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    Stats stats;
    stats.frames_processed = frames_processed_;
    stats.frames_rejected = frames_rejected_;
    stats.avg_processing_time_ms = frames_processed_ > 0 
        ? total_processing_time_ms_ / frames_processed_ 
        : 0.0;
    return stats;
}

void FrameProcessor::reset() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    std::lock_guard<std::mutex> tracking_lock(tracking_mutex_);
    frames_processed_ = 0;
    frames_rejected_ = 0;
    total_processing_time_ms_ = 0.0;
    model_->clear();
    previous_keypoints_.clear();
    previous_descriptors_ = cv::Mat();
    pose_history_.clear();
    current_pose_ = CameraPose();
    std::cerr << "[FrameProcessor] Reset: cleared model and statistics" << std::endl;
}

std::pair<std::vector<cv::KeyPoint>, cv::Mat> FrameProcessor::detectAnchorPoints(const cv::Mat& frame) {
    cv::Mat gray;
    
    // Convert to grayscale if needed
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame.clone();
    }
    
    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    feature_detector_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    return std::make_pair(keypoints, descriptors);
}

CameraPose FrameProcessor::estimateCameraPose(
    const std::vector<cv::DMatch>& matches,
    const std::vector<cv::KeyPoint>& prev_keypoints,
    const std::vector<cv::KeyPoint>& curr_keypoints
) {
    CameraPose pose;
    
    if (matches.size() < 4) {
        // Insufficient matches, return previous pose
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        pose = current_pose_;
        pose.confidence = 0.0;
        return pose;
    }
    
    // Extract matched points
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(prev_keypoints[match.queryIdx].pt);
        points2.push_back(curr_keypoints[match.trainIdx].pt);
    }
    
    // Find fundamental matrix using RANSAC
    cv::Mat fundamental_matrix, mask;
    fundamental_matrix = cv::findFundamentalMat(
        points1, points2,
        cv::FM_RANSAC,
        3.0,  // ransacReprojThreshold
        0.99, // confidence
        mask
    );
    
    if (fundamental_matrix.empty()) {
        // Failed to find fundamental matrix
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        pose = current_pose_;
        pose.confidence = 0.0;
        return pose;
    }
    
    // Count inliers
    int inlier_count = cv::countNonZero(mask);
    double match_ratio = static_cast<double>(inlier_count) / matches.size();
    
    if (match_ratio < 0.3) {
        // Too few inliers, use previous pose
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        pose = current_pose_;
        pose.confidence = 0.0;
        return pose;
    }
    
    // Compute essential matrix from fundamental matrix
    cv::Mat essential_matrix = camera_matrix_.t() * fundamental_matrix * camera_matrix_;
    
    // Recover pose from essential matrix
    cv::Mat R, t;
    cv::recoverPose(essential_matrix, points1, points2, camera_matrix_, R, t, mask);
    
    // Convert OpenCV matrices to Eigen
    Eigen::Matrix3f R_eigen;
    Eigen::Vector3f t_eigen;
    cv::cv2eigen(R, R_eigen);
    cv::cv2eigen(t, t_eigen);
    
    // Convert rotation matrix to axis-angle representation
    Eigen::AngleAxisf aa(R_eigen);
    pose.rotation = aa.axis() * aa.angle();
    pose.translation = t_eigen;
    pose.rotation_matrix = R_eigen;
    pose.updateTransform();
    pose.confidence = match_ratio;
    
    // Update timestamp
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    pose.timestamp = static_cast<uint64_t>(now);
    
    // Update current pose
    {
        std::lock_guard<std::mutex> lock(tracking_mutex_);
        current_pose_ = pose;
        pose_history_.try_push(pose);
    }
    
    return pose;
}

void FrameProcessor::initializeCameraIntrinsics() {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    
    if (calibration_.isValid()) {
        // Use calibration parameters
        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
            static_cast<double>(calibration_.fx), 0, static_cast<double>(calibration_.cx),
            0, static_cast<double>(calibration_.fy), static_cast<double>(calibration_.cy),
            0, 0, 1
        );
        
        // Use provided distortion coefficients or default
        if (!calibration_.distortion_coeffs.empty()) {
            distortion_coeffs_ = calibration_.distortion_coeffs.clone();
        } else {
            distortion_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
        }
    } else {
        // Default camera intrinsics (should be calibrated for real use)
        // Assuming a typical phone camera with 1080p resolution
        float fx = static_cast<float>(config_.width) * 1.2f;  // Focal length in pixels
        float fy = static_cast<float>(config_.height) * 1.2f;
        float cx = static_cast<float>(config_.width) / 2.0f;   // Principal point
        float cy = static_cast<float>(config_.height) / 2.0f;
        
        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1
        );
        
        // Default distortion coefficients (assuming minimal distortion)
        distortion_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
    }
}

void FrameProcessor::setCalibration(const CameraCalibration& calib) {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    calibration_ = calib;
    
    // Reinitialize camera intrinsics with new calibration
    initializeCameraIntrinsics();
}

CameraCalibration FrameProcessor::getCalibration() const {
    std::lock_guard<std::mutex> lock(calibration_mutex_);
    return calibration_;
}

void FrameProcessor::generateModelMesh(float max_edge_length, float cell_size) {
    if (model_) {
        model_->generateMesh(max_edge_length, cell_size);
    }
}

cv::Mat FrameProcessor::convertRGBtoLAB(const cv::Mat& rgb_frame) {
    cv::Mat lab_frame;
    
    // OpenCV uses BGR by default, convert to LAB
    if (rgb_frame.channels() == 3) {
        cv::cvtColor(rgb_frame, lab_frame, cv::COLOR_BGR2Lab);
    } else if (rgb_frame.channels() == 4) {
        cv::cvtColor(rgb_frame, lab_frame, cv::COLOR_BGRA2BGR);
        cv::cvtColor(lab_frame, lab_frame, cv::COLOR_BGR2Lab);
    } else {
        // Already grayscale or single channel, convert to LAB
        cv::cvtColor(rgb_frame, lab_frame, cv::COLOR_GRAY2BGR);
        cv::cvtColor(lab_frame, lab_frame, cv::COLOR_BGR2Lab);
    }
    
    return lab_frame;
}

cv::Mat FrameProcessor::extractLuminance(const cv::Mat& lab_frame) {
    std::vector<cv::Mat> lab_channels;
    cv::split(lab_frame, lab_channels);
    
    // L channel is the first channel (index 0) in LAB color space
    return lab_channels[0];
}

cv::Mat FrameProcessor::computeImageGradient(const cv::Mat& frame) {
    cv::Mat gray;
    
    // Convert to grayscale if needed
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame.clone();
    }
    
    // Convert to float for gradient computation
    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F, 1.0 / 255.0);
    
    // Compute gradients using Scharr operator (more accurate than Sobel)
    cv::Mat grad_x, grad_y;
    cv::Scharr(gray_float, grad_x, CV_32F, 1, 0, 1.0/32.0);
    cv::Scharr(gray_float, grad_y, CV_32F, 0, 1, 1.0/32.0);
    
    // Compute gradient magnitude
    cv::Mat grad_magnitude;
    cv::magnitude(grad_x, grad_y, grad_magnitude);
    
    return grad_magnitude;
}

std::pair<float, float> FrameProcessor::autoTuneDepthParams(const cv::Mat& L, const cv::Mat& gradI) {
    // Compute statistics
    cv::Scalar mean_L_scalar, stddev_L_scalar;
    cv::meanStdDev(L, mean_L_scalar, stddev_L_scalar);
    double mean_L = mean_L_scalar[0];
    double std_L = stddev_L_scalar[0];
    
    cv::Scalar mean_grad_scalar, stddev_grad_scalar;
    cv::meanStdDev(gradI, mean_grad_scalar, stddev_grad_scalar);
    double mean_grad = mean_grad_scalar[0];
    double std_grad = stddev_grad_scalar[0];
    
    // Normalize to depth range [0, 10.0]
    const float Z_MAX = 10.0f;
    
    // Avoid division by zero
    float alpha = 0.0f;
    if (mean_L + 2.0 * std_L > 1e-6) {
        alpha = Z_MAX / static_cast<float>(mean_L + 2.0 * std_L);
    }
    
    float beta = 0.0f;
    if (mean_grad + 2.0 * std_grad > 1e-6) {
        beta = Z_MAX / static_cast<float>(mean_grad + 2.0 * std_grad);
    }
    
    return std::make_pair(alpha, beta);
}

cv::Mat FrameProcessor::computeDepthMap(const cv::Mat& frame) {
    // Convert RGB to LAB
    cv::Mat lab_frame = convertRGBtoLAB(frame);
    
    // Extract luminance
    cv::Mat L = extractLuminance(lab_frame);
    L.convertTo(L, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]
    
    // Compute gradient
    cv::Mat gradI = computeImageGradient(frame);
    
    // Auto-tune parameters
    auto [alpha, beta] = autoTuneDepthParams(L, gradI);
    
    // Compute depth: Z = α·L + β·∇I
    cv::Mat depth_map;
    cv::Mat alpha_L, beta_gradI;
    L.convertTo(alpha_L, CV_32F);
    gradI.convertTo(beta_gradI, CV_32F);
    
    alpha_L *= alpha;
    beta_gradI *= beta;
    
    depth_map = alpha_L + beta_gradI;
    
    // Clamp depth to valid range [0, Z_MAX]
    const float Z_MAX = 10.0f;
    cv::threshold(depth_map, depth_map, Z_MAX, Z_MAX, cv::THRESH_TRUNC);
    cv::threshold(depth_map, depth_map, 0.0f, 0.0f, cv::THRESH_TOZERO);
    
    return depth_map;
}

cv::Mat FrameProcessor::composeDepthMaps() {
    return depth_buffer_.compose();
}

void FrameProcessor::stitchHyperplanes(const cv::Mat& depth_map, const CameraPose& pose, const cv::Mat& frame) {
    if (depth_map.empty() || depth_map.type() != CV_32F) {
        return;
    }
    
    // Get camera intrinsics
    double fx = camera_matrix_.at<double>(0, 0);
    double fy = camera_matrix_.at<double>(1, 1);
    double cx = camera_matrix_.at<double>(0, 2);
    double cy = camera_matrix_.at<double>(1, 2);
    
    // Get scale factor from calibration
    float scale_factor = 1.0f;
    {
        std::lock_guard<std::mutex> lock(calibration_mutex_);
        if (calibration_.isValid()) {
            scale_factor = calibration_.scale_factor;
        }
    }
    
    // Get inverse transform (world to camera)
    Eigen::Matrix4f transform_inv = pose.transform.inverse();
    
    // Sample rate to reduce point cloud density (for performance)
    const int sample_rate = 2;  // Sample every 2nd pixel
    
    // Spatial hashing parameters for duplicate detection
    const float epsilon = 0.01f;  // Distance threshold for duplicates
    const float cell_size_hash = epsilon * 2.0f;
    
    // Transform depth map to 3D points
    for (int y = 0; y < depth_map.rows; y += sample_rate) {
        for (int x = 0; x < depth_map.cols; x += sample_rate) {
            float Z = depth_map.at<float>(y, x);
            
            // Skip invalid depths
            if (Z <= 0.0f || !std::isfinite(Z)) {
                continue;
            }
            
            // Apply scale factor to depth
            float Z_scaled = Z * scale_factor;
            
            // Back-project to 3D camera coordinates
            float X_cam = static_cast<float>((x - cx) * Z_scaled / fx);
            float Y_cam = static_cast<float>((y - cy) * Z_scaled / fy);
            float Z_cam = Z_scaled;
            
            // Transform to world coordinates
            Eigen::Vector4f point_cam(X_cam, Y_cam, Z_cam, 1.0f);
            Eigen::Vector4f point_world = transform_inv * point_cam;
            
            // Get color from frame for UV mapping
            uint8_t r = 255, g = 255, b = 255;
            if (!frame.empty() && y < frame.rows && x < frame.cols) {
                if (frame.channels() >= 3) {
                    cv::Vec3b color = frame.at<cv::Vec3b>(y, x);
                    b = color[0];
                    g = color[1];
                    r = color[2];
                }
            }
            
            // Compute normal (simplified - use gradient of depth)
            float nx = 0.0f, ny = 0.0f, nz = 1.0f;
            
            // Simple normal estimation from depth gradient
            if (x > 0 && x < depth_map.cols - 1 && y > 0 && y < depth_map.rows - 1) {
                float dz_dx = depth_map.at<float>(y, x + 1) - depth_map.at<float>(y, x - 1);
                float dz_dy = depth_map.at<float>(y + 1, x) - depth_map.at<float>(y - 1, x);
                
                // Normal in camera space
                Eigen::Vector3f normal_cam(-dz_dx, -dz_dy, 2.0f * cell_size_hash);
                normal_cam.normalize();
                
                // Transform normal to world space
                Eigen::Matrix3f R_inv = transform_inv.block<3, 3>(0, 0);
                Eigen::Vector3f normal_world = R_inv * normal_cam;
                normal_world.normalize();
                
                nx = normal_world.x();
                ny = normal_world.y();
                nz = normal_world.z();
            }
            
            // Create vertex
            Vertex vertex;
            vertex.x = point_world.x();
            vertex.y = point_world.y();
            vertex.z = point_world.z();
            vertex.nx = nx;
            vertex.ny = ny;
            vertex.nz = nz;
            
            // UV coordinates from pixel position
            vertex.u = static_cast<float>(x) / static_cast<float>(depth_map.cols);
            vertex.v = static_cast<float>(y) / static_cast<float>(depth_map.rows);
            
            vertex.r = r;
            vertex.g = g;
            vertex.b = b;
            vertex.a = 255;
            
            // Add vertex to model
            model_->addVertex(vertex);
        }
    }
    
    // Note: Mesh generation is done separately via generateModelMesh()
    // to avoid performance impact during frame processing
}

} // namespace forge_engine

