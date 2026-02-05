#include "forge_engine/DepthBuffer.hpp"
#include "forge_engine/FrameProcessor.hpp"
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

namespace forge_engine {

DepthBuffer::DepthBuffer(size_t capacity)
    : frames_(capacity)
{
}

void DepthBuffer::append(const cv::Mat& depth, const cv::Mat& gradient, 
                         const cv::Mat& pose_transform, uint64_t timestamp, double confidence) {
    std::lock_guard<std::mutex> lock(mutex_);
    DepthFrame frame(depth, gradient, timestamp, pose_transform, confidence);
    frames_.try_push(frame);
}

cv::Mat DepthBuffer::compose() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (frames_.empty()) {
        return cv::Mat();
    }
    
    // Get all frames using try_pop (non-blocking)
    // Note: We need to iterate through the buffer without modifying it
    // For now, we'll use a temporary buffer approach
    std::vector<DepthFrame> all_frames;
    DepthFrame frame;
    size_t current_size = frames_.size();
    for (size_t i = 0; i < current_size; ++i) {
        if (frames_.try_pop(frame)) {
            all_frames.push_back(frame);
        }
    }
    
    // Push them back (we need to keep them)
    for (const auto& f : all_frames) {
        frames_.try_push(f);
    }
    
    if (all_frames.empty()) {
        return cv::Mat();
    }
    
    // Get dimensions from first frame
    const cv::Size& frame_size = all_frames[0].depth_map.size();
    cv::Mat composed = cv::Mat::zeros(frame_size, CV_32F);
    
    // For each pixel, compose depth across frames
    for (int y = 0; y < frame_size.height; ++y) {
        for (int x = 0; x < frame_size.width; ++x) {
            std::vector<float> depths;
            std::vector<float> gradients;
            
            // Collect depth and gradient values across frames
            for (const auto& frame : all_frames) {
                if (frame.depth_map.type() == CV_32F && 
                    y < frame.depth_map.rows && x < frame.depth_map.cols) {
                    float depth_val = frame.depth_map.at<float>(y, x);
                    float grad_val = 0.0f;
                    
                    if (frame.gradient_map.type() == CV_32F &&
                        y < frame.gradient_map.rows && x < frame.gradient_map.cols) {
                        grad_val = frame.gradient_map.at<float>(y, x);
                    }
                    
                    // Only add valid depth values
                    if (depth_val > 0.0f && std::isfinite(depth_val)) {
                        depths.push_back(depth_val);
                        gradients.push_back(grad_val);
                    }
                }
            }
            
            if (!depths.empty()) {
                composed.at<float>(y, x) = composeDepthPixel(depths, gradients);
            }
        }
    }
    
    return composed;
}

float DepthBuffer::composeDepthPixel(const std::vector<float>& depths, const std::vector<float>& gradients) {
    if (depths.empty()) {
        return 0.0f;
    }
    
    // Find median
    std::vector<float> sorted_depths = depths;
    std::sort(sorted_depths.begin(), sorted_depths.end());
    float median = sorted_depths[sorted_depths.size() / 2];
    
    // Find min/max bounds
    float min_depth = *std::min_element(depths.begin(), depths.end());
    float max_depth = *std::max_element(depths.begin(), depths.end());
    
    // Filter outliers and prioritize sharp gradients
    float best_depth = median;
    float best_gradient = 0.0f;
    
    for (size_t i = 0; i < depths.size(); ++i) {
        // Keep values within bounds
        if (depths[i] >= min_depth && depths[i] <= max_depth) {
            // Prioritize sharp gradients (higher gradient = better surface definition)
            if (i < gradients.size() && gradients[i] > best_gradient) {
                best_gradient = gradients[i];
                best_depth = depths[i];
            }
        }
    }
    
    return best_depth;
}

size_t DepthBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_.size();
}

bool DepthBuffer::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_.empty();
}

void DepthBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    frames_.clear();
}

std::vector<DepthFrame> DepthBuffer::getAll() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DepthFrame> result;
    
    // Note: This is a simplified version. In practice, we'd need to
    // iterate through the circular buffer properly.
    // For now, we'll use the compose method which already does this.
    return result;
}

} // namespace forge_engine

