# Project Forge-3D: Hyperplane Composition & Color-Depth Algorithm

## Overview

This document describes the core algorithm used in Project Forge-3D for real-time 3D reconstruction from 2D video streams. The system uses feature detection (FAST/ORB) for camera pose estimation, color-depth heuristics for depth estimation, and min-max composition with median filtering to merge depth information across frames.

## Table of Contents

1. [Mathematical Foundations](#1-mathematical-foundations)
2. [Algorithm Pseudocode](#2-algorithm-pseudocode)
3. [Data Structures](#3-data-structures)
4. [Detailed Algorithm Steps](#4-detailed-algorithm-steps)
5. [Performance Characteristics](#5-performance-characteristics)
6. [Parameter Configuration](#6-parameter-configuration)
7. [Error Handling and Edge Cases](#7-error-handling-and-edge-cases)

## 1. Mathematical Foundations

### 1.1 Feature Detection and Camera Pose Problem

The system identifies stable anchor points (corners/edges) that remain consistent across frames as the camera moves. Let $P = \{k_1, k_2, \ldots, k_n\}$ be a set of keypoints detected in frame $t$.

**Keypoint Detection:**
- FAST (Features from Accelerated Segment Test): Detects corners by comparing pixel intensities in a circular pattern
- ORB (Oriented FAST and Rotated BRIEF): Adds rotation invariance and descriptor computation

**Camera Pose Estimation:**
Given keypoint correspondences between frames $t$ and $t+1$, we solve for camera rotation $\theta$ and translation $T$:

$$\begin{bmatrix} u' \\ v' \\ 1 \end{bmatrix} = K \begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

Where:
- $(u, v)$ and $(u', v')$ are corresponding keypoint coordinates in frames $t$ and $t+1$
- $K$ is the camera intrinsic matrix
- $R$ is the rotation matrix (derived from $\theta$)
- $T$ is the translation vector
- $(X, Y, Z)$ is the 3D point in world coordinates

**PnP Algorithm:**
We use Perspective-n-Point (PnP) algorithm (e.g., EPnP, P3P) to estimate pose from 2D-3D correspondences when 3D structure is known, or optical flow for frame-to-frame tracking.

### 1.2 Color-Depth Heuristic

The depth $Z$ at pixel $(x, y)$ is calculated as a function of luminance $L$ and image gradient $\nabla I$:

$$Z(x, y) = \alpha \cdot L(x, y) + \beta \cdot \nabla I(x, y)$$

Where:
- $L(x, y)$ is the luminance channel from LAB color space
- $\nabla I(x, y)$ is the image gradient magnitude: $\nabla I = \sqrt{(\frac{\partial I}{\partial x})^2 + (\frac{\partial I}{\partial y})^2}$
- $\alpha$ and $\beta$ are auto-tuned parameters based on frame statistics

**RGB to LAB Conversion:**
1. RGB → XYZ (using sRGB color space):
   - Linearize RGB values: $R' = f(R/255)$, $G' = f(G/255)$, $B' = f(B/255)$
   - Apply transformation matrix to get XYZ
2. XYZ → LAB:
   - Normalize by white point (D65): $X_n = X/X_w$, $Y_n = Y/Y_w$, $Z_n = Z/Z_w$
   - Apply LAB transformation: $L^* = 116 \cdot f(Y_n) - 16$, $a^* = 500 \cdot (f(X_n) - f(Y_n))$, $b^* = 200 \cdot (f(Y_n) - f(Z_n))$

**Gradient Calculation:**
Using Sobel or Scharr operators:
- $G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix} * I$
- $G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix} * I$
- $\nabla I = \sqrt{G_x^2 + G_y^2}$

**Auto-Tuning Parameters:**
Parameters $\alpha$ and $\beta$ are computed per frame to normalize depth range:

$$\alpha = \frac{Z_{max}}{\mu_L + 2\sigma_L}, \quad \beta = \frac{Z_{max}}{\mu_{\nabla I} + 2\sigma_{\nabla I}}$$

Where:
- $\mu_L, \sigma_L$ are mean and standard deviation of luminance
- $\mu_{\nabla I}, \sigma_{\nabla I}$ are mean and standard deviation of gradient magnitude
- $Z_{max}$ is the maximum depth value (e.g., 10.0 units)

### 1.3 Min-Max Composition Function

For a point in 3D space $(X, Y, Z)$, the final depth is computed from $N$ observed depth values across frames:

$$Z_{final}(x, y) = \text{composition}(Z_1, Z_2, \ldots, Z_N)$$

**Composition Algorithm:**
1. Collect depth values: $\{Z_1, Z_2, \ldots, Z_N\}$ for pixel $(x, y)$ across $N$ frames
2. Compute median: $Z_{median} = \text{median}(\{Z_i\})$
3. Compute bounds: $Z_{min} = \min(\{Z_i\})$, $Z_{max} = \max(\{Z_i\})$
4. Filter outliers: Keep only $Z_i$ where $Z_{min} \leq Z_i \leq Z_{max}$
5. Prioritize sharp gradients: Select $Z_i$ with maximum $\nabla I_i$ from filtered set

This approach:
- Rejects noise through median filtering
- Maintains plausible depth bounds
- Prioritizes surface definition from sharp shading gradients

### 1.4 Hyperplane Stitching

Multiple depth maps from different camera viewpoints are merged into a unified 3D model:

**3D Point Transformation:**
For each depth value $Z(x, y)$ in frame $t$ with camera pose $(\theta_t, T_t)$:

$$\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = K^{-1} \begin{bmatrix} u \cdot Z \\ v \cdot Z \\ Z \\ 1 \end{bmatrix}$$

Then transform to world coordinates:
$$\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix} = \begin{bmatrix} R_t & T_t \\ 0 & 1 \end{bmatrix}^{-1} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

**Point Cloud Merging:**
- Use spatial hashing to identify and merge duplicate points
- Apply distance threshold: points within $\epsilon$ distance are considered duplicates
- Average or weighted average of duplicate points based on confidence (gradient magnitude)

**Mesh Generation:**
- Option 1: Delaunay triangulation in 2D projection, then lift to 3D
- Option 2: Poisson surface reconstruction from point cloud
- Option 3: Marching cubes on voxel grid

## 2. Algorithm Pseudocode

```
ALGORITHM: HyperplaneComposition3DReconstruction
INPUT: Video stream frames {F₁, F₂, ..., Fₜ}
OUTPUT: 3D model M with vertices, faces, and UV coordinates

INITIALIZE:
    - DepthBuffer D with capacity N (e.g., N=10)
    - CameraPoseHistory P
    - Model3D M
    - FeatureDetector FD (FAST or ORB)
    - KeypointTracker KT

FOR each frame Fₜ:
    // Step 1: Feature Detection
    keypointsₜ = FD.detect(Fₜ)
    matches = KT.match(keypointsₜ₋₁, keypointsₜ)
    
    // Step 2: Camera Pose Estimation
    IF matches.size() >= 4:
        (θₜ, Tₜ) = estimatePose(matches, P)
        P.append(θₜ, Tₜ)
    ELSE:
        (θₜ, Tₜ) = P.last()  // Use previous pose
    
    // Step 3: Color-Depth Calculation
    LABₜ = convertRGBtoLAB(Fₜ)
    Lₜ = extractLuminance(LABₜ)
    ∇Iₜ = computeGradient(Fₜ)
    
    // Auto-tune parameters
    (α, β) = autoTuneDepthParams(Lₜ, ∇Iₜ)
    
    // Compute depth map
    FOR each pixel (x, y):
        Zₜ(x, y) = α · Lₜ(x, y) + β · ∇Iₜ(x, y)
    
    // Step 4: Composition
    D.append(Zₜ, ∇Iₜ)
    Z_composed = composeDepthMaps(D)  // Min-max with median
    
    // Step 5: Hyperplane Stitching
    pointCloudₜ = transformTo3D(Z_composed, θₜ, Tₜ)
    M.merge(pointCloudₜ)
    
    // Step 6: Mesh Update (periodic, not every frame)
    IF t mod 10 == 0:
        M.generateMesh()
        M.updateUVCoordinates(Fₜ)

RETURN M
```

## 3. Data Structures

### 3.1 DepthBuffer

```cpp
class DepthBuffer {
    struct DepthFrame {
        cv::Mat depth_map;      // Z values
        cv::Mat gradient_map;   // ∇I values
        CameraPose pose;         // θ, T
        uint64_t timestamp;
    };
    
    CircularBuffer<DepthFrame> frames;  // Last N frames
    size_t capacity;  // N (default: 10)
    
    // Methods
    void append(const cv::Mat& depth, const cv::Mat& gradient, const CameraPose& pose);
    cv::Mat compose();  // Returns composed depth map
};
```

### 3.2 CameraPose

```cpp
struct CameraPose {
    Eigen::Vector3f rotation;      // θ (axis-angle or Euler angles)
    Eigen::Vector3f translation;   // T
    Eigen::Matrix3f rotation_matrix;  // R (computed from θ)
    Eigen::Matrix4f transform;     // [R|T; 0|1]
    double confidence;              // Based on match quality
    uint64_t timestamp;
};
```

### 3.3 KeypointTracker

```cpp
class KeypointTracker {
    std::vector<cv::KeyPoint> previous_keypoints;
    cv::Mat previous_descriptors;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    // Methods
    std::vector<cv::DMatch> match(
        const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const cv::Mat& desc1,
        const cv::Mat& desc2
    );
};
```

## 4. Detailed Algorithm Steps

### Step 1: Feature Detection

```
detectAnchorPoints(frame):
    // Convert to grayscale if needed
    gray = convertToGrayscale(frame)
    
    // Detect keypoints
    IF using FAST:
        detector = cv::FastFeatureDetector::create(threshold=20)
    ELSE IF using ORB:
        detector = cv::ORB::create(nfeatures=1000)
    
    keypoints = detector->detect(gray)
    descriptors = detector->compute(gray, keypoints)
    
    RETURN (keypoints, descriptors)
```

### Step 2: Camera Pose Estimation

```
estimateCameraPose(matches, previous_pose):
    IF matches.size() < 4:
        RETURN previous_pose  // Insufficient matches
    
    // Extract 2D-2D correspondences
    points1 = extractPoints(matches, frame_t-1)
    points2 = extractPoints(matches, frame_t)
    
    // Estimate fundamental matrix
    F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC)
    
    // Recover pose (rotation and translation)
    E = K^T · F · K  // Essential matrix
    cv::recoverPose(E, points1, points2, K, R, T)
    
    // Convert to axis-angle representation
    theta = rotationMatrixToAxisAngle(R)
    
    RETURN CameraPose(theta, T, R)
```

### Step 3: Color-Depth Calculation

```
computeDepthMap(frame):
    // Convert RGB to LAB
    lab = rgbToLAB(frame)
    L = extractLuminance(lab)
    
    // Compute gradient
    grad_x = cv::Sobel(frame, CV_32F, 1, 0, ksize=3)
    grad_y = cv::Sobel(frame, CV_32F, 0, 1, ksize=3)
    grad_magnitude = sqrt(grad_x^2 + grad_y^2)
    
    // Auto-tune parameters
    mean_L = cv::mean(L)[0]
    std_L = cv::mean(cv::abs(L - mean_L))[0]
    mean_grad = cv::mean(grad_magnitude)[0]
    std_grad = cv::mean(cv::abs(grad_magnitude - mean_grad))[0]
    
    alpha = Z_MAX / (mean_L + 2 * std_L)
    beta = Z_MAX / (mean_grad + 2 * std_grad)
    
    // Compute depth
    depth = alpha * L + beta * grad_magnitude
    
    RETURN depth
```

### Step 4: Composition

```
composeDepthMaps(depth_buffer):
    // Get all depth maps
    depth_maps = depth_buffer.getAll()
    
    // Initialize composed depth map
    composed = zeros(depth_maps[0].size())
    
    FOR each pixel (x, y):
        // Collect depth values across frames
        depths = []
        gradients = []
        
        FOR each depth_frame in depth_maps:
            IF depth_frame.depth(x, y) is valid:
                depths.append(depth_frame.depth(x, y))
                gradients.append(depth_frame.gradient(x, y))
        
        IF depths.size() == 0:
            continue
        
        // Apply median filter
        sorted_depths = sort(depths)
        median = sorted_depths[depths.size() / 2]
        
        // Find bounds
        min_depth = min(depths)
        max_depth = max(depths)
        
        // Filter and prioritize sharp gradients
        best_depth = median
        best_gradient = 0
        
        FOR i in range(len(depths)):
            IF min_depth <= depths[i] <= max_depth:
                IF gradients[i] > best_gradient:
                    best_gradient = gradients[i]
                    best_depth = depths[i]
        
        composed(x, y) = best_depth
    
    RETURN composed
```

### Step 5: Hyperplane Stitching

```
stitchHyperplanes(composed_depth, pose, model):
    // Transform depth map to 3D points
    point_cloud = []
    
    FOR each pixel (x, y):
        Z = composed_depth(x, y)
        
        // Back-project to 3D
        X_cam = (x - cx) * Z / fx
        Y_cam = (y - cy) * Z / fy
        Z_cam = Z
        
        // Transform to world coordinates
        point_cam = [X_cam, Y_cam, Z_cam, 1]
        point_world = pose.transform^(-1) * point_cam
        
        point_cloud.append(point_world[0:3])
    
    // Merge with existing model
    model.mergePointCloud(point_cloud, epsilon=0.01)
    
    // Generate mesh (periodic)
    IF shouldGenerateMesh():
        model.generateMesh()
    
    RETURN model
```

## 5. Performance Characteristics

**Time Complexity:**
- Feature Detection: $O(W \cdot H)$ where $W \times H$ is frame resolution
- Pose Estimation: $O(M \log M)$ where $M$ is number of matches
- Depth Calculation: $O(W \cdot H)$
- Composition: $O(N \cdot W \cdot H)$ where $N$ is buffer size
- Stitching: $O(P \log P)$ where $P$ is number of points

**Space Complexity:**
- Depth Buffer: $O(N \cdot W \cdot H)$
- Point Cloud: $O(P)$ where $P$ grows with frames processed
- Model: $O(V + F)$ where $V$ is vertices, $F$ is faces

**Optimization Strategies:**
1. **Parallel Processing**: Use GCD to process frames concurrently
2. **Spatial Downsampling**: Process every $k$-th pixel for initial depth estimation
3. **Temporal Sampling**: Skip composition for some frames, update periodically
4. **Memory Management**: Limit depth buffer size, periodically flush old frames
5. **SIMD Operations**: Use Accelerate Framework for color conversion and gradient computation

## 6. Parameter Configuration

**Default Parameters:**
- Depth Buffer Size: $N = 10$ frames
- Feature Detector: ORB with 1000 features
- Depth Range: $Z_{max} = 10.0$ units
- Composition Window: Last 10 frames
- Spatial Hash Resolution: $\epsilon = 0.01$ units
- Mesh Generation Frequency: Every 10 frames
- Gradient Kernel: Scharr (3×3) or Sobel (3×3)

**Tunable Parameters:**
- `alpha_weight`: Weight for luminance component (auto-tuned)
- `beta_weight`: Weight for gradient component (auto-tuned)
- `depth_buffer_size`: Number of frames to maintain
- `feature_threshold`: Sensitivity of feature detection
- `match_ratio`: Ratio test for feature matching (e.g., 0.7)
- `outlier_threshold`: Threshold for RANSAC in pose estimation

## 7. Error Handling and Edge Cases

**Insufficient Features:**
- If < 4 keypoints detected, use previous pose
- If match ratio < 0.3, skip pose update

**Invalid Depth Values:**
- Filter out negative or zero depths
- Clamp depth to valid range [0, Z_max]

**Memory Constraints:**
- Limit point cloud size, use spatial hashing
- Periodically flush old depth frames
- Implement LRU cache for depth buffer

**Camera Motion:**
- Detect large motion (translation > threshold)
- Reset tracking if motion too large
- Use optical flow for smooth tracking

## Implementation Notes

The algorithm is implemented in C++ using:
- **OpenCV**: For feature detection, image processing, and camera calibration
- **Eigen3**: For matrix operations and transformations
- **Accelerate Framework**: For optimized color space conversions (macOS)
- **Grand Central Dispatch (GCD)**: For parallel frame processing

The main processing pipeline is in `FrameProcessor::performReconstruction()`, which orchestrates all the algorithm steps described above.

## References

- OpenCV Documentation: https://docs.opencv.org/
- Eigen Documentation: https://eigen.tuxfamily.org/
- glTF 2.0 Specification: https://www.khronos.org/gltf/
- Computer Vision: Algorithms and Applications (Szeliski, 2010)

