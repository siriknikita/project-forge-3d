# Project Forge-3D: Hyperplane Composition & Color-Depth Algorithm

## Overview

This document describes the core algorithm used in Project Forge-3D for real-time 3D reconstruction from 2D video streams. The system uses feature detection (FAST/ORB) for camera pose estimation, color-depth heuristics for depth estimation, min-max composition with median filtering to merge depth information across frames, spatial hashing for mesh generation from point clouds, and camera calibration for accurate depth-to-world coordinate conversion.

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

**Depth-to-World Coordinate Conversion:**
When camera calibration is provided, depth values are scaled using the calibration scale factor:

$$Z_{world} = Z_{depth} \cdot s$$

Where $s$ is the `scale_factor` from camera calibration (e.g., meters per depth unit).

The back-projection to 3D camera coordinates uses calibrated intrinsics:

$$X_{cam} = \frac{(x - c_x) \cdot Z_{world}}{f_x}, \quad Y_{cam} = \frac{(y - c_y) \cdot Z_{world}}{f_y}, \quad Z_{cam} = Z_{world}$$

Where $(f_x, f_y)$ are focal lengths and $(c_x, c_y)$ is the principal point from calibration.

### 1.5 Mesh Generation from Point Cloud

The system generates triangular meshes from point clouds using spatial neighbor analysis. This converts unconnected vertices into a proper 3D mesh with face connectivity.

**Spatial Hashing:**
The 3D space is partitioned into a uniform grid with cell size $c$ (default: 0.01 units). Each vertex is hashed to its containing cell:

$$\text{cell}_x = \lfloor \frac{x - x_{min}}{c} \rfloor, \quad \text{cell}_y = \lfloor \frac{y - y_{min}}{c} \rfloor, \quad \text{cell}_z = \lfloor \frac{z - z_{min}}{c} \rfloor$$

**Neighbor Finding:**
For each vertex $v_i$, the algorithm:
1. Identifies the cell containing $v_i$ and all 26 adjacent cells (3×3×3 neighborhood)
2. Collects candidate neighbors from these cells
3. Filters candidates by maximum edge length: $d(v_i, v_j) \leq L_{max}$
4. Selects the $k$ nearest neighbors (default: $k = 8$)

**Triangle Formation:**
For each vertex $v_i$ with neighbors $\{n_1, n_2, \ldots, n_k\}$, triangles are formed by pairing neighbors:

$$\text{Triangle}(v_i, n_j, n_k) \text{ for } j < k$$

**Quality Filtering:**
Each candidate triangle is validated:

1. **Edge Length Check:** All edges must satisfy $d \leq L_{max}$
2. **Area Check:** Triangle area must exceed minimum threshold: $\text{Area} \geq A_{min}$ (default: $10^{-6}$)
3. **Normal Consistency:** Triangle normal $\vec{n}_{tri}$ should be consistent with vertex normals:
   $$\vec{n}_{tri} \cdot \vec{n}_v \geq \theta_{normal}$$ (default: $\theta_{normal} = 0.3$, ~73°)

**Triangle Normal Computation:**
$$\vec{n}_{tri} = \frac{(\vec{v}_1 - \vec{v}_0) \times (\vec{v}_2 - \vec{v}_0)}{|(\vec{v}_1 - \vec{v}_0) \times (\vec{v}_2 - \vec{v}_0)|}$$

**Winding Order:**
Triangles are oriented to ensure consistent winding (counter-clockwise when viewed from outside):
- If $\vec{n}_{tri} \cdot \vec{n}_v \geq 0$: Use order $(v_i, n_j, n_k)$
- Otherwise: Reverse to $(v_i, n_k, n_j)$

### 1.6 Camera Calibration

The system supports camera calibration for accurate depth-to-world coordinate conversion.

**Calibration Parameters:**
- $f_x, f_y$: Focal lengths in pixels
- $c_x, c_y$: Principal point (optical center) in pixels
- $s$: Scale factor for depth-to-world conversion (e.g., meters per depth unit)
- $\mathbf{d}$: Distortion coefficients $(k_1, k_2, p_1, p_2, k_3)$

**Camera Intrinsic Matrix:**
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

**Default Calibration:**
If calibration is not provided, the system uses estimated intrinsics:
- $f_x = 1.2 \cdot W$, $f_y = 1.2 \cdot H$ (where $W \times H$ is frame resolution)
- $c_x = W/2$, $c_y = H/2$
- $s = 1.0$ (assumes depth units are already in world units)
- $\mathbf{d} = \mathbf{0}$ (no distortion)

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
    pointCloudₜ = transformTo3D(Z_composed, θₜ, Tₜ, calibration)
    M.merge(pointCloudₜ)
    
    // Note: Mesh generation is done separately via API call
    // M.generateMesh(max_edge_length, cell_size) when requested

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

### 3.4 CameraCalibration

```cpp
struct CameraCalibration {
    float fx, fy;              // Focal lengths in pixels
    float cx, cy;              // Principal point
    float scale_factor;         // Depth-to-world scale
    cv::Mat distortion_coeffs;  // Lens distortion (k1, k2, p1, p2, k3)
    
    bool isValid() const;       // Check if calibration is valid
};
```

### 3.5 Model3D

```cpp
class Model3D {
    // Vertices and faces
    std::vector<Vertex> vertices_;
    std::vector<Face> faces_;
    
    // Mesh generation
    size_t generateMesh(
        float max_edge_length = 0.1f,
        float cell_size = 0.01f
    );
    
    // Helper methods for mesh generation
    float distanceSquared(const Vertex& v1, const Vertex& v2) const;
    float triangleArea(const Vertex& v0, const Vertex& v1, const Vertex& v2) const;
    bool isValidTriangle(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                         float max_edge_length, float min_area) const;
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
stitchHyperplanes(composed_depth, pose, model, calibration):
    // Get scale factor from calibration
    scale_factor = calibration.isValid() ? calibration.scale_factor : 1.0
    
    // Transform depth map to 3D points
    point_cloud = []
    
    FOR each pixel (x, y):
        Z_depth = composed_depth(x, y)
        
        // Apply scale factor
        Z_world = Z_depth * scale_factor
        
        // Back-project to 3D using calibrated intrinsics
        X_cam = (x - cx) * Z_world / fx
        Y_cam = (y - cy) * Z_world / fy
        Z_cam = Z_world
        
        // Transform to world coordinates
        point_cam = [X_cam, Y_cam, Z_cam, 1]
        point_world = pose.transform^(-1) * point_cam
        
        point_cloud.append(point_world[0:3])
    
    // Merge with existing model
    model.mergePointCloud(point_cloud, epsilon=0.01)
    
    // Note: Mesh generation is done separately via API
    // model.generateMesh(max_edge_length, cell_size)
    
    RETURN model
```

### Step 6: Mesh Generation

```
generateMesh(point_cloud, max_edge_length, cell_size):
    // Clear existing faces
    faces.clear()
    
    IF point_cloud.size() < 3:
        RETURN 0  // Need at least 3 vertices
    
    // Compute bounding box
    (min_x, min_y, min_z, max_x, max_y, max_z) = computeBoundingBox(point_cloud)
    
    // Build spatial hash grid
    grid_x = ceil((max_x - min_x) / cell_size) + 1
    grid_y = ceil((max_y - min_y) / cell_size) + 1
    grid_z = ceil((max_z - min_z) / cell_size) + 1
    
    spatial_hash = {}
    FOR each vertex v_i in point_cloud:
        cell_x = floor((v_i.x - min_x) / cell_size)
        cell_y = floor((v_i.y - min_y) / cell_size)
        cell_z = floor((v_i.z - min_z) / cell_size)
        spatial_hash[cell_x, cell_y, cell_z].append(i)
    
    // Generate triangles
    FOR each vertex v_i:
        // Find neighbors in 3×3×3 cell neighborhood
        candidates = []
        FOR dx, dy, dz in [-1, 0, 1]:
            cell = spatial_hash[cell_x + dx, cell_y + dy, cell_z + dz]
            FOR each vertex v_j in cell:
                IF v_j != v_i AND distance(v_i, v_j) <= max_edge_length:
                    candidates.append((j, distance²(v_i, v_j)))
        
        // Sort by distance and take k nearest
        sort(candidates by distance)
        neighbors = candidates[0:min(k, len(candidates))]
        
        // Form triangles with pairs of neighbors
        FOR j in range(len(neighbors)):
            FOR k in range(j+1, len(neighbors)):
                n1 = neighbors[j]
                n2 = neighbors[k]
                
                IF isValidTriangle(v_i, n1, n2, max_edge_length, min_area):
                    // Check normal consistency
                    n_tri = computeTriangleNormal(v_i, n1, n2)
                    dot = n_tri · v_i.normal
                    
                    IF dot >= 0:
                        faces.append((i, n1, n2))
                    ELSE:
                        faces.append((i, n2, n1))  // Reverse winding
    
    RETURN len(faces)
```

## 5. Performance Characteristics

**Time Complexity:**
- Feature Detection: $O(W \cdot H)$ where $W \times H$ is frame resolution
- Pose Estimation: $O(M \log M)$ where $M$ is number of matches
- Depth Calculation: $O(W \cdot H)$
- Composition: $O(N \cdot W \cdot H)$ where $N$ is buffer size
- Stitching: $O(P)$ where $P$ is number of points (with spatial sampling)
- Mesh Generation: $O(P \cdot k^2)$ where $P$ is vertices and $k$ is neighbors per vertex (typically $k=8$)

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
- Mesh Generation: On-demand via API (not automatic)
- Mesh Parameters:
  - Maximum edge length: $L_{max} = 0.1$ units
  - Spatial hash cell size: $c = 0.01$ units
  - Number of neighbors: $k = 8$
  - Minimum triangle area: $A_{min} = 10^{-6}$
- Gradient Kernel: Scharr (3×3) or Sobel (3×3)
- Camera Calibration: Default estimated values (can be set via API)

**Tunable Parameters:**
- `alpha_weight`: Weight for luminance component (auto-tuned)
- `beta_weight`: Weight for gradient component (auto-tuned)
- `depth_buffer_size`: Number of frames to maintain
- `feature_threshold`: Sensitivity of feature detection
- `match_ratio`: Ratio test for feature matching (e.g., 0.7)
- `outlier_threshold`: Threshold for RANSAC in pose estimation
- `max_edge_length`: Maximum triangle edge length for mesh generation (default: 0.1)
- `cell_size`: Spatial hash cell size for mesh generation (default: 0.01)
- `fx, fy`: Camera focal lengths in pixels (calibration)
- `cx, cy`: Camera principal point in pixels (calibration)
- `scale_factor`: Depth-to-world scale factor (calibration, default: 1.0)

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

### API Endpoints

The system provides REST API endpoints for mesh generation and camera calibration:

**Mesh Generation:**
- `POST /model/generate-mesh?token=<token>&max_edge_length=<float>&cell_size=<float>`
  - Generates mesh from current point cloud
  - Returns number of faces generated

**Camera Calibration:**
- `GET /camera/calibration?token=<token>`
  - Returns current calibration parameters
- `POST /camera/calibration?token=<token>&fx=<float>&fy=<float>&cx=<float>&cy=<float>&scale_factor=<float>`
  - Sets camera calibration parameters
  - Updates camera intrinsics for subsequent frame processing

**Model Status:**
- `GET /model/status?token=<token>`
  - Returns model statistics including vertex count and face count

## References

- OpenCV Documentation: https://docs.opencv.org/
- Eigen Documentation: https://eigen.tuxfamily.org/
- glTF 2.0 Specification: https://www.khronos.org/gltf/
- Computer Vision: Algorithms and Applications (Szeliski, 2010)
- Spatial Hashing for Real-Time Collision Detection (Teschner et al., 2003)
- Surface Reconstruction from Point Clouds (Berger et al., 2017)

