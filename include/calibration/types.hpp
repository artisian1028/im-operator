#ifndef CALIBRATION_TYPES_HPP
#define CALIBRATION_TYPES_HPP

#include <cstdint>

namespace calibration {

enum class CalibrationAlgorithm {
    THREE_POINT,             // Three-point rod calibration (intrinsics + extrinsics)
    RIGHT_TRIANGLE,          // Right-triangle world coordinate registration
    CHECKERBOARD_DETECT,     // Detect chessboard corners at sub-pixel accuracy
    CHECKERBOARD_CALIBRATE,  // Zhang's camera calibration from multiple chessboard views
    DLT,                     // Direct Linear Transform: N 3D→2D point correspondences
    STEREO_CALIBRATE,        // Dual-camera stereo extrinsic calibration
    BUNDLE_ADJUST            // Multi-camera joint refinement via sparse LM
};

enum class CalibrationError {
    Ok = 0,
    NullInput,
    InvalidCameraCount,
    InvalidFrameCount,
    InvalidConfiguration,
    InsufficientObservations,
    SingularMatrix,
    OptimizationFailed,
    InternalError
};

inline const char* calibration_error_message(CalibrationError err) {
    switch (err) {
        case CalibrationError::Ok:
            return "Success";
        case CalibrationError::NullInput:
            return "Null input pointer";
        case CalibrationError::InvalidCameraCount:
            return "Invalid camera count (must be >= 1)";
        case CalibrationError::InvalidFrameCount:
            return "Invalid frame count (must be >= 3)";
        case CalibrationError::InvalidConfiguration:
            return "Invalid calibration configuration parameter";
        case CalibrationError::InsufficientObservations:
            return "Insufficient observations for calibration";
        case CalibrationError::SingularMatrix:
            return "Singular matrix encountered during computation";
        case CalibrationError::OptimizationFailed:
            return "Non-linear optimization failed to converge";
        case CalibrationError::InternalError:
            return "Internal processing error";
        default:
            return "Unknown error";
    }
}

inline bool operator!(CalibrationError err) {
    return err != CalibrationError::Ok;
}

inline bool ok(CalibrationError err) {
    return err == CalibrationError::Ok;
}

// 2D image point observation
struct Point2D {
    double x;
    double y;
};

// One frame observation: 3 collinear marker points projected onto the image plane
struct RodObservation {
    Point2D marker_a;  // First marker (proximal)
    Point2D marker_b;  // Second marker (middle, AB distance from A)
    Point2D marker_c;  // Third marker (distal, BC distance from B)
};

// All frame observations for a single camera
struct CameraObservations {
    const RodObservation* frames;  // Array of frame_count observations
    int frame_count;
};

// Camera intrinsic parameters (pinhole model with radial + tangential distortion)
struct CameraIntrinsics {
    double fx = 0.0;   // Focal length in pixels (x)
    double fy = 0.0;   // Focal length in pixels (y)
    double cx = 0.0;   // Principal point x
    double cy = 0.0;   // Principal point y
    double k1 = 0.0;   // Radial distortion coefficient 1
    double k2 = 0.0;   // Radial distortion coefficient 2
    double k3 = 0.0;   // Radial distortion coefficient 3
    double p1 = 0.0;   // Tangential distortion coefficient 1
    double p2 = 0.0;   // Tangential distortion coefficient 2
};

// Camera extrinsic parameters (world-to-camera transform)
struct CameraExtrinsics {
    double rotation[9] = {  // 3x3 rotation matrix, row-major
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    double translation[3] = {0.0, 0.0, 0.0};  // Translation vector
};

// Complete calibration result for one camera
struct CameraCalibration {
    CameraIntrinsics intrinsics;
    CameraExtrinsics extrinsics;
    double reprojection_error = 0.0;  // RMS reprojection error in pixels
};

// Configuration for three-point calibration
struct ThreePointConfig {
    double ab_distance = 150.0;   // Distance between marker A and B (mm)
    double bc_distance = 250.0;   // Distance between marker B and C (mm)
    int image_width = 0;          // Image width in pixels
    int image_height = 0;         // Image height in pixels
    int max_iterations = 100;     // Maximum LM optimization iterations
    double tolerance = 1e-6;      // Convergence tolerance
    bool fix_principal_point = false;  // Fix cx,cy to image center
    bool fix_aspect_ratio = false;     // Fix fx == fy
    bool estimate_distortion = true;   // Enable distortion estimation
};

// One-frame observation of the 3 triangle corner markers
struct TriangleObservation {
    Point2D marker_o;  // Right-angle vertex (world origin)
    Point2D marker_x;  // Short leg endpoint (X-axis direction)
    Point2D marker_y;  // Long leg endpoint (Y-axis direction)
};

// Configuration for right-triangle world coordinate registration
struct TriangleConfig {
    double ox_length = 300.0;   // OX distance in mm (short leg, 3 units)
    double oy_length = 400.0;   // OY distance in mm (long leg, 4 units)
    int image_width = 0;
    int image_height = 0;
};

// Result of world coordinate registration
struct WorldRegistration {
    CameraExtrinsics world_to_camera;  // Transform from world frame to camera 0's frame
    double fit_error = 0.0;           // RMS residual of rigid fit (mm)
};

// ============================================================
//  Checkerboard calibration types
// ============================================================

// Detected chessboard corners from a single image
struct CheckerboardCorners {
    Point2D* points;       // corners in row-major order (rows*cols points)
    int rows;              // inner corner rows
    int cols;              // inner corner columns
    bool valid;            // detection succeeded
};

// Configuration for checkerboard detection
struct CheckerboardConfig {
    int cols = 9;          // inner corners in X direction
    int rows = 6;          // inner corners in Y direction
    double square_size = 1.0; // square side length in mm
    bool sub_pixel = true; // enable sub-pixel refinement
    int sub_pixel_window = 11;  // half-window for sub-pixel refinement
    int sub_pixel_iters = 30;   // max sub-pixel iterations
    double sub_pixel_eps = 0.001; // sub-pixel convergence tolerance
};

// Multiple checkerboard views for Zhang calibration
struct CheckerboardViews {
    const CheckerboardCorners* corners;  // array of views
    int view_count;
};

// DLT: N 3D world points → their 2D image projections
struct DltCorrespondence {
    double world_x, world_y, world_z;  // 3D world coordinates
    double image_x, image_y;           // 2D image coordinates
};

struct DltParams {
    const DltCorrespondence* correspondences;
    int count;  // must be >= 6
};

// DLT output: 3x4 camera matrix P = K*[R|t]
struct DltResult {
    double P[12];  // 3x4 camera matrix, row-major
    CameraIntrinsics K;
    CameraExtrinsics extrinsics;
    double residual = 0.0;
};

// Stereo calibration config
struct StereoCalibrateParams {
    const CheckerboardCorners* left_corners;
    const CheckerboardCorners* right_corners;
    int view_count;
    const CameraIntrinsics* left_intrinsics;   // from individual calibration
    const CameraIntrinsics* right_intrinsics;  // from individual calibration
    bool fix_intrinsics = true;
};

// Bundle adjustment config
struct BundleAdjustParams {
    const CameraObservations* cameras;
    int camera_count;
    int frame_count;
    CameraIntrinsics* intrinsics;  // in/out
    CameraExtrinsics* extrinsics;  // in/out per camera
    double ab_distance = 150.0;    // rod AB distance in mm
    double bc_distance = 250.0;    // rod BC distance in mm
    int max_iterations = 50;
    double tolerance = 1e-6;
    bool fix_intrinsics = true;
};

// ============================================================
//  Large-scale calibration types
// ============================================================

// Camera graph: which cameras see common 3D points
struct CameraPair {
    int cam_i;           // camera index
    int cam_j;           // camera index
    int common_pts;      // number of shared 3D observations
    double score;        // matching quality score
};

struct CameraGraph {
    CameraPair* pairs;       // sorted by common_pts descending
    int pair_count;
    int camera_count;
    int** visibility;        // [cam][point_idx] = observed? (for sparse access)
    int* vis_counts;         // per-camera observed point count
    int total_points;
};

// PnP solver: N 3D→2D correspondences → camera pose
struct PnPParams {
    const Point2D* image_pts;      // N 2D image observations
    const double* world_pts;       // N 3D world coordinates (3*N doubles)
    int point_count;               // must be >= 4
    const CameraIntrinsics* intrinsics;
};

struct PnPResult {
    CameraExtrinsics pose;         // R, t
    int inliers;                   // number of inliers (for RANSAC mode)
    double rms_error;
};

// Incremental SfM
struct SfMView {
    const CheckerboardCorners* corners;
    int camera_id;
};

struct SfMConfig {
    const SfMView* views;
    int view_count;
    const CameraIntrinsics* intrinsics;  // per-view (or shared)
    int intrinsics_count;                // 1 for shared, view_count for per-camera
    double square_size;
    int cols, rows;                     // checkerboard dimensions
    int max_iterations = 50;
    double tolerance = 1e-6;
};

struct SfMResult {
    CameraExtrinsics* extrinsics;       // per-view pose (view_count)
    double* points_3d;                  // reconstructed 3D points
    int point_count;
    double rms_error;
    bool valid;
};

// Sparse BA: Schur-complement-based joint optimization
// Uses analytical Jacobian for efficiency on 50-100+ cameras
struct SparseBAParams {
    CameraIntrinsics* intrinsics;    // C cameras × intrinsics array
    CameraExtrinsics* extrinsics;    // C cameras × extrinsics array
    int camera_count;
    double* points_3d;               // P points × 3 coordinates
    int point_count;
    const Point2D* observations;     // all 2D observations, flat array
    const int* obs_camera;           // camera index per observation
    const int* obs_point;            // point index per observation
    int obs_count;
    int max_iterations = 30;
    double tolerance = 1e-6;
    bool fix_intrinsics = true;
};

}  // namespace calibration

#endif  // CALIBRATION_TYPES_HPP
