#ifndef CALIBRATION_TYPES_HPP
#define CALIBRATION_TYPES_HPP

#include <cstdint>

namespace calibration {

enum class CalibrationAlgorithm {
    THREE_POINT,    // Three-point rod calibration (intrinsics + extrinsics)
    RIGHT_TRIANGLE  // Right-triangle world coordinate registration
};

enum class CalibrationError {
    Ok = 0,
    NullInput,
    InvalidCameraCount,
    InvalidFrameCount,
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

}  // namespace calibration

#endif  // CALIBRATION_TYPES_HPP
