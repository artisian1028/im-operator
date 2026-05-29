#include "calibration/algorithms.hpp"

#include <array>

namespace calibration {

// ============================================================
//  Metadata
// ============================================================

std::string algorithm_name(CalibrationAlgorithm algo) {
    switch (algo) {
        case CalibrationAlgorithm::THREE_POINT:           return "THREE_POINT (3-point rod)";
        case CalibrationAlgorithm::RIGHT_TRIANGLE:        return "RIGHT_TRIANGLE (world registration)";
        case CalibrationAlgorithm::CHECKERBOARD_DETECT:   return "CHECKERBOARD_DETECT (chessboard corners)";
        case CalibrationAlgorithm::CHECKERBOARD_CALIBRATE: return "CHECKERBOARD_CALIBRATE (Zhang)";
        case CalibrationAlgorithm::DLT:                   return "DLT (Direct Linear Transform)";
        case CalibrationAlgorithm::STEREO_CALIBRATE:      return "STEREO_CALIBRATE (dual-camera)";
        case CalibrationAlgorithm::BUNDLE_ADJUST:         return "BUNDLE_ADJUST (sparse LM)";
        default: return "Unknown";
    }
}

int algorithm_window_size(CalibrationAlgorithm /*algo*/) {
    return 0;  // Not image-based, no processing window
}

// ============================================================
//  Input validation
// ============================================================

CalibrationError validate_calibration_inputs(const CameraObservations* cameras,
                                              int camera_count,
                                              const ThreePointConfig* config) {
    if (!cameras || !config) return CalibrationError::NullInput;
    if (camera_count < 1) return CalibrationError::InvalidCameraCount;

    for (int c = 0; c < camera_count; ++c) {
        if (!cameras[c].frames && cameras[c].frame_count > 0) return CalibrationError::NullInput;
        if (cameras[c].frame_count < 3) return CalibrationError::InvalidFrameCount;
    }

    if (config->image_width <= 0 || config->image_height <= 0) return CalibrationError::InvalidConfiguration;
    if (config->ab_distance <= 0.0 || config->bc_distance <= 0.0) return CalibrationError::InvalidConfiguration;
    if (config->max_iterations <= 0) return CalibrationError::InvalidConfiguration;
    if (config->tolerance <= 0.0) return CalibrationError::InvalidConfiguration;

    return CalibrationError::Ok;
}

// ============================================================
//  Algorithm registry
// ============================================================

using AlgoFunc = CalibrationError (*)(const CameraObservations*, int, CameraCalibration*,
                                       const ThreePointConfig*);

struct AlgorithmEntry {
    CalibrationAlgorithm algorithm;
    AlgoFunc func;
};

// Registry only includes algorithms that share the same function signature.
// RIGHT_TRIANGLE has a different API (TriangleConfig* instead of multi-camera)
// and is dispatched separately via process_calibration overload.
static constexpr int kAlgorithmCount = 1;

static const std::array<AlgorithmEntry, kAlgorithmCount> kRegistry = {{
    {CalibrationAlgorithm::THREE_POINT, process_three_point},
}};

static_assert(kRegistry.size() == kAlgorithmCount,
              "kRegistry size must match kAlgorithmCount");

static AlgoFunc find_func(CalibrationAlgorithm algo) {
    for (const auto& entry : kRegistry) {
        if (entry.algorithm == algo) return entry.func;
    }
    return nullptr;
}

// ============================================================
//  Main dispatch
// ============================================================

CalibrationError process_calibration(const CameraObservations* cameras,
                                      int camera_count,
                                      CameraCalibration* results,
                                      CalibrationAlgorithm algorithm,
                                      const ThreePointConfig* config) {
    CalibrationError err = validate_calibration_inputs(cameras, camera_count, config);
    if (err != CalibrationError::Ok) return err;

    if (!results) return CalibrationError::NullInput;

    AlgoFunc func = find_func(algorithm);
    if (!func) return CalibrationError::InternalError;

    return func(cameras, camera_count, results, config);
}

}  // namespace calibration
