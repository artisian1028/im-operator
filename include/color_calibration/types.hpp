#ifndef COLOR_CALIBRATION_TYPES_HPP
#define COLOR_CALIBRATION_TYPES_HPP

#include <cstdint>
#include <vector>
#include <string>

namespace color_calibration {

enum class ColorCalibrationAlgorithm {
    DETECT_CHART,           // Detect X-Rite ColorChecker Classic in image
    EXTRACT_PATCHES,        // Extract average color from each detected patch
    SOLVE_CCM,              // Solve optimal CCM via least squares
    GENERATE_LINEARIZATION  // Generate linearization LUT from gray patches
};

enum class ColorCalibrationError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ChartNotFound,
    InsufficientPatches,
    SingularMatrix,
    InternalError
};

inline const char* color_calibration_error_message(ColorCalibrationError err) {
    switch (err) {
        case ColorCalibrationError::Ok:                  return "Success";
        case ColorCalibrationError::NullInput:           return "Null input pointer";
        case ColorCalibrationError::InvalidDimensions:    return "Invalid image dimensions";
        case ColorCalibrationError::InvalidBitDepth:      return "Invalid bit depth (must be 8-16)";
        case ColorCalibrationError::InvalidChannels:      return "Invalid channel count (must be 3)";
        case ColorCalibrationError::ChartNotFound:        return "Color chart not found in image";
        case ColorCalibrationError::InsufficientPatches:  return "Too few patches extracted";
        case ColorCalibrationError::SingularMatrix:       return "Singular matrix in solver";
        case ColorCalibrationError::InternalError:       return "Internal processing error";
        default:                                          return "Unknown error";
    }
}

inline bool operator!(ColorCalibrationError err) { return err != ColorCalibrationError::Ok; }
inline bool ok(ColorCalibrationError err) { return err == ColorCalibrationError::Ok; }
inline bool is_valid_bit_depth(int bd) { return bd >= 8 && bd <= 16; }
inline bool is_valid_dimensions(int w, int h) { return w > 0 && h > 0; }

// A detected patch region in relative coordinates (0-1 range)
struct PatchRegion {
    float cx;       // center x relative to image width [0, 1]
    float cy;       // center y relative to image height [0, 1]
    float half_w;   // half-width relative to image width
    float half_h;   // half-height relative to image height
};

// 24 patches detected from a chart image
struct ChartDetection {
    PatchRegion patches[24];
    bool valid[24]; // whether each patch was successfully detected
};

// RGB measurement for one patch
struct PatchColor {
    float r, g, b;
};

// 24 measured colors
struct ChartMeasurements {
    PatchColor colors[24];
    int count; // number of valid measurements
};

// Matrix type for solver
enum class MatrixType {
    LINEAR_3X3,
    LINEAR_4X3,
    POLYNOMIAL_3X9
};

// Resulting CCM (stores both type and data)
struct SolvedMatrix {
    MatrixType type;
    float m[27];    // max size (3x9), zeros for unused entries
    int rows;       // 3
    int cols;       // 3, 4, or 9
};

// Linearization LUT
struct LinearizationLUT {
    std::vector<float> lut; // size = lut_size, values in [0, 1]
    int lut_size = 256;
};

// Solver parameters
struct SolveCCMParams {
    const PatchColor* measured = nullptr;    // 24 measured RGB values
    const PatchColor* reference = nullptr;   // 24 reference RGB values
    int patch_count = 24;
    MatrixType matrix_type = MatrixType::LINEAR_3X3;
};

// Linearization parameters
struct LinearizationParams {
    const PatchColor* measured = nullptr;    // gray patch measurements (rows 19-24)
    const PatchColor* reference = nullptr;   // gray patch references
    int gray_count = 6;
    int lut_size = 256;
};

} // namespace color_calibration

#endif
