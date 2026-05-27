#ifndef COLOR_CALIBRATION_ALGORITHMS_HPP
#define COLOR_CALIBRATION_ALGORITHMS_HPP

#include "types.hpp"
#include "ccm/types.hpp"
#include "lut/types.hpp"
#include <cstdint>
#include <string>

namespace color_calibration {

std::string algorithm_name(ColorCalibrationAlgorithm algo);

// --- Detect chart ---
ColorCalibrationError process_detect_chart(const uint8_t* input,
                                            int width, int height,
                                            int channels, int bit_depth,
                                            ChartDetection* result);

// --- Extract patches ---
ColorCalibrationError process_extract_patches(const uint8_t* input,
                                               int width, int height,
                                               int channels, int bit_depth,
                                               const ChartDetection* detection,
                                               ChartMeasurements* result);

// --- Solve CCM ---
ColorCalibrationError process_solve_ccm(const SolveCCMParams& params,
                                          SolvedMatrix* result);

// --- Generate linearization LUT ---
ColorCalibrationError process_generate_linearization(const LinearizationParams& params,
                                                       LinearizationLUT* result);

// --- Convenience: full pipeline ---
// detect + extract + solve, outputs a CCMatrix3x3 ready for ccm module
ColorCalibrationError calibrate_from_chart(const uint8_t* input,
                                             int width, int height,
                                             int channels, int bit_depth,
                                             ccm::CCMatrix3x3* out_matrix,
                                             float* out_error = nullptr);

// detect + extract + generate linearization LUT
ColorCalibrationError linearize_from_chart(const uint8_t* input,
                                             int width, int height,
                                             int channels, int bit_depth,
                                             LinearizationLUT* out_lut);

// --- ColorChecker Classic reference values (sRGB D65, normalized 0-1) ---
void get_colorchecker_reference(PatchColor refs[24]);

} // namespace color_calibration

#endif
