#ifndef CALIBRATION_ALGORITHMS_HPP
#define CALIBRATION_ALGORITHMS_HPP

#include "types.hpp"

#include <string>

namespace calibration {

// --- Metadata helpers ---

std::string algorithm_name(CalibrationAlgorithm algo);
int algorithm_window_size(CalibrationAlgorithm algo);

// --- Input validation ---

CalibrationError validate_calibration_inputs(const CameraObservations* cameras,
                                              int camera_count,
                                              const ThreePointConfig* config);

// --- Main dispatch ---

CalibrationError process_calibration(const CameraObservations* cameras,
                                      int camera_count,
                                      CameraCalibration* results,
                                      CalibrationAlgorithm algorithm,
                                      const ThreePointConfig* config);

// --- Individual algorithm functions ---

CalibrationError process_three_point(const CameraObservations* cameras,
                                      int camera_count,
                                      CameraCalibration* results,
                                      const ThreePointConfig* config);

// Right-triangle world coordinate registration
// Uses pre-calibrated cameras to register the world coordinate system.
CalibrationError process_right_triangle(const CameraObservations* cameras,
                                         int camera_count,
                                         const CameraCalibration* calibrations,
                                         const TriangleConfig* config,
                                         WorldRegistration* world_reg);

}  // namespace calibration

#endif  // CALIBRATION_ALGORITHMS_HPP
