#ifndef CALIBRATION_ALGORITHMS_HPP
#define CALIBRATION_ALGORITHMS_HPP

#include "types.hpp"

#include <string>

namespace calibration {

// --- Metadata helpers ---

std::string algorithm_name(CalibrationAlgorithm algo);
int algorithm_window_size(CalibrationAlgorithm algo);

// CUDA support
bool has_cuda();
const char* cuda_device_name();

// --- Input validation ---

CalibrationError validate_calibration_inputs(const CameraObservations* cameras,
                                              int camera_count,
                                              const ThreePointConfig* config);

// --- Main dispatch (legacy) ---

CalibrationError process_calibration(const CameraObservations* cameras,
                                      int camera_count,
                                      CameraCalibration* results,
                                      CalibrationAlgorithm algorithm,
                                      const ThreePointConfig* config);

// --- Individual algorithm functions (legacy) ---

CalibrationError process_three_point(const CameraObservations* cameras,
                                      int camera_count,
                                      CameraCalibration* results,
                                      const ThreePointConfig* config);

CalibrationError process_right_triangle(const CameraObservations* cameras,
                                         int camera_count,
                                         const CameraCalibration* calibrations,
                                         const TriangleConfig* config,
                                         WorldRegistration* world_reg);

// ============================================================
//  NEW: Checkerboard / Zhang / DLT / Stereo / Bundle
// ============================================================

// Detect corners: input RGB or grayscale image, output sub-pixel corners
CalibrationError process_checkerboard_detect(const uint8_t* image,
                                              int width, int height,
                                              int channels, int bit_depth,
                                              const CheckerboardConfig* config,
                                              CheckerboardCorners* corners);

// Zhang calibration: multiple checkerboard views → camera intrinsics + extrinsics
CalibrationError process_checkerboard_calibrate(const CheckerboardCorners* corners,
                                                 int view_count,
                                                 int image_width, int image_height,
                                                 const CheckerboardConfig* config,
                                                 CameraIntrinsics* intrinsics,
                                                 CameraExtrinsics* extrinsics,
                                                 bool estimate_distortion = true);

// DLT: N 3D↔2D correspondences → 3x4 camera matrix
CalibrationError process_dlt(const DltParams* params, DltResult* result);

// Stereo calibration: checkerboard views from two cameras → stereo extrinsics
CalibrationError process_stereo_calibrate(const StereoCalibrateParams* params,
                                           CameraExtrinsics* stereo_R,
                                           CameraExtrinsics* stereo_t,
                                           double* rms_error = nullptr);

// Bundle adjustment: sparse LM joint optimization of all params
CalibrationError process_bundle_adjust(BundleAdjustParams* params,
                                        double* final_rms = nullptr);

// ============================================================
//  Large-scale calibration: 50-100+ cameras
// ============================================================

// Build camera visibility graph from checkerboard observations
CalibrationError process_camera_graph(const SfMView* views, int view_count,
                                       const CheckerboardConfig* config,
                                       CameraGraph* graph);

// PnP: N 3D↔2D correspondences → camera pose (with known intrinsics)
CalibrationError process_pnp_solver(const PnPParams* params, PnPResult* result);

// Incremental SfM: register all cameras and reconstruct sparse 3D points
CalibrationError process_incremental_sfm(const SfMConfig* config, SfMResult* result);

// Sparse Schur-complement Bundle Adjustment for large-scale camera networks
CalibrationError process_sparse_ba(SparseBAParams* params, double* final_rms = nullptr);

}  // namespace calibration

#endif  // CALIBRATION_ALGORITHMS_HPP
