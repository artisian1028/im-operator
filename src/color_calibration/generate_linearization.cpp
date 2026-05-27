#include "common.hpp"
#include "color_calibration/algorithms.hpp"
#include <cstring>
#include <vector>

namespace color_calibration {

// Generate linearization LUT from gray patch measurements.
//
// The ColorChecker bottom row (patches 19-24) are gray steps:
//   White (90%), N8 (59%), N6.5 (36%), N5 (20%), N3.5 (9%), Black (3%)
//
// We measure the actual RGB values for these patches, then interpolate
// to create a full LUT that maps measured values → linear values.
//
// For each gray patch, the linear reference luminance is known.
// We create a monotonic curve from these 6 points via piecewise linear
// interpolation (or cubic spline for smoothness).
ColorCalibrationError process_generate_linearization(const LinearizationParams& params,
                                                       LinearizationLUT* result) {
    if (!params.measured || !params.reference || !result) return ColorCalibrationError::NullInput;
    if (params.gray_count < 3) return ColorCalibrationError::InsufficientPatches;
    int lut_size = params.lut_size;
    if (lut_size < 16 || lut_size > 65536) lut_size = 256;

    result->lut_size = lut_size;
    result->lut.resize(lut_size);

    // Convert measurements to luminance (Y)
    int N = params.gray_count;
    std::vector<float> measured_y(N), ref_y(N);
    for (int i = 0; i < N; i++) {
        // Use green channel as luminance proxy (gray patches have R≈G≈B)
        measured_y[i] = params.measured[i].g;
        ref_y[i] = params.reference[i].g;
    }

    // Sort by measured value (should already be sorted but ensure)
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            if (measured_y[i] > measured_y[j]) {
                std::swap(measured_y[i], measured_y[j]);
                std::swap(ref_y[i], ref_y[j]);
            }
        }
    }

    // Piecewise linear interpolation from measured → reference mapping
    // For each LUT entry (representing a measured value), find the corresponding linear value
    for (int k = 0; k < lut_size; k++) {
        float t = static_cast<float>(k) / static_cast<float>(lut_size - 1);

        // Find which segment t falls in
        float linear_val;
        if (t <= measured_y[0]) {
            // Below first point: extrapolate linearly from first segment
            if (N >= 2 && measured_y[1] > measured_y[0]) {
                float slope = ref_y[1] / measured_y[1];
                linear_val = t * slope;
            } else {
                linear_val = 0.0f;
            }
        } else if (t >= measured_y[N - 1]) {
            // Above last point: extrapolate
            if (N >= 2 && measured_y[N-1] > measured_y[N-2]) {
                float slope = (ref_y[N-1] - ref_y[N-2]) / (measured_y[N-1] - measured_y[N-2]);
                linear_val = ref_y[N-1] + (t - measured_y[N-1]) * slope;
            } else {
                linear_val = 1.0f;
            }
        } else {
            // Interpolate between two points
            int i = 0;
            while (i < N - 1 && measured_y[i + 1] < t) i++;
            float frac = (t - measured_y[i]) / (measured_y[i + 1] - measured_y[i] + 1e-10f);
            linear_val = ref_y[i] + frac * (ref_y[i + 1] - ref_y[i]);
        }

        linear_val = std::max(0.0f, std::min(1.0f, linear_val));
        result->lut[k] = linear_val;
    }

    return ColorCalibrationError::Ok;
}

} // namespace color_calibration
