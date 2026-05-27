#include "color_calibration/algorithms.hpp"
#include "common.hpp"
#include <cstring>

namespace color_calibration {

// ============================================================
//  Metadata
// ============================================================

std::string algorithm_name(ColorCalibrationAlgorithm algo) {
    switch (algo) {
        case ColorCalibrationAlgorithm::DETECT_CHART:            return "Detect Color Chart";
        case ColorCalibrationAlgorithm::EXTRACT_PATCHES:          return "Extract Patches";
        case ColorCalibrationAlgorithm::SOLVE_CCM:                return "Solve CCM";
        case ColorCalibrationAlgorithm::GENERATE_LINEARIZATION:   return "Generate Linearization LUT";
        default: return "Unknown";
    }
}

// ============================================================
//  ColorChecker Classic reference values (sRGB D65, linearized)
//  Values from published spectral measurements, converted to
//  linear sRGB primaries under D65 illuminant.
// ============================================================

void get_colorchecker_reference(PatchColor refs[24]) {
    // Reference values for X-Rite ColorChecker Classic under D65
    // These are linear sRGB values (not gamma-encoded!)
    // Source: color-science.org / published ColorChecker data

    const float values[24][3] = {
        // Row 1
        {0.400f, 0.350f, 0.255f}, // 1.  Dark Skin
        {0.725f, 0.610f, 0.510f}, // 2.  Light Skin
        {0.300f, 0.410f, 0.635f}, // 3.  Blue Sky
        {0.255f, 0.400f, 0.235f}, // 4.  Foliage
        {0.410f, 0.380f, 0.670f}, // 5.  Blue Flower (Blue C)
        {0.365f, 0.575f, 0.520f}, // 6.  Bluish Green

        // Row 2
        {0.630f, 0.400f, 0.135f}, // 7.  Orange
        {0.320f, 0.325f, 0.610f}, // 8.  Purplish Blue (Purple)
        {0.530f, 0.235f, 0.205f}, // 9.  Moderate Red
        {0.315f, 0.210f, 0.355f}, // 10. Purple
        {0.545f, 0.610f, 0.140f}, // 11. Yellow Green
        {0.640f, 0.545f, 0.060f}, // 12. Orange Yellow

        // Row 3
        {0.130f, 0.215f, 0.475f}, // 13. Blue
        {0.255f, 0.430f, 0.180f}, // 14. Green
        {0.410f, 0.120f, 0.130f}, // 15. Red
        {0.655f, 0.585f, 0.020f}, // 16. Yellow
        {0.525f, 0.265f, 0.520f}, // 17. Magenta
        {0.180f, 0.440f, 0.515f}, // 18. Cyan

        // Row 4 — Gray scale
        {0.810f, 0.810f, 0.810f}, // 19. White (90%)
        {0.525f, 0.525f, 0.525f}, // 20. Neutral 8 (59%)
        {0.315f, 0.315f, 0.315f}, // 21. Neutral 6.5 (36%)
        {0.175f, 0.175f, 0.175f}, // 22. Neutral 5 (20%)
        {0.078f, 0.078f, 0.078f}, // 23. Neutral 3.5 (9%)
        {0.028f, 0.028f, 0.028f}, // 24. Black (3%)
    };

    for (int i = 0; i < 24; i++) {
        refs[i].r = values[i][0];
        refs[i].g = values[i][1];
        refs[i].b = values[i][2];
    }
}

// ============================================================
//  Convenience: full pipeline
// ============================================================

ColorCalibrationError calibrate_from_chart(const uint8_t* input,
                                             int width, int height,
                                             int channels, int bit_depth,
                                             ccm::CCMatrix3x3* out_matrix,
                                             float* out_error) {
    if (!input || !out_matrix) return ColorCalibrationError::NullInput;

    // Step 1: Detect
    ChartDetection detection;
    auto e1 = process_detect_chart(input, width, height, channels, bit_depth, &detection);
    if (e1 != ColorCalibrationError::Ok) return e1;

    // Step 2: Extract
    ChartMeasurements measured;
    auto e2 = process_extract_patches(input, width, height, channels, bit_depth,
                                       &detection, &measured);
    if (e2 != ColorCalibrationError::Ok) return e2;

    // Step 3: Get reference
    PatchColor reference[24];
    get_colorchecker_reference(reference);

    // Step 4: Solve
    SolveCCMParams sp;
    sp.measured = measured.colors;
    sp.reference = reference;
    sp.patch_count = 24;
    sp.matrix_type = MatrixType::LINEAR_3X3;

    SolvedMatrix sm;
    auto e3 = process_solve_ccm(sp, &sm);
    if (e3 != ColorCalibrationError::Ok) return e3;

    // Copy result to CCMatrix3x3
    for (int i = 0; i < 9; i++) out_matrix->m[i] = sm.m[i];

    // Compute residual error if requested
    if (out_error) {
        float total_err = 0.0f;
        for (int i = 0; i < measured.count; i++) {
            float r = measured.colors[i].r, g = measured.colors[i].g, b = measured.colors[i].b;
            float pr = sm.m[0]*r + sm.m[1]*g + sm.m[2]*b;
            float pg = sm.m[3]*r + sm.m[4]*g + sm.m[5]*b;
            float pb = sm.m[6]*r + sm.m[7]*g + sm.m[8]*b;
            float dr = pr - reference[i].r, dg = pg - reference[i].g, db = pb - reference[i].b;
            total_err += dr*dr + dg*dg + db*db;
        }
        *out_error = std::sqrt(total_err / static_cast<float>(measured.count * 3));
    }

    return ColorCalibrationError::Ok;
}

ColorCalibrationError linearize_from_chart(const uint8_t* input,
                                             int width, int height,
                                             int channels, int bit_depth,
                                             LinearizationLUT* out_lut) {
    if (!input || !out_lut) return ColorCalibrationError::NullInput;

    // Step 1: Detect
    ChartDetection detection;
    auto e1 = process_detect_chart(input, width, height, channels, bit_depth, &detection);
    if (e1 != ColorCalibrationError::Ok) return e1;

    // Step 2: Extract all patches
    ChartMeasurements measured;
    auto e2 = process_extract_patches(input, width, height, channels, bit_depth,
                                       &detection, &measured);
    if (e2 != ColorCalibrationError::Ok) return e2;

    // Step 3: Only use gray patches (indices 18-23, 0-based)
    PatchColor gray_measured[6], gray_ref[6];
    PatchColor refs[24];
    get_colorchecker_reference(refs);

    for (int i = 0; i < 6; i++) {
        gray_measured[i] = measured.colors[18 + i];
        gray_ref[i] = refs[18 + i];
    }

    // Step 4: Generate linearization
    LinearizationParams lp;
    lp.measured = gray_measured;
    lp.reference = gray_ref;
    lp.gray_count = 6;

    return process_generate_linearization(lp, out_lut);
}

} // namespace color_calibration
