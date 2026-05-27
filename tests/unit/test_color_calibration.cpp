#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "color_calibration/algorithms.hpp"
#include "ccm/types.hpp"

using namespace color_calibration;

// --- Synthetic chart builder ---

// Build a simple synthetic ColorChecker-like image with known patch colors
void build_synthetic_chart(std::vector<uint8_t>& img, int w, int h, int bit_depth) {
    int ch = 3;
    img.resize(static_cast<size_t>(w) * h * ch * (bit_depth <= 8 ? 1 : 2));
    int max_val = (1 << bit_depth) - 1;

    PatchColor refs[24];
    get_colorchecker_reference(refs);

    // Chart occupies 10%-90% of image area
    int x0 = w / 10, x1 = w * 9 / 10;
    int y0 = h / 10, y1 = h * 9 / 10;
    int cw = (x1 - x0) / 6;
    int ch_h = (y1 - y0) / 4;

    auto write_px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (bit_depth <= 8) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            img[idx+0] = r; img[idx+1] = g; img[idx+2] = b;
        } else {
            auto* d16 = reinterpret_cast<uint16_t*>(img.data());
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            d16[idx+0] = static_cast<uint16_t>(r);
            d16[idx+1] = static_cast<uint16_t>(g);
            d16[idx+2] = static_cast<uint16_t>(b);
        }
    };

    // Fill background (dark border = chart frame)
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            write_px(x, y, 10, 10, 10);

    // Fill patches
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 6; col++) {
            int i = row * 6 + col;
            int px = x0 + col * cw + cw / 4;
            int py0 = y0 + row * ch_h + ch_h / 4;
            int px1 = x0 + (col + 1) * cw - cw / 4;
            int py1 = y0 + (row + 1) * ch_h - ch_h / 4;

            // Apply sRGB gamma to reference for display
            auto linear_to_srgb = [](float v) -> float {
                if (v <= 0.0031308f) return 12.92f * v;
                return 1.055f * std::pow(v, 1.0f/2.4f) - 0.055f;
            };

            int rv = static_cast<int>(linear_to_srgb(refs[i].r) * max_val);
            int gv = static_cast<int>(linear_to_srgb(refs[i].g) * max_val);
            int bv = static_cast<int>(linear_to_srgb(refs[i].b) * max_val);

            for (int y = py0; y <= py1; y++)
                for (int x = px; x <= px1; x++)
                    write_px(x, y, static_cast<uint8_t>(rv), static_cast<uint8_t>(gv), static_cast<uint8_t>(bv));
        }
    }
}

// --- Detection tests ---

TEST(ColorCalibrationTest, DetectionOnSyntheticChart) {
    int w = 200, h = 150;
    std::vector<uint8_t> img;
    build_synthetic_chart(img, w, h, 8);

    ChartDetection det;
    auto err = process_detect_chart(img.data(), w, h, 3, 8, &det);
    EXPECT_EQ(err, ColorCalibrationError::Ok);
    for (int i = 0; i < 24; i++) EXPECT_TRUE(det.valid[i]);
}

TEST(ColorCalibrationTest, DetectionRejectsNullInput) {
    ChartDetection det;
    auto err = process_detect_chart(nullptr, 100, 100, 3, 8, &det);
    EXPECT_EQ(err, ColorCalibrationError::NullInput);
}

// --- Extraction tests ---

TEST(ColorCalibrationTest, ExtractionFromSyntheticChart) {
    int w = 200, h = 150;
    std::vector<uint8_t> img;
    build_synthetic_chart(img, w, h, 8);

    ChartDetection det;
    process_detect_chart(img.data(), w, h, 3, 8, &det);

    ChartMeasurements meas;
    auto err = process_extract_patches(img.data(), w, h, 3, 8, &det, &meas);
    EXPECT_EQ(err, ColorCalibrationError::Ok);
    EXPECT_GE(meas.count, 20);
}

// --- Solver tests ---

TEST(ColorCalibrationTest, SolveCCMIdentity) {
    // If measured == reference, the solver should return identity matrix
    PatchColor refs[24];
    get_colorchecker_reference(refs);

    SolveCCMParams p;
    p.measured = refs;
    p.reference = refs;
    p.patch_count = 24;
    p.matrix_type = MatrixType::LINEAR_3X3;

    SolvedMatrix sm;
    auto err = process_solve_ccm(p, &sm);
    EXPECT_EQ(err, ColorCalibrationError::Ok);

    // Identity matrix check: diagonal ≈ 1.0, off-diagonal ≈ 0.0
    EXPECT_NEAR(sm.m[0], 1.0f, 0.1f);
    EXPECT_NEAR(sm.m[4], 1.0f, 0.1f);
    EXPECT_NEAR(sm.m[8], 1.0f, 0.1f);
    EXPECT_NEAR(sm.m[1], 0.0f, 0.1f);
}

TEST(ColorCalibrationTest, SolveCCMPolynomial) {
    PatchColor refs[24];
    get_colorchecker_reference(refs);

    SolveCCMParams p;
    p.measured = refs;
    p.reference = refs;
    p.patch_count = 24;
    p.matrix_type = MatrixType::POLYNOMIAL_3X9;

    SolvedMatrix sm;
    auto err = process_solve_ccm(p, &sm);
    EXPECT_EQ(err, ColorCalibrationError::Ok);
    EXPECT_EQ(sm.rows, 3);
    EXPECT_EQ(sm.cols, 9);
}

TEST(ColorCalibrationTest, SolveCCMRejectsTooFewPatches) {
    PatchColor refs[24];
    get_colorchecker_reference(refs);
    SolveCCMParams p;
    p.measured = refs;
    p.reference = refs;
    p.patch_count = 2; // too few
    p.matrix_type = MatrixType::LINEAR_3X3;

    SolvedMatrix sm;
    auto err = process_solve_ccm(p, &sm);
    EXPECT_EQ(err, ColorCalibrationError::InsufficientPatches);
}

// --- Linearization tests ---

TEST(ColorCalibrationTest, LinearizationIdentity) {
    PatchColor gray_measured[6], gray_ref[6];
    get_colorchecker_reference(gray_ref);
    // Copy gray patches (indices 18-23) as-is
    for (int i = 0; i < 6; i++) gray_measured[i] = gray_ref[18 + i];

    LinearizationParams p;
    p.measured = gray_measured;
    p.reference = gray_ref + 18;
    p.gray_count = 6;

    LinearizationLUT lut;
    auto err = process_generate_linearization(p, &lut);
    EXPECT_EQ(err, ColorCalibrationError::Ok);
    EXPECT_EQ(lut.lut_size, 256);
    // LUT should be approximately identity for identity case
    for (int i = 0; i < 256; i++) {
        float t = static_cast<float>(i) / 255.0f;
        EXPECT_NEAR(lut.lut[i], t, 0.15f);
    }
}

// --- Full pipeline test ---

TEST(ColorCalibrationTest, FullPipelineToCCM) {
    int w = 200, h = 150;
    std::vector<uint8_t> img;
    build_synthetic_chart(img, w, h, 8);

    ccm::CCMatrix3x3 matrix;
    float error;
    auto err = calibrate_from_chart(img.data(), w, h, 3, 8, &matrix, &error);
    EXPECT_EQ(err, ColorCalibrationError::Ok);
    // Since we used gamma-encoded chart image but solver expects linear,
    // the matrix won't be exactly identity, but it should exist
    EXPECT_GE(error, 0.0f);
}

TEST(ColorCalibrationTest, FullPipelineToLinearization) {
    int w = 200, h = 150;
    std::vector<uint8_t> img;
    build_synthetic_chart(img, w, h, 8);

    LinearizationLUT lut;
    auto err = linearize_from_chart(img.data(), w, h, 3, 8, &lut);
    EXPECT_EQ(err, ColorCalibrationError::Ok);
    EXPECT_GT(lut.lut.size(), 0);
}

TEST(ColorCalibrationTest, ReferenceValues) {
    PatchColor refs[24];
    get_colorchecker_reference(refs);

    // Verify the reference values are plausible
    // White patch (index 18) should be bright
    EXPECT_GT(refs[18].r, 0.5f);
    EXPECT_GT(refs[18].g, 0.5f);
    EXPECT_GT(refs[18].b, 0.5f);

    // Black patch (index 23) should be dark
    EXPECT_LT(refs[23].r, 0.1f);
    EXPECT_LT(refs[23].g, 0.1f);
    EXPECT_LT(refs[23].b, 0.1f);

    // Gray patches should have R≈G≈B
    for (int i = 18; i < 24; i++) {
        EXPECT_NEAR(refs[i].r, refs[i].g, 0.01f);
        EXPECT_NEAR(refs[i].g, refs[i].b, 0.01f);
    }
}
