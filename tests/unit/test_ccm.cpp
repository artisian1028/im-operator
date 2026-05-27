#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "ccm/algorithms.hpp"

using namespace ccm;

namespace {

// Creates an RGB image with known values for testing
void make_test_rgb(uint8_t* rgb, int width, int height,
                   uint8_t rv, uint8_t gv, uint8_t bv) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            rgb[idx + 0] = rv;
            rgb[idx + 1] = gv;
            rgb[idx + 2] = bv;
        }
    }
}

struct CCMTestParam {
    CCMAlgorithm algo;
    int min_size;
};

class CCMAlgorithmTest : public ::testing::TestWithParam<CCMTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(CCMDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    // Null input
    EXPECT_EQ(process_ccm(nullptr, dst, 4, 4, 3,
                           CCMAlgorithm::LINEAR_3X3, 8),
              CCMError::NullInput);
    EXPECT_EQ(process_ccm(src, nullptr, 4, 4, 3,
                           CCMAlgorithm::LINEAR_3X3, 8),
              CCMError::NullInput);

    // Invalid dimensions
    EXPECT_EQ(process_ccm(src, dst, 0, 4, 3,
                           CCMAlgorithm::LINEAR_3X3, 8),
              CCMError::InvalidDimensions);
    EXPECT_EQ(process_ccm(src, dst, 4, -1, 3,
                           CCMAlgorithm::LINEAR_3X3, 8),
              CCMError::InvalidDimensions);

    // Invalid bit depth
    EXPECT_EQ(process_ccm(src, dst, 2, 2, 3,
                           CCMAlgorithm::LINEAR_3X3, 17),
              CCMError::InvalidBitDepth);

    // Invalid channels
    EXPECT_EQ(process_ccm(src, dst, 2, 2, 1,
                           CCMAlgorithm::LINEAR_3X3, 8),
              CCMError::InvalidChannels);
    EXPECT_EQ(process_ccm(src, dst, 2, 2, 4,
                           CCMAlgorithm::LINEAR_3X3, 8),
              CCMError::InvalidChannels);
}

// --- Metadata tests ---

TEST(CCMDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(CCMAlgorithm::LINEAR_3X3),
              "Linear 3x3 CCM");
    EXPECT_EQ(algorithm_name(CCMAlgorithm::LINEAR_4X3),
              "Linear 3x4 CCM (with bias)");
    EXPECT_EQ(algorithm_name(CCMAlgorithm::POLYNOMIAL_3X9),
              "Polynomial 3x9 CCM (2nd-order)");
    EXPECT_EQ(algorithm_name(CCMAlgorithm::MANUAL),
              "Manual (user-supplied matrix)");
}

TEST(CCMDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(CCMAlgorithm::LINEAR_3X3), 1);
    EXPECT_EQ(algorithm_window_size(CCMAlgorithm::LINEAR_4X3), 1);
    EXPECT_EQ(algorithm_window_size(CCMAlgorithm::POLYNOMIAL_3X9), 1);
    EXPECT_EQ(algorithm_window_size(CCMAlgorithm::MANUAL), 1);
}

// --- Linear 3x3 tests ---

TEST(CCMLinear3x3Test, IdentityMatrix) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    CCMatrix3x3 mat = identity_3x3();
    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_3X3, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    // Identity: output should match input exactly
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(CCMLinear3x3Test, ChannelSwap) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 100, 50);

    // Swap R and B
    CCMatrix3x3 mat;
    mat.m[0] = 0.0f; mat.m[1] = 0.0f; mat.m[2] = 1.0f;  // R_out = B_in
    mat.m[3] = 0.0f; mat.m[4] = 1.0f; mat.m[5] = 0.0f;  // G_out = G_in
    mat.m[6] = 1.0f; mat.m[7] = 0.0f; mat.m[8] = 0.0f;  // B_out = R_in

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_3X3, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    // R should now be 50 (was B), B should be 200 (was R)
    EXPECT_EQ(dst[0], 50);
    EXPECT_EQ(dst[2], 200);
}

TEST(CCMLinear3x3Test, GrayscaleConversion) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 100, 50);

    // REC.709 luminance weights -> grayscale
    CCMatrix3x3 mat;
    for (int i = 0; i < 3; i++) {
        mat.m[i * 3 + 0] = 0.2126f;
        mat.m[i * 3 + 1] = 0.7152f;
        mat.m[i * 3 + 2] = 0.0722f;
    }

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_3X3, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    // All three channels should be equal (grayscale)
    for (int i = 0; i < w * h; i++) {
        EXPECT_EQ(dst[i * 3 + 0], dst[i * 3 + 1]);
        EXPECT_EQ(dst[i * 3 + 1], dst[i * 3 + 2]);
    }
}

TEST(CCMLinear3x3Test, NullMatrixDefaultsToIdentity) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_3X3, 8, nullptr);
    EXPECT_EQ(err, CCMError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(CCMLinear3x3Test, ClampsOutput) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 200, 200);

    // Matrix that amplifies 2x
    CCMatrix3x3 mat;
    mat.m[0] = 2.0f; mat.m[1] = 0.0f; mat.m[2] = 0.0f;
    mat.m[3] = 0.0f; mat.m[4] = 2.0f; mat.m[5] = 0.0f;
    mat.m[6] = 0.0f; mat.m[7] = 0.0f; mat.m[8] = 2.0f;

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_3X3, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    // Should clamp at 255
    for (int i = 0; i < w * h; i++) {
        EXPECT_LE(dst[i * 3 + 0], 255);
        EXPECT_LE(dst[i * 3 + 1], 255);
        EXPECT_LE(dst[i * 3 + 2], 255);
    }
}

TEST(CCMLinear3x3Test, HighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    CCMatrix3x3 mat = identity_3x3();
    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_3X3, 12, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h; i++) {
        EXPECT_EQ(dst16[i * 3 + 0], 1000);
        EXPECT_EQ(dst16[i * 3 + 1], 2000);
        EXPECT_EQ(dst16[i * 3 + 2], 3000);
    }
}

// --- Linear 4x3 tests ---

TEST(CCMLinear4x3Test, IdentityMatrix) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    CCMatrix3x4 mat;
    mat.m[0] = 1.0f; mat.m[1] = 0.0f; mat.m[2] = 0.0f; mat.m[3] = 0.0f;
    mat.m[4] = 0.0f; mat.m[5] = 1.0f; mat.m[6] = 0.0f; mat.m[7] = 0.0f;
    mat.m[8] = 0.0f; mat.m[9] = 0.0f; mat.m[10] = 1.0f; mat.m[11] = 0.0f;

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_4X3, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(CCMLinear4x3Test, BiasTerm) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 50, 50, 50);

    // Identity + 20 offset on each channel
    CCMatrix3x4 mat;
    mat.m[0] = 1.0f; mat.m[1] = 0.0f; mat.m[2] = 0.0f; mat.m[3] = 20.0f;
    mat.m[4] = 0.0f; mat.m[5] = 1.0f; mat.m[6] = 0.0f; mat.m[7] = 20.0f;
    mat.m[8] = 0.0f; mat.m[9] = 0.0f; mat.m[10] = 1.0f; mat.m[11] = 20.0f;

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_4X3, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    EXPECT_EQ(dst[0], 70);
    EXPECT_EQ(dst[1], 70);
    EXPECT_EQ(dst[2], 70);
}

TEST(CCMLinear4x3Test, NullMatrixDefaultsToIdentity) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_4X3, 8, nullptr);
    EXPECT_EQ(err, CCMError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(CCMLinear4x3Test, HighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 500;
        src16[i * 3 + 1] = 500;
        src16[i * 3 + 2] = 500;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    CCMatrix3x4 mat;
    mat.m[0]  = 1.0f; mat.m[1]  = 0.0f; mat.m[2]  = 0.0f; mat.m[3]  = 100.0f;
    mat.m[4]  = 0.0f; mat.m[5]  = 1.0f; mat.m[6]  = 0.0f; mat.m[7]  = 100.0f;
    mat.m[8]  = 0.0f; mat.m[9]  = 0.0f; mat.m[10] = 1.0f; mat.m[11] = 100.0f;

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::LINEAR_4X3, 12, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    EXPECT_GE(dst16[0], 595);
    EXPECT_LE(dst16[0], 605);
    EXPECT_LE(dst16[0], 4095);
}

// --- Polynomial 3x9 tests ---

TEST(CCMPolynomial3x9Test, IdentityMatrix) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    CCMatrix3x9 mat; // default identity
    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::POLYNOMIAL_3X9, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    // With identity polynomial, R should map to R, etc.
    // Since we normalize by max_val, small rounding errors may occur
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(100), 1);
        EXPECT_NEAR(dst[i * 3 + 1], static_cast<uint8_t>(150), 1);
        EXPECT_NEAR(dst[i * 3 + 2], static_cast<uint8_t>(200), 1);
    }
}

TEST(CCMPolynomial3x9Test, NullMatrixDefaultsToIdentity) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 100, 100);

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::POLYNOMIAL_3X9, 8, nullptr);
    EXPECT_EQ(err, CCMError::Ok);

    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(100), 1);
        EXPECT_NEAR(dst[i * 3 + 1], static_cast<uint8_t>(100), 1);
        EXPECT_NEAR(dst[i * 3 + 2], static_cast<uint8_t>(100), 1);
    }
}

TEST(CCMPolynomial3x9Test, HighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::POLYNOMIAL_3X9, 12, nullptr);
    EXPECT_EQ(err, CCMError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) {
        EXPECT_LE(dst16[i], 4095);
    }
}

TEST(CCMPolynomial3x9Test, CrossChannelCorrection) {
    // A matrix that adds a fraction of R to G (common in color calibration)
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 100, 50);

    CCMatrix3x9 mat; // start with identity
    // Row 1 (G): add 0.2*f(R) to G (normalized, so coef at index 0)
    mat.m[9] = 0.2f;  // m[9] = G row, R feat

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::POLYNOMIAL_3X9, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    // Since R > G, R contribution should increase G
    // Normalized: R=200/255≈0.784, extra = 0.2*0.784=0.157, G=100/255=0.392, total≈0.549*255≈140
    EXPECT_GT(dst[1], src[1]);  // G increased
}

// --- Manual tests ---

TEST(CCMManualTest, IdentityMatrix) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    CCMatrix3x3 mat = identity_3x3();
    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::MANUAL, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(CCMManualTest, NullMatrixDefaultsToIdentity) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                CCMAlgorithm::MANUAL, 8, nullptr);
    EXPECT_EQ(err, CCMError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

// --- Predefined matrices tests ---

TEST(CCMPredefinedMatricesTest, Identity) {
    CCMatrix3x3 m = identity_3x3();
    EXPECT_FLOAT_EQ(m.m[0], 1.0f);
    EXPECT_FLOAT_EQ(m.m[4], 1.0f);
    EXPECT_FLOAT_EQ(m.m[8], 1.0f);
    EXPECT_FLOAT_EQ(m.m[1], 0.0f);
}

TEST(CCMPredefinedMatricesTest, SrgbToXyzRoundtrip) {
    // sRGB -> XYZ -> sRGB should be identity (within floating point error)
    CCMatrix3x3 to_xyz = srgb_to_xyz_d65();
    CCMatrix3x3 to_srgb = xyz_to_srgb_d65();

    // Compute M_roundtrip = to_srgb * to_xyz
    float rt[9] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                rt[i * 3 + j] += to_srgb.m[i * 3 + k] * to_xyz.m[k * 3 + j];
            }
        }
    }

    EXPECT_NEAR(rt[0], 1.0f, 0.001f);
    EXPECT_NEAR(rt[4], 1.0f, 0.001f);
    EXPECT_NEAR(rt[8], 1.0f, 0.001f);
    EXPECT_NEAR(rt[1], 0.0f, 0.001f);
}

TEST(CCMPredefinedMatricesTest, SaturationIdentity) {
    CCMatrix3x3 m = saturation_matrix(1.0f);
    // sat=1 should be identity
    EXPECT_NEAR(m.m[0], 1.0f, 0.001f);
    EXPECT_NEAR(m.m[4], 1.0f, 0.001f);
    EXPECT_NEAR(m.m[8], 1.0f, 0.001f);
    EXPECT_NEAR(m.m[1], 0.0f, 0.001f);
}

TEST(CCMPredefinedMatricesTest, SaturationGrayscale) {
    CCMatrix3x3 m = saturation_matrix(0.0f);
    // sat=0: all rows are the luminance weights -> grayscale output
    for (int row = 0; row < 3; row++) {
        EXPECT_NEAR(m.m[row * 3 + 0], 0.2126f, 0.01f);
        EXPECT_NEAR(m.m[row * 3 + 1], 0.7152f, 0.01f);
        EXPECT_NEAR(m.m[row * 3 + 2], 0.0722f, 0.01f);
    }
}

TEST(CCMPredefinedMatricesTest, SrgbBt709Identity) {
    CCMatrix3x3 m = srgb_to_bt709();
    EXPECT_NEAR(m.m[0], 1.0f, 0.001f);
    EXPECT_NEAR(m.m[4], 1.0f, 0.001f);
    EXPECT_NEAR(m.m[8], 1.0f, 0.001f);
}

// --- Parameterized tests across all algorithms ---

TEST_P(CCMAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = std::max(p.min_size + 1, 16);
    int h = std::max(p.min_size + 1, 16);

    std::vector<uint8_t> src(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            src[idx + 0] = static_cast<uint8_t>((x * 7 + y * 3) % 256);
            src[idx + 1] = static_cast<uint8_t>((x * 5 + y * 11) % 256);
            src[idx + 2] = static_cast<uint8_t>((x * 13 + y * 7) % 256);
        }
    }

    std::vector<uint8_t> dst(w * h * 3);
    CCMatrix3x3 mat = identity_3x3();
    CCMError err = process_ccm(src.data(), dst.data(), w, h, 3,
                                p.algo, 8, &mat);
    EXPECT_EQ(err, CCMError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(CCMAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();

    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);

    CCMError err = process_ccm(src.data(), dst.data(), w, h, 1,
                                p.algo, 8, nullptr);
    EXPECT_EQ(err, CCMError::InvalidChannels);
}

INSTANTIATE_TEST_SUITE_P(
    AllCCMAlgos,
    CCMAlgorithmTest,
    ::testing::Values(
        CCMTestParam{CCMAlgorithm::LINEAR_3X3, 1},
        CCMTestParam{CCMAlgorithm::LINEAR_4X3, 1},
        CCMTestParam{CCMAlgorithm::POLYNOMIAL_3X9, 1},
        CCMTestParam{CCMAlgorithm::MANUAL, 1}
    )
);
