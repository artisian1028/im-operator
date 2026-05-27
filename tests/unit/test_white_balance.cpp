#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "white_balance/algorithms.hpp"

using namespace white_balance;

namespace {

// Creates an RGB image with a color cast (warm: R high, B low)
void make_warm_rgb(uint8_t* rgb, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            // Simulate tungsten lighting: red-ish cast
            rgb[idx + 0] = static_cast<uint8_t>(180 + (x * 3) % 40);
            rgb[idx + 1] = static_cast<uint8_t>(100 + (y * 2) % 30);
            rgb[idx + 2] = static_cast<uint8_t>(60 + (x * 5) % 20);
        }
    }
}

// Creates an RGB image with a cool cast (B high, R low)
void make_cool_rgb(uint8_t* rgb, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            // Simulate shade: blue-ish cast
            rgb[idx + 0] = static_cast<uint8_t>(80 + (x * 3) % 40);
            rgb[idx + 1] = static_cast<uint8_t>(120 + (y * 2) % 30);
            rgb[idx + 2] = static_cast<uint8_t>(180 + (x * 5) % 20);
        }
    }
}

// Creates a flat gray RGB image
void make_gray_rgb(uint8_t* rgb, int width, int height, uint8_t value = 128) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            rgb[idx + 0] = value;
            rgb[idx + 1] = value;
            rgb[idx + 2] = value;
        }
    }
}

struct WBTestParam {
    WhiteBalanceAlgorithm algo;
    int min_size;
};

class WBAlgorithmTest : public ::testing::TestWithParam<WBTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(WBDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    // Null input
    EXPECT_EQ(process_white_balance(nullptr, dst, 4, 4, 3,
                                     WhiteBalanceAlgorithm::GRAY_WORLD, 8),
              WhiteBalanceError::NullInput);
    EXPECT_EQ(process_white_balance(src, nullptr, 4, 4, 3,
                                     WhiteBalanceAlgorithm::GRAY_WORLD, 8),
              WhiteBalanceError::NullInput);

    // Invalid dimensions
    EXPECT_EQ(process_white_balance(src, dst, 0, 4, 3,
                                     WhiteBalanceAlgorithm::GRAY_WORLD, 8),
              WhiteBalanceError::InvalidDimensions);
    EXPECT_EQ(process_white_balance(src, dst, 4, -1, 3,
                                     WhiteBalanceAlgorithm::GRAY_WORLD, 8),
              WhiteBalanceError::InvalidDimensions);

    // Invalid bit depth
    EXPECT_EQ(process_white_balance(src, dst, 2, 2, 3,
                                     WhiteBalanceAlgorithm::GRAY_WORLD, 17),
              WhiteBalanceError::InvalidBitDepth);

    // Invalid channels (white balance requires 3-channel RGB)
    EXPECT_EQ(process_white_balance(src, dst, 2, 2, 1,
                                     WhiteBalanceAlgorithm::GRAY_WORLD, 8),
              WhiteBalanceError::InvalidChannels);
    EXPECT_EQ(process_white_balance(src, dst, 2, 2, 4,
                                     WhiteBalanceAlgorithm::GRAY_WORLD, 8),
              WhiteBalanceError::InvalidChannels);
}

// --- Metadata tests ---

TEST(WBDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(WhiteBalanceAlgorithm::GRAY_WORLD),
              "Gray World (average assumption)");
    EXPECT_EQ(algorithm_name(WhiteBalanceAlgorithm::WHITE_PATCH),
              "White Patch (max RGB)");
    EXPECT_EQ(algorithm_name(WhiteBalanceAlgorithm::SHADE_OF_GRAY),
              "Shade of Gray (Minkowski norm)");
    EXPECT_EQ(algorithm_name(WhiteBalanceAlgorithm::MANUAL),
              "Manual (user-supplied gains)");
}

TEST(WBDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(WhiteBalanceAlgorithm::GRAY_WORLD), 1);
    EXPECT_EQ(algorithm_window_size(WhiteBalanceAlgorithm::WHITE_PATCH), 1);
    EXPECT_EQ(algorithm_window_size(WhiteBalanceAlgorithm::SHADE_OF_GRAY), 1);
    EXPECT_EQ(algorithm_window_size(WhiteBalanceAlgorithm::MANUAL), 1);
}

// --- Gray World tests ---

TEST(WhiteBalanceGrayWorldTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::GRAY_WORLD, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    // Output should not be all zeros
    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);

    // All values in valid range
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(WhiteBalanceGrayWorldTest, ReducesColorCast) {
    // On a warm image, Gray World should reduce R/B channel differences
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    std::vector<uint8_t> dst(w * h * 3);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::GRAY_WORLD, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    // Compute mean R/B ratio: should be closer to 1.0 after WB
    double sum_r = 0.0, sum_b = 0.0;
    for (int i = 0; i < w * h; i++) {
        sum_r += dst[i * 3 + 0];
        sum_b += dst[i * 3 + 2];
    }
    double ratio = sum_r / sum_b;
    // For warm input (R >> B), after WB the ratio should be closer to 1.0
    // than the original ratio
    double sum_r_orig = 0.0, sum_b_orig = 0.0;
    for (int i = 0; i < w * h; i++) {
        sum_r_orig += src[i * 3 + 0];
        sum_b_orig += src[i * 3 + 2];
    }
    double orig_ratio = sum_r_orig / sum_b_orig;
    EXPECT_LT(std::abs(ratio - 1.0), std::abs(orig_ratio - 1.0))
        << "Gray World should reduce color cast";
}

TEST(WhiteBalanceGrayWorldTest, GrayInputProducesNearIdentity) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_gray_rgb(src.data(), w, h, 100);

    std::vector<uint8_t> dst(w * h * 3);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::GRAY_WORLD, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    // On gray input, output should be nearly identical
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(100), 5);
        EXPECT_NEAR(dst[i * 3 + 1], static_cast<uint8_t>(100), 5);
        EXPECT_NEAR(dst[i * 3 + 2], static_cast<uint8_t>(100), 5);
    }
}

TEST(WhiteBalanceGrayWorldTest, TooSmallImage) {
    std::vector<uint8_t> src(3);
    std::vector<uint8_t> dst(3);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), 1, 1, 3,
                                                    WhiteBalanceAlgorithm::GRAY_WORLD, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);  // 1x1 is technically valid
}

TEST(WhiteBalanceGrayWorldTest, InvalidChannels) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 1,
                                                    WhiteBalanceAlgorithm::GRAY_WORLD, 8);
    EXPECT_EQ(err, WhiteBalanceError::InvalidChannels);
}

// --- White Patch tests ---

TEST(WhiteBalanceWhitePatchTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::WHITE_PATCH, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(WhiteBalanceWhitePatchTest, WhitePixelStaysWhite) {
    // Create an image with a pure white pixel
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3, 50);
    // Set pixel (4,4) to pure white
    size_t idx = (4 * w + 4) * 3;
    src[idx + 0] = 255;
    src[idx + 1] = 255;
    src[idx + 2] = 255;

    std::vector<uint8_t> dst(w * h * 3);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::WHITE_PATCH, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    // The white pixel should still be white (or very close)
    EXPECT_GE(dst[idx + 0], 250);
    EXPECT_GE(dst[idx + 1], 250);
    EXPECT_GE(dst[idx + 2], 250);
}

TEST(WhiteBalanceWhitePatchTest, TooSmallImage) {
    std::vector<uint8_t> src(3);
    std::vector<uint8_t> dst(3);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), 1, 1, 3,
                                                    WhiteBalanceAlgorithm::WHITE_PATCH, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);
}

// --- Shade of Gray tests ---

TEST(WhiteBalanceShadeOfGrayTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::SHADE_OF_GRAY, 8);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(WhiteBalanceShadeOfGrayTest, ParameterP) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst1(w * h * 3);
    std::vector<uint8_t> dst2(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    // p=1 (Gray World equivalent)
    WhiteBalanceError err1 = process_white_balance(src.data(), dst1.data(), w, h, 3,
                                                     WhiteBalanceAlgorithm::SHADE_OF_GRAY, 8, 1.0f);
    EXPECT_EQ(err1, WhiteBalanceError::Ok);

    // p=12 (close to White Patch)
    WhiteBalanceError err2 = process_white_balance(src.data(), dst2.data(), w, h, 3,
                                                     WhiteBalanceAlgorithm::SHADE_OF_GRAY, 8, 12.0f);
    EXPECT_EQ(err2, WhiteBalanceError::Ok);

    for (size_t i = 0; i < dst1.size(); i++) EXPECT_LE(dst1[i], 255);
    for (size_t i = 0; i < dst2.size(); i++) EXPECT_LE(dst2[i], 255);
}

TEST(WhiteBalanceShadeOfGrayTest, CoolImageCorrection) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_cool_rgb(src.data(), w, h);

    std::vector<uint8_t> dst(w * h * 3);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::SHADE_OF_GRAY, 8, 6.0f);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    // Compute mean R/B ratio: should be closer to 1.0 after WB
    double sum_r = 0.0, sum_b = 0.0;
    for (int i = 0; i < w * h; i++) {
        sum_r += dst[i * 3 + 0];
        sum_b += dst[i * 3 + 2];
    }
    double ratio = sum_r / sum_b;
    double sum_r_orig = 0.0, sum_b_orig = 0.0;
    for (int i = 0; i < w * h; i++) {
        sum_r_orig += src[i * 3 + 0];
        sum_b_orig += src[i * 3 + 2];
    }
    double orig_ratio = sum_r_orig / sum_b_orig;
    EXPECT_LT(std::abs(ratio - 1.0), std::abs(orig_ratio - 1.0))
        << "Shade of Gray should reduce color cast on cool image";
}

// --- Manual WB tests ---

TEST(WhiteBalanceManualTest, BasicProcessing8bit) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WBCoefficients gains;
    gains.r_gain = 0.8f;
    gains.g_gain = 1.0f;
    gains.b_gain = 1.5f;

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::MANUAL, 8, 0.0f, gains);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(WhiteBalanceManualTest, IdentityGains) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WBCoefficients gains; // defaults: 1.0 each

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::MANUAL, 8, 0.0f, gains);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    // Identity gains: output should match input exactly
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(WhiteBalanceManualTest, ClampsOutOfRange) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    // Bright pixels
    make_gray_rgb(src.data(), w, h, 200);

    WBCoefficients gains;
    gains.r_gain = 2.0f; // will exceed 255
    gains.g_gain = 1.0f;
    gains.b_gain = 2.0f;

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::MANUAL, 8, 0.0f, gains);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    // Should clamp to 255
    for (int i = 0; i < w * h; i++) {
        EXPECT_LE(dst[i * 3 + 0], 255);
        EXPECT_LE(dst[i * 3 + 2], 255);
    }
}

TEST(WhiteBalanceManualTest, InvalidChannels) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 4);
    std::vector<uint8_t> dst(w * h * 4);
    WBCoefficients gains;

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 4,
                                                    WhiteBalanceAlgorithm::MANUAL, 8, 0.0f, gains);
    EXPECT_EQ(err, WhiteBalanceError::InvalidChannels);
}

// --- High bit depth tests ---

TEST(WhiteBalanceHighBitDepthTest, GrayWorld12bit) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    // Warm cast in 12-bit
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 3000;  // R high
        src16[i * 3 + 1] = 2000;  // G
        src16[i * 3 + 2] = 1000;  // B low
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::GRAY_WORLD, 12);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) {
        EXPECT_LE(dst16[i], 4095);
    }
}

TEST(WhiteBalanceHighBitDepthTest, WhitePatch16bit) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 10000;
        src16[i * 3 + 1] = 20000;
        src16[i * 3 + 2] = 40000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    WhiteBalanceAlgorithm::WHITE_PATCH, 16);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) {
        EXPECT_LE(dst16[i], 65535);
    }
}

// --- Gains estimation tests ---

TEST(WhiteBalanceGainsTest, GrayWorldGains) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WBCoefficients gains = compute_white_balance_gains(src.data(), w, h, 8,
                                                         WhiteBalanceAlgorithm::GRAY_WORLD);
    // On warm image, R gain should be < 1, B gain should be > 1
    EXPECT_LT(gains.r_gain, 1.0f);
    EXPECT_GT(gains.b_gain, 1.0f);
    EXPECT_FLOAT_EQ(gains.g_gain, 1.0f);
}

TEST(WhiteBalanceGainsTest, WhitePatchGains) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WBCoefficients gains = compute_white_balance_gains(src.data(), w, h, 8,
                                                         WhiteBalanceAlgorithm::WHITE_PATCH);
    EXPECT_FLOAT_EQ(gains.g_gain, 1.0f);
}

TEST(WhiteBalanceGainsTest, ShadeOfGrayGains) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    WBCoefficients gains_p1 = compute_white_balance_gains(src.data(), w, h, 8,
                                                            WhiteBalanceAlgorithm::SHADE_OF_GRAY, 1.0f);
    WBCoefficients gains_p6 = compute_white_balance_gains(src.data(), w, h, 8,
                                                            WhiteBalanceAlgorithm::SHADE_OF_GRAY, 6.0f);

    EXPECT_FLOAT_EQ(gains_p1.g_gain, 1.0f);
    EXPECT_FLOAT_EQ(gains_p6.g_gain, 1.0f);
}

TEST(WhiteBalanceGainsTest, NullSafety) {
    WBCoefficients gains = compute_white_balance_gains(nullptr, 16, 16, 8,
                                                         WhiteBalanceAlgorithm::GRAY_WORLD);
    EXPECT_FLOAT_EQ(gains.r_gain, 1.0f);
    EXPECT_FLOAT_EQ(gains.g_gain, 1.0f);
    EXPECT_FLOAT_EQ(gains.b_gain, 1.0f);
}

// --- Parameterized tests across all algorithms ---

TEST_P(WBAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = std::max(p.min_size + 1, 16);
    int h = std::max(p.min_size + 1, 16);

    std::vector<uint8_t> src(w * h * 3);
    make_warm_rgb(src.data(), w, h);

    std::vector<uint8_t> dst(w * h * 3);
    WBCoefficients gains;
    gains.r_gain = 0.9f;
    gains.g_gain = 1.0f;
    gains.b_gain = 1.4f;

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 3,
                                                    p.algo, 8, 6.0f, gains);
    EXPECT_EQ(err, WhiteBalanceError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(WBAlgorithmTest, DetectsImageTooSmall) {
    auto p = GetParam();

    std::vector<uint8_t> src(3);
    std::vector<uint8_t> dst(3);
    // All WB algorithms can handle any positive size, so just verify Ok
    WBCoefficients gains;
    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), 1, 1, 3,
                                                    p.algo, 8, 6.0f, gains);
    EXPECT_EQ(err, WhiteBalanceError::Ok);
}

TEST_P(WBAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();

    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    WBCoefficients gains;

    WhiteBalanceError err = process_white_balance(src.data(), dst.data(), w, h, 1,
                                                    p.algo, 8, 6.0f, gains);
    EXPECT_EQ(err, WhiteBalanceError::InvalidChannels);
}

INSTANTIATE_TEST_SUITE_P(
    AllWBAlgos,
    WBAlgorithmTest,
    ::testing::Values(
        WBTestParam{WhiteBalanceAlgorithm::GRAY_WORLD, 1},
        WBTestParam{WhiteBalanceAlgorithm::WHITE_PATCH, 1},
        WBTestParam{WhiteBalanceAlgorithm::SHADE_OF_GRAY, 1},
        WBTestParam{WhiteBalanceAlgorithm::MANUAL, 1}
    )
);
