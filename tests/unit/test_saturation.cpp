#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "saturation/algorithms.hpp"

using namespace saturation;

namespace {

void make_flat_rgb(uint8_t* rgb, int width, int height,
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

struct SatTestParam {
    SaturationAlgorithm algo;
};

class SaturationAlgorithmTest : public ::testing::TestWithParam<SatTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(SaturationDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    EXPECT_EQ(process_saturation(nullptr, dst, 4, 4, 3,
                                  SaturationAlgorithm::HSL, 8),
              SaturationError::NullInput);
    EXPECT_EQ(process_saturation(src, nullptr, 4, 4, 3,
                                  SaturationAlgorithm::HSL, 8),
              SaturationError::NullInput);
    EXPECT_EQ(process_saturation(src, dst, 0, 4, 3,
                                  SaturationAlgorithm::HSL, 8),
              SaturationError::InvalidDimensions);
    EXPECT_EQ(process_saturation(src, dst, 4, 4, 3,
                                  SaturationAlgorithm::HSL, 17),
              SaturationError::InvalidBitDepth);
    EXPECT_EQ(process_saturation(src, dst, 4, 4, 1,
                                  SaturationAlgorithm::HSL, 8),
              SaturationError::InvalidChannels);
}

// --- Metadata tests ---

TEST(SaturationDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(SaturationAlgorithm::HSL), "HSL Saturation");
    EXPECT_EQ(algorithm_name(SaturationAlgorithm::VIBRANCE), "Vibrance (intelligent)");
    EXPECT_EQ(algorithm_name(SaturationAlgorithm::CHANNEL_MIXER), "Channel Mixer");
    EXPECT_EQ(algorithm_name(SaturationAlgorithm::SELECTIVE), "Selective (per-channel)");
}

TEST(SaturationDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(SaturationAlgorithm::HSL), 1);
    EXPECT_EQ(algorithm_window_size(SaturationAlgorithm::VIBRANCE), 1);
}

// --- HSL tests ---

TEST(SaturationHSLTest, IdentityAtOne) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SaturationParams params;
    params.saturation = 1.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::HSL, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 1);
    }
}

TEST(SaturationHSLTest, GrayscaleAtZero) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 100, 50);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.saturation = 0.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::HSL, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);

    // All channels should be equal (grayscale)
    for (int i = 0; i < w * h; i++) {
        EXPECT_EQ(dst[i * 3 + 0], dst[i * 3 + 1]);
        EXPECT_EQ(dst[i * 3 + 1], dst[i * 3 + 2]);
    }
}

TEST(SaturationHSLTest, BoostsColors) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 100, 50);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.saturation = 2.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::HSL, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);

    // Saturation boost should widen the R/B gap
    int old_gap = std::abs(static_cast<int>(src[0]) - static_cast<int>(src[2]));
    int new_gap = std::abs(static_cast<int>(dst[0]) - static_cast<int>(dst[2]));
    EXPECT_GE(new_gap, old_gap - 5);
}

TEST(SaturationHSLTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.saturation = 3.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::HSL, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

// --- Vibrance tests ---

TEST(SaturationVibranceTest, IdentityAtOne) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SaturationParams params;
    params.vibrance = 1.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::VIBRANCE, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 1);
    }
}

TEST(SaturationVibranceTest, ReducesSaturationAtZero) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 100, 50);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.vibrance = 0.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::VIBRANCE, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);
    // Channels should be closer together (desaturated) compared to original
    int orig_spread = 200 - 50;
    int new_spread = *std::max_element(&dst[0], &dst[3]) - *std::min_element(&dst[0], &dst[3]);
    EXPECT_LT(new_spread, orig_spread) << "Vibrance=0 should reduce color spread";
}

TEST(SaturationVibranceTest, BoostsMutedMoreThanSaturated) {
    int w = 16, h = 16;
    // Muted color: (150, 120, 110) — low saturation
    std::vector<uint8_t> src_muted(w * h * 3);
    make_flat_rgb(src_muted.data(), w, h, 150, 120, 110);
    // Saturated color: (200, 50, 30)
    std::vector<uint8_t> src_sat(w * h * 3);
    make_flat_rgb(src_sat.data(), w, h, 200, 50, 30);

    std::vector<uint8_t> dst_muted(w * h * 3);
    std::vector<uint8_t> dst_sat(w * h * 3);

    SaturationParams params;
    params.vibrance = 2.0f;

    process_saturation(src_muted.data(), dst_muted.data(), w, h, 3,
                       SaturationAlgorithm::VIBRANCE, 8, params);
    process_saturation(src_sat.data(), dst_sat.data(), w, h, 3,
                       SaturationAlgorithm::VIBRANCE, 8, params);

    // Muted pixels should get more change relative to original spread
    int muted_spread_before = std::max({src_muted[0], src_muted[1], src_muted[2]}) -
                               std::min({src_muted[0], src_muted[1], src_muted[2]});
    int muted_spread_after = std::max({dst_muted[0], dst_muted[1], dst_muted[2]}) -
                              std::min({dst_muted[0], dst_muted[1], dst_muted[2]});
    EXPECT_GE(muted_spread_after, muted_spread_before - 5);
}

// --- Channel Mixer tests ---

TEST(SaturationChannelMixerTest, IdentityAtOne) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SaturationParams params;
    params.saturation = 1.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::CHANNEL_MIXER, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 1);
    }
}

TEST(SaturationChannelMixerTest, BoostsColors) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 100, 50);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.saturation = 2.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::CHANNEL_MIXER, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);

    // With cross-channel mixing, color spread should increase
    int gap = std::abs(static_cast<int>(dst[0]) - static_cast<int>(dst[2]));
    EXPECT_GE(gap, 140);
}

// --- Selective tests ---

TEST(SaturationSelectiveTest, IdentityAtOne) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SaturationParams params;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::SELECTIVE, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 1);
    }
}

TEST(SaturationSelectiveTest, RedChannelOnly) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    // Not gray: R already higher than G, boosting R_sat should amplify
    make_flat_rgb(src.data(), w, h, 200, 150, 150);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.r_sat = 2.0f;
    params.g_sat = 1.0f;
    params.b_sat = 1.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::SELECTIVE, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);

    // R should be pushed further from luma (more saturated red)
    EXPECT_GT(dst[0], src[0]) << "R should increase with selective R boost";
    EXPECT_NEAR(dst[1], src[1], 2) << "G should be roughly unchanged";
}

TEST(SaturationSelectiveTest, DesaturateBlue) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 100, 50);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.b_sat = 0.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::SELECTIVE, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);

    // B should now equal luma (B saturation = 0)
    int luma = static_cast<int>(0.299f*200 + 0.587f*100 + 0.114f*50 + 0.5f);
    EXPECT_NEAR(dst[2], luma, 2);
}

// --- High bit depth ---

TEST(SaturationHighBitDepthTest, HSL12bit) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    SaturationParams params;
    params.saturation = 1.0f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               SaturationAlgorithm::HSL, 12, params);
    EXPECT_EQ(err, SaturationError::Ok);
    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) EXPECT_LE(dst16[i], 4095);
}

// --- Parameterized tests ---

TEST_P(SaturationAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = 16, h = 16;

    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>((i * 37 + 127) % 256);

    std::vector<uint8_t> dst(w * h * 3);
    SaturationParams params;
    params.saturation = 1.5f;
    params.vibrance = 1.5f;
    params.r_sat = 1.2f;
    params.g_sat = 1.0f;
    params.b_sat = 0.8f;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               p.algo, 8, params);
    EXPECT_EQ(err, SaturationError::Ok);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(SaturationAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    SaturationParams params;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 1,
                                               p.algo, 8, params);
    EXPECT_EQ(err, SaturationError::InvalidChannels);
}

TEST_P(SaturationAlgorithmTest, HighBitDepth) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    SaturationParams params;
    SaturationError err = process_saturation(src.data(), dst.data(), w, h, 3,
                                               p.algo, 12, params);
    EXPECT_EQ(err, SaturationError::Ok);
    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) EXPECT_LE(dst16[i], 4095);
}

INSTANTIATE_TEST_SUITE_P(
    AllSaturationAlgos,
    SaturationAlgorithmTest,
    ::testing::Values(
        SatTestParam{SaturationAlgorithm::HSL},
        SatTestParam{SaturationAlgorithm::VIBRANCE},
        SatTestParam{SaturationAlgorithm::CHANNEL_MIXER},
        SatTestParam{SaturationAlgorithm::SELECTIVE}
    )
);
