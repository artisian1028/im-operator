#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "tone/algorithms.hpp"

using namespace tone;

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

struct ToneTestParam {
    ToneAlgorithm algo;
};

class ToneAlgorithmTest : public ::testing::TestWithParam<ToneTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(ToneDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    EXPECT_EQ(process_tone(nullptr, dst, 4, 4, 3,
                            ToneAlgorithm::GAMMA, 8),
              ToneError::NullInput);
    EXPECT_EQ(process_tone(src, nullptr, 4, 4, 3,
                            ToneAlgorithm::GAMMA, 8),
              ToneError::NullInput);
    EXPECT_EQ(process_tone(src, dst, 0, 4, 3,
                            ToneAlgorithm::GAMMA, 8),
              ToneError::InvalidDimensions);
    EXPECT_EQ(process_tone(src, dst, 4, -1, 3,
                            ToneAlgorithm::GAMMA, 8),
              ToneError::InvalidDimensions);
    EXPECT_EQ(process_tone(src, dst, 4, 4, 3,
                            ToneAlgorithm::GAMMA, 17),
              ToneError::InvalidBitDepth);
    EXPECT_EQ(process_tone(src, dst, 4, 4, 1,
                            ToneAlgorithm::GAMMA, 8),
              ToneError::InvalidChannels);
}

// --- Metadata tests ---

TEST(ToneDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(ToneAlgorithm::GAMMA), "Gamma Correction");
    EXPECT_EQ(algorithm_name(ToneAlgorithm::S_CURVE), "S-Curve Contrast");
    EXPECT_EQ(algorithm_name(ToneAlgorithm::LEVELS), "Levels Adjustment");
    EXPECT_EQ(algorithm_name(ToneAlgorithm::CURVES_3POINT), "3-Point Curves");
    EXPECT_EQ(algorithm_name(ToneAlgorithm::SHADOWS_HIGHLIGHTS), "Shadows / Highlights");
}

TEST(ToneDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(ToneAlgorithm::GAMMA), 1);
    EXPECT_EQ(algorithm_window_size(ToneAlgorithm::S_CURVE), 1);
}

// --- Gamma tests ---

TEST(ToneGammaTest, IdentityAtGammaOne) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    ToneParams params;
    params.gamma = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::GAMMA, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 1);
    }
}

TEST(ToneGammaTest, BrightensDarkImage) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 50, 50, 50);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.gamma = 2.2f; // brightens midtones
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::GAMMA, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_GT(dst[0], 50); // should be brighter
}

TEST(ToneGammaTest, DarkensBrightImage) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 200, 200);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.gamma = 0.45f; // darkens midtones
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::GAMMA, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_LT(dst[0], 200); // should be darker
}

TEST(ToneGammaTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.gamma = 3.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::GAMMA, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(ToneGammaTest, HighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    ToneParams params;
    params.gamma = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::GAMMA, 12, params);
    EXPECT_EQ(err, ToneError::Ok);
    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) EXPECT_LE(dst16[i], 4095);
}

// --- S-Curve tests ---

TEST(ToneSCurveTest, IdentityAtZeroContrast) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 128, 128, 128);

    ToneParams params;
    params.contrast = 0.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::S_CURVE, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(128), 2);
    }
}

TEST(ToneSCurveTest, IncreasesContrast) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    // Dark half / bright half
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            uint8_t v = (x < w / 2) ? 80 : 180;
            src[idx + 0] = v;
            src[idx + 1] = v;
            src[idx + 2] = v;
        }
    }

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.contrast = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::S_CURVE, 8, params);
    EXPECT_EQ(err, ToneError::Ok);

    // After S-curve, dark should be darker or equal, bright should be brighter or equal
    int dark_val = dst[0];
    int bright_val = dst[(w / 2 + 1) * 3];
    EXPECT_LE(dark_val, 90);
    EXPECT_GE(bright_val, 180);
}

TEST(ToneSCurveTest, ShadowHighlightControl) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 30, 30, 30); // very dark

    ToneParams params;
    params.contrast = 0.0f;
    params.shadows = 1.0f; // lift shadows
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::S_CURVE, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_GT(dst[0], 30); // shadows should be lifted
}

// --- Levels tests ---

TEST(ToneLevelsTest, IdentityAtDefaults) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    ToneParams params; // all defaults: black=0, white=1, mid=0.5
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::LEVELS, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 1);
    }
}

TEST(ToneLevelsTest, BlackClipping) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 30, 30, 30);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.black_point = 0.2f; // clip below 20% (~51)
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::LEVELS, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_LT(dst[0], 30); // values below black point are compressed
}

TEST(ToneLevelsTest, WhiteClipping) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 230, 230, 230);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.white_point = 0.8f; // clip above 80% (~204)
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::LEVELS, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_GT(dst[0], 230); // values above white point are expanded
}

// --- Curves 3-point tests ---

TEST(ToneCurves3PointTest, IdentityAtDefaults) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    ToneParams params;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::CURVES_3POINT, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 2);
    }
}

TEST(ToneCurves3PointTest, ContrastCurve) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            uint8_t v = (x < w / 2) ? 60 : 200;
            src[idx + 0] = v;
            src[idx + 1] = v;
            src[idx + 2] = v;
        }
    }

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.contrast = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::CURVES_3POINT, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(ToneCurves3PointTest, ShadowHighlightLift) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 50, 50, 50);

    ToneParams params;
    params.shadows = 0.5f;
    params.highlights = -0.3f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::CURVES_3POINT, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

// --- Shadows/Highlights tests ---

TEST(ToneShadowsHighlightsTest, IdentityAtDefaults) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    ToneParams params;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::SHADOWS_HIGHLIGHTS, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 1);
    }
}

TEST(ToneShadowsHighlightsTest, LiftsShadows) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 30, 30, 30);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.shadows = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::SHADOWS_HIGHLIGHTS, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_GT(dst[0], 30) << "Shadows should be lifted";
}

TEST(ToneShadowsHighlightsTest, RecoversHighlights) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 240, 240, 240);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.highlights = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::SHADOWS_HIGHLIGHTS, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_LT(dst[0], 240) << "Highlights should be recovered (darkened)";
}

TEST(ToneShadowsHighlightsTest, DoesNotAffectMidtones) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 128, 128, 128);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.shadows = 1.0f;
    params.highlights = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  ToneAlgorithm::SHADOWS_HIGHLIGHTS, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    EXPECT_NEAR(dst[0], 128, 2) << "Midtones should be unchanged";
}

// --- Parameterized tests across all algorithms ---

TEST_P(ToneAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = 16, h = 16;

    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>((i * 37 + 127) % 256);

    std::vector<uint8_t> dst(w * h * 3);
    ToneParams params;
    params.gamma = 1.5f;
    params.contrast = 0.3f;
    params.shadows = 0.2f;
    params.highlights = -0.1f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  p.algo, 8, params);
    EXPECT_EQ(err, ToneError::Ok);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(ToneAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    ToneParams params;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 1,
                                  p.algo, 8, params);
    EXPECT_EQ(err, ToneError::InvalidChannels);
}

TEST_P(ToneAlgorithmTest, HighBitDepth) {
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
    ToneParams params;
    params.gamma = 1.0f;
    ToneError err = process_tone(src.data(), dst.data(), w, h, 3,
                                  p.algo, 12, params);
    EXPECT_EQ(err, ToneError::Ok);
    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) EXPECT_LE(dst16[i], 4095);
}

INSTANTIATE_TEST_SUITE_P(
    AllToneAlgos,
    ToneAlgorithmTest,
    ::testing::Values(
        ToneTestParam{ToneAlgorithm::GAMMA},
        ToneTestParam{ToneAlgorithm::S_CURVE},
        ToneTestParam{ToneAlgorithm::LEVELS},
        ToneTestParam{ToneAlgorithm::CURVES_3POINT},
        ToneTestParam{ToneAlgorithm::SHADOWS_HIGHLIGHTS}
    )
);
