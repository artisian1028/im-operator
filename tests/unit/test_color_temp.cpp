#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "color_temp/algorithms.hpp"

using namespace color_temp;

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

struct ColorTempTestParam {
    ColorTempAlgorithm algo;
};

class ColorTempAlgorithmTest : public ::testing::TestWithParam<ColorTempTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(ColorTempDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    EXPECT_EQ(process_color_temp(nullptr, dst, 4, 4, 3,
                                  ColorTempAlgorithm::KELVIN, 8),
              ColorTempError::NullInput);
    EXPECT_EQ(process_color_temp(src, nullptr, 4, 4, 3,
                                  ColorTempAlgorithm::KELVIN, 8),
              ColorTempError::NullInput);
    EXPECT_EQ(process_color_temp(src, dst, 0, 4, 3,
                                  ColorTempAlgorithm::KELVIN, 8),
              ColorTempError::InvalidDimensions);
    EXPECT_EQ(process_color_temp(src, dst, 4, 4, 3,
                                  ColorTempAlgorithm::KELVIN, 17),
              ColorTempError::InvalidBitDepth);
    EXPECT_EQ(process_color_temp(src, dst, 4, 4, 1,
                                  ColorTempAlgorithm::KELVIN, 8),
              ColorTempError::InvalidChannels);
}

// --- Metadata tests ---

TEST(ColorTempDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(ColorTempAlgorithm::KELVIN), "Kelvin Temperature");
    EXPECT_EQ(algorithm_name(ColorTempAlgorithm::PRESET), "Illuminant Preset");
    EXPECT_EQ(algorithm_name(ColorTempAlgorithm::MANUAL), "Manual RGB Multipliers");
    EXPECT_EQ(algorithm_name(ColorTempAlgorithm::WHITE_BALANCE), "Auto White Balance");
}

TEST(ColorTempDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(ColorTempAlgorithm::KELVIN), 1);
    EXPECT_EQ(algorithm_window_size(ColorTempAlgorithm::PRESET), 1);
}

// --- Kelvin conversion tests ---

TEST(ColorTempKelvinTest, KelvinToMultipliersWarm) {
    float r, b;
    kelvin_to_rgb_multipliers(3200, r, b);
    // Warm (3200K) blackbody: red-dominant, blue-suppressed → R > 1, B < 1
    EXPECT_GT(r, 1.0f) << "Warm blackbody: R cast should be > 1";
    EXPECT_LT(b, 1.0f) << "Warm blackbody: B cast should be < 1";
}

TEST(ColorTempKelvinTest, KelvinToMultipliersCool) {
    float r, b;
    kelvin_to_rgb_multipliers(10000, r, b);
    // Cool (10000K) blackbody: blue-dominant, red-suppressed → R < 1, B > 1
    EXPECT_LT(r, 1.0f) << "Cool blackbody: R cast should be < 1";
    EXPECT_GT(b, 1.0f) << "Cool blackbody: B cast should be > 1";
}

TEST(ColorTempKelvinTest, KelvinToMultipliersNeutral) {
    float r, b;
    kelvin_to_rgb_multipliers(6500, r, b);
    // D65: both near 1.0
    EXPECT_NEAR(r, 1.0f, 0.2f);
    EXPECT_NEAR(b, 1.0f, 0.2f);
}

TEST(ColorTempKelvinTest, KelvinClampsOutOfRange) {
    float r_low, b_low, r_high, b_high;
    kelvin_to_rgb_multipliers(500, r_low, b_low);
    kelvin_to_rgb_multipliers(50000, r_high, b_high);
    EXPECT_GE(r_low, 0.0f);
    EXPECT_GE(b_low, 0.0f);
    EXPECT_GE(r_high, 0.0f);
    EXPECT_GE(b_high, 0.0f);
}

// --- Kelvin algorithm tests (apply creative temperature look) ---

TEST(ColorTempKelvinTest, IdentityAtD65) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::KELVIN, 8, 6500);
    EXPECT_EQ(err, ColorTempError::Ok);
    // D65 is near-identity; allow small tolerance from normalization
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_NEAR(dst[i], src[i], 5);
    }
}

TEST(ColorTempKelvinTest, WarmTempWarmCast) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 200, 200);

    std::vector<uint8_t> dst(w * h * 3);
    // 2000K = candlelight: warm look, R boosted, B suppressed
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::KELVIN, 8, 2000);
    EXPECT_EQ(err, ColorTempError::Ok);
    // Warm cast: R should increase, B should decrease
    EXPECT_GE(dst[0], 200);
    EXPECT_LE(dst[2], 200);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(ColorTempKelvinTest, CoolTempCoolCast) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 200, 200);

    std::vector<uint8_t> dst(w * h * 3);
    // 12000K = blue sky: cool look, B boosted, R suppressed
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::KELVIN, 8, 12000);
    EXPECT_EQ(err, ColorTempError::Ok);
    // Cool cast: B should increase, R should decrease
    EXPECT_GE(dst[2], 200);
    EXPECT_LE(dst[0], 200);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

// --- Illuminant preset tests ---

TEST(ColorTempIlluminantTest, AllPresetsValid) {
    IlluminantPreset presets[] = {
        IlluminantPreset::CANDLE,
        IlluminantPreset::TUNGSTEN_40W,
        IlluminantPreset::TUNGSTEN_100W,
        IlluminantPreset::HALOGEN,
        IlluminantPreset::WARM_FLUORESCENT,
        IlluminantPreset::COOL_WHITE_FLUO,
        IlluminantPreset::MIDDAY_SUN,
        IlluminantPreset::CLOUDY,
        IlluminantPreset::SHADE,
        IlluminantPreset::OVERCAST,
        IlluminantPreset::BLUE_SKY
    };

    for (auto p : presets) {
        float r, b;
        illuminant_to_rgb_multipliers(p, r, b);
        EXPECT_GE(r, 0.0f);
        EXPECT_GE(b, 0.0f);

        int k = illuminant_kelvin(p);
        EXPECT_GE(k, 1000);
        EXPECT_LE(k, 40000);

        const char* name = illuminant_name(p);
        EXPECT_NE(name, nullptr);
        EXPECT_GT(strlen(name), 0);
    }
}

TEST(ColorTempIlluminantTest, WarmPresetsCorrectWarmScene) {
    // Tungsten scene: too red, too little blue → correction: suppress R (R<1), boost B (B>1)
    float r, b;
    illuminant_to_rgb_multipliers(IlluminantPreset::TUNGSTEN_100W, r, b);
    EXPECT_LT(r, 1.0f) << "Tungsten correction: should reduce red (R < 1)";
    EXPECT_GT(b, 1.0f) << "Tungsten correction: should boost blue (B > 1)";
}

TEST(ColorTempIlluminantTest, CoolPresetsCorrectCoolScene) {
    float r, b;
    illuminant_to_rgb_multipliers(IlluminantPreset::SHADE, r, b);
    EXPECT_GT(r, 1.0f) << "Shade correction: should boost red (R > 1)";
    EXPECT_LT(b, 1.0f) << "Shade correction: should reduce blue (B < 1)";
}

TEST(ColorTempIlluminantTest, D65IsNeutral) {
    float r, b;
    illuminant_to_rgb_multipliers(IlluminantPreset::CLOUDY, r, b);
    EXPECT_NEAR(r, 1.0f, 0.1f);
    EXPECT_NEAR(b, 1.0f, 0.1f);
}

// --- Preset algorithm tests ---

TEST(ColorTempPresetTest, TungstenCorrection) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    // Simulate tungsten-lit scene: reddish, low blue
    make_flat_rgb(src.data(), w, h, 200, 120, 60);

    std::vector<uint8_t> dst(w * h * 3);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::PRESET, 8, 6500,
                                              IlluminantPreset::TUNGSTEN_100W);
    EXPECT_EQ(err, ColorTempError::Ok);

    // After correction, R/B ratio should be closer to 1
    double sum_r = 0, sum_b = 0;
    for (int i = 0; i < w * h; i++) {
        sum_r += dst[i * 3 + 0];
        sum_b += dst[i * 3 + 2];
    }
    double corrected_ratio = sum_r / sum_b;
    double orig_ratio = 200.0 / 60.0;
    EXPECT_LT(std::abs(corrected_ratio - 1.0), std::abs(orig_ratio - 1.0));
}

TEST(ColorTempPresetTest, CandlelightCorrection) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 100, 30);

    std::vector<uint8_t> dst(w * h * 3);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::PRESET, 8, 6500,
                                              IlluminantPreset::CANDLE);
    EXPECT_EQ(err, ColorTempError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(ColorTempPresetTest, IdentityAtD65) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::PRESET, 8, 6500,
                                              IlluminantPreset::CLOUDY);
    EXPECT_EQ(err, ColorTempError::Ok);
    // D65 preset is 1.0/1.0 — strict identity
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

// --- Manual tests ---

TEST(ColorTempManualTest, IdentityGains) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::MANUAL, 8, 6500,
                                              IlluminantPreset::CLOUDY, 1.0f, 1.0f);
    EXPECT_EQ(err, ColorTempError::Ok);
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(ColorTempManualTest, CustomGains) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 100, 100);

    std::vector<uint8_t> dst(w * h * 3);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::MANUAL, 8, 6500,
                                              IlluminantPreset::CLOUDY, 0.5f, 2.0f);
    EXPECT_EQ(err, ColorTempError::Ok);
    EXPECT_EQ(dst[0], 50);  // R halved
    EXPECT_EQ(dst[1], 100); // G unchanged
    EXPECT_EQ(dst[2], 200); // B doubled
}

TEST(ColorTempManualTest, ClampsOutput) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 200, 200, 200);

    std::vector<uint8_t> dst(w * h * 3);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::MANUAL, 8, 6500,
                                              IlluminantPreset::CLOUDY, 3.0f, 3.0f);
    EXPECT_EQ(err, ColorTempError::Ok);
    // Should clamp at 255
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

// --- Auto White Balance tests ---

TEST(ColorTempAutoWBTest, NeutralizesWarmCast) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    // Warm cast: R high, B low
    make_flat_rgb(src.data(), w, h, 200, 150, 80);

    std::vector<uint8_t> dst(w * h * 3);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::WHITE_BALANCE, 8);
    EXPECT_EQ(err, ColorTempError::Ok);

    // After auto WB, R/B ratio should be closer to 1
    double sum_r = 0, sum_b = 0;
    for (int i = 0; i < w * h; i++) {
        sum_r += dst[i * 3 + 0];
        sum_b += dst[i * 3 + 2];
    }
    double wb_ratio = sum_r / sum_b;
    double orig_ratio = 200.0 / 80.0;
    EXPECT_LT(std::abs(wb_ratio - 1.0), std::abs(orig_ratio - 1.0));
}

TEST(ColorTempAutoWBTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::WHITE_BALANCE, 8);
    EXPECT_EQ(err, ColorTempError::Ok);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

// --- High bit depth tests ---

TEST(ColorTempTest, HighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              ColorTempAlgorithm::KELVIN, 12, 6500);
    EXPECT_EQ(err, ColorTempError::Ok);
    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) EXPECT_LE(dst16[i], 4095);
}

// --- Parameterized tests across all algorithms ---

TEST_P(ColorTempAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = 16, h = 16;

    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>((i * 37 + 127) % 256);

    std::vector<uint8_t> dst(w * h * 3);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              p.algo, 8, 6500,
                                              IlluminantPreset::CLOUDY, 1.0f, 1.0f);
    EXPECT_EQ(err, ColorTempError::Ok);
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(ColorTempAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 1,
                                              p.algo, 8, 6500,
                                              IlluminantPreset::CLOUDY, 1.0f, 1.0f);
    EXPECT_EQ(err, ColorTempError::InvalidChannels);
}

TEST_P(ColorTempAlgorithmTest, HighBitDepth) {
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
    ColorTempError err = process_color_temp(src.data(), dst.data(), w, h, 3,
                                              p.algo, 12, 6500,
                                              IlluminantPreset::CLOUDY, 1.0f, 1.0f);
    EXPECT_EQ(err, ColorTempError::Ok);
    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) EXPECT_LE(dst16[i], 4095);
}

INSTANTIATE_TEST_SUITE_P(
    AllColorTempAlgos,
    ColorTempAlgorithmTest,
    ::testing::Values(
        ColorTempTestParam{ColorTempAlgorithm::KELVIN},
        ColorTempTestParam{ColorTempAlgorithm::PRESET},
        ColorTempTestParam{ColorTempAlgorithm::MANUAL},
        ColorTempTestParam{ColorTempAlgorithm::WHITE_BALANCE}
    )
);
