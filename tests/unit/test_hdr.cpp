#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "hdr/algorithms.hpp"

using namespace hdr;

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

void make_flat_rgb_16(uint8_t* rgb, int width, int height,
                       uint16_t rv, uint16_t gv, uint16_t bv) {
    auto* d16 = reinterpret_cast<uint16_t*>(rgb);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            d16[idx + 0] = rv;
            d16[idx + 1] = gv;
            d16[idx + 2] = bv;
        }
    }
}

struct HdrTestParam {
    HdrAlgorithm algo;
};

class HdrAlgorithmTest : public ::testing::TestWithParam<HdrTestParam> {};

} // anonymous namespace

// ============================================================
//  Validation tests
// ============================================================

TEST(HdrDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    EXPECT_EQ(process_hdr(nullptr, dst, 4, 4, 3, HdrAlgorithm::REINHARD, 8),
              HdrError::NullInput);
    EXPECT_EQ(process_hdr(src, nullptr, 4, 4, 3, HdrAlgorithm::REINHARD, 8),
              HdrError::NullInput);
    EXPECT_EQ(process_hdr(src, dst, 0, 4, 3, HdrAlgorithm::REINHARD, 8),
              HdrError::InvalidDimensions);
    EXPECT_EQ(process_hdr(src, dst, 4, -1, 3, HdrAlgorithm::REINHARD, 8),
              HdrError::InvalidDimensions);
    EXPECT_EQ(process_hdr(src, dst, 4, 4, 1, HdrAlgorithm::REINHARD, 8),
              HdrError::InvalidChannels);
}

TEST(HdrDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(HdrAlgorithm::REINHARD), "Reinhard");
    EXPECT_EQ(algorithm_name(HdrAlgorithm::REINHARD_EXT), "Reinhard Extended");
    EXPECT_EQ(algorithm_name(HdrAlgorithm::FILMIC_ACES), "Filmic ACES");
    EXPECT_EQ(algorithm_name(HdrAlgorithm::HABLE), "Hable (Uncharted 2)");
    EXPECT_EQ(algorithm_name(HdrAlgorithm::DRAGO), "Drago Adaptive Log");
    EXPECT_EQ(algorithm_name(HdrAlgorithm::ADAPTIVE_LOCAL), "Adaptive Local");
    EXPECT_EQ(algorithm_name(HdrAlgorithm::EXPONENTIAL), "Exponential");
    EXPECT_EQ(algorithm_name(HdrAlgorithm::LOGARITHMIC), "Logarithmic");
}

TEST(HdrDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(HdrAlgorithm::REINHARD), 1);
    EXPECT_EQ(algorithm_window_size(HdrAlgorithm::ADAPTIVE_LOCAL), 5);
}

// ============================================================
//  Reinhard tests
// ============================================================

TEST(HdrReinhardTest, OutputInRange8Bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrParams params;
    HdrError err = process_reinhard(src.data(), dst.data(), w, h, 3, 8, params);
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrReinhardTest, CompressesHighValues) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 250, 250, 250);

    std::vector<uint8_t> dst(w * h * 3);
    HdrParams params;
    params.gamma = 1.0f; // linear output
    HdrError err = process_reinhard(src.data(), dst.data(), w, h, 3, 8, params);
    EXPECT_EQ(err, HdrError::Ok);
    // Reinhard compresses L=250/255 ≈ 0.98 to 0.98/1.98 ≈ 0.495
    // so output should be around 126 (0.495 * 255)
    EXPECT_LT(dst[0], 200);
}

TEST(HdrReinhardTest, PreservesZero) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 0, 0, 0);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_reinhard(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (int i = 0; i < 3; i++) EXPECT_EQ(dst[i], 0);
}

TEST(HdrReinhardTest, ExposureAffectsBrightness) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 100, 100);

    std::vector<uint8_t> dst_default(w * h * 3);
    std::vector<uint8_t> dst_bright(w * h * 3);
    std::vector<uint8_t> dst_dark(w * h * 3);

    HdrParams p_default; // exposure=0
    HdrParams p_bright; p_bright.exposure = 2.0f;
    HdrParams p_dark;   p_dark.exposure = -2.0f;

    process_reinhard(src.data(), dst_default.data(), w, h, 3, 8, p_default);
    process_reinhard(src.data(), dst_bright.data(), w, h, 3, 8, p_bright);
    process_reinhard(src.data(), dst_dark.data(), w, h, 3, 8, p_dark);

    EXPECT_GT(dst_bright[0], dst_default[0]) << "Higher exposure should brighten";
    EXPECT_LT(dst_dark[0], dst_default[0]) << "Lower exposure should darken";
}

// ============================================================
//  Reinhard Extended tests
// ============================================================

TEST(HdrReinhardExtTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrParams params;
    HdrError err = process_reinhard_ext(src.data(), dst.data(), w, h, 3, 8, params);
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrReinhardExtTest, KeyAffectsMidtones) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 128, 128, 128);

    std::vector<uint8_t> dst_low_key(w * h * 3);
    std::vector<uint8_t> dst_high_key(w * h * 3);

    HdrParams p_low;  p_low.key = 0.05f; p_low.gamma = 1.0f;
    HdrParams p_high; p_high.key = 0.5f; p_high.gamma = 1.0f;

    process_reinhard_ext(src.data(), dst_low_key.data(), w, h, 3, 8, p_low);
    process_reinhard_ext(src.data(), dst_high_key.data(), w, h, 3, 8, p_high);

    // Lower key = darker image, higher key = brighter image
    EXPECT_LT(dst_low_key[0], dst_high_key[0]);
}

// ============================================================
//  Filmic ACES tests
// ============================================================

TEST(HdrFilmicACESTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_filmic_aces(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrFilmicACESTest, Monotonic) {
    int w = 8, h = 1;
    std::vector<uint8_t> src(w * h * 3);
    for (int x = 0; x < w; x++) {
        uint8_t v = static_cast<uint8_t>(x * 32);
        src[x * 3 + 0] = v;
        src[x * 3 + 1] = v;
        src[x * 3 + 2] = v;
    }

    std::vector<uint8_t> dst(w * h * 3);
    HdrParams params;
    params.gamma = 1.0f;
    HdrError err = process_filmic_aces(src.data(), dst.data(), w, h, 3, 8, params);
    EXPECT_EQ(err, HdrError::Ok);

    // Output should be monotonic: brighter input → brighter output
    for (int x = 1; x < w; x++) {
        EXPECT_GE(dst[x * 3], dst[(x - 1) * 3]);
    }
}

// ============================================================
//  Hable tests
// ============================================================

TEST(HdrHableTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_hable(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrHableTest, StrengthAffectsShoulder) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 220, 220, 220);

    std::vector<uint8_t> dst_weak(w * h * 3);
    std::vector<uint8_t> dst_strong(w * h * 3);

    HdrParams p_w; p_w.strength = 0.5f; p_w.gamma = 1.0f;
    HdrParams p_s; p_s.strength = 2.0f; p_s.gamma = 1.0f;

    process_hable(src.data(), dst_weak.data(), w, h, 3, 8, p_w);
    process_hable(src.data(), dst_strong.data(), w, h, 3, 8, p_s);

    EXPECT_GT(dst_strong[0], dst_weak[0]) << "Stronger strength should compress more in highlights";
}

// ============================================================
//  Drago tests
// ============================================================

TEST(HdrDragoTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_drago(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrDragoTest, BiasAffectsResult) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 128, 128, 128);

    std::vector<uint8_t> dst_low(w * h * 3);
    std::vector<uint8_t> dst_high(w * h * 3);

    HdrParams p_low;  p_low.key = 0.3f; p_low.gamma = 1.0f;
    HdrParams p_high; p_high.key = 0.9f; p_high.gamma = 1.0f;

    process_drago(src.data(), dst_low.data(), w, h, 3, 8, p_low);
    process_drago(src.data(), dst_high.data(), w, h, 3, 8, p_high);

    // Different bias should produce different results
    EXPECT_NE(dst_low[0], dst_high[0]);
}

// ============================================================
//  Exponential tests
// ============================================================

TEST(HdrExponentialTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_exponential(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrExponentialTest, CompressesHighValues) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 250, 250, 250);

    std::vector<uint8_t> dst(w * h * 3);
    HdrParams params;
    params.gamma = 1.0f;
    HdrError err = process_exponential(src.data(), dst.data(), w, h, 3, 8, params);
    EXPECT_EQ(err, HdrError::Ok);
    EXPECT_LT(dst[0], 250) << "Should compress bright values";
}

// ============================================================
//  Logarithmic tests
// ============================================================

TEST(HdrLogarithmicTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_logarithmic(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrLogarithmicTest, Monotonic) {
    int w = 8, h = 1;
    std::vector<uint8_t> src(w * h * 3);
    for (int x = 0; x < w; x++) {
        uint8_t v = static_cast<uint8_t>(x * 32);
        src[x * 3 + 0] = v;
        src[x * 3 + 1] = v;
        src[x * 3 + 2] = v;
    }

    std::vector<uint8_t> dst(w * h * 3);
    HdrParams params;
    params.gamma = 1.0f;
    HdrError err = process_logarithmic(src.data(), dst.data(), w, h, 3, 8, params);
    EXPECT_EQ(err, HdrError::Ok);

    for (int x = 1; x < w; x++) {
        EXPECT_GE(dst[x * 3], dst[(x - 1) * 3]) << "x=" << x;
    }
}

// ============================================================
//  Transfer function tests
// ============================================================

TEST(HdrTransferTest, PQ_RoundTrip) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 128, 128, 128);

    std::vector<uint8_t> pq(w * h * 3);
    std::vector<uint8_t> back(w * h * 3);

    // Linear -> PQ
    HdrError e1 = process_linear_to_pq(src.data(), pq.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(e1, HdrError::Ok);

    // PQ -> Linear
    HdrError e2 = process_pq_to_linear(pq.data(), back.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(e2, HdrError::Ok);

    // Round-trip should approximately preserve values
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(back[i * 3], src[i * 3], 2) << "i=" << i;
    }
}

TEST(HdrTransferTest, HLG_RoundTrip) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    std::vector<uint8_t> hlg(w * h * 3);
    std::vector<uint8_t> back(w * h * 3);

    // Linear -> HLG
    HdrError e1 = process_linear_to_hlg(src.data(), hlg.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(e1, HdrError::Ok);

    // HLG -> Linear
    HdrError e2 = process_hlg_to_linear(hlg.data(), back.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(e2, HdrError::Ok);

    // HLG is designed to be slightly lossy, allow 3 LSB tolerance
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(back[i * 3], src[i * 3], 3) << "i=" << i;
    }
}

TEST(HdrTransferTest, PQ_OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_linear_to_pq(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrTransferTest, HLG_OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_linear_to_hlg(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

// ============================================================
//  Adaptive Local tests
// ============================================================

TEST(HdrAdaptiveLocalTest, OutputInRange) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_adaptive_local(src.data(), dst.data(), w, h, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HdrAdaptiveLocalTest, RejectsSmallImage) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};
    HdrError err = process_adaptive_local(src, dst, 2, 2, 3, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::ImageTooSmall);
}

// ============================================================
//  Parameterized tests: all algorithms
// ============================================================

TEST_P(HdrAlgorithmTest, OutputInRange8Bit) {
    auto p = GetParam();
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>((i * 37 + 127) % 256);

    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_hdr(src.data(), dst.data(), w, h, 3, p.algo, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST_P(HdrAlgorithmTest, OutputInRange16Bit) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = static_cast<uint16_t>((i * 7919 + 3343) % 65536);
        src16[i * 3 + 1] = static_cast<uint16_t>((i * 6271 + 7727) % 65536);
        src16[i * 3 + 2] = static_cast<uint16_t>((i * 4591 + 1447) % 65536);
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    HdrError err = process_hdr(src.data(), dst.data(), w, h, 3, p.algo, 16, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) EXPECT_LE(dst16[i], 65535);
}

TEST_P(HdrAlgorithmTest, PreservesZero) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3, 0);
    std::vector<uint8_t> dst(w * h * 3);
    HdrError err = process_hdr(src.data(), dst.data(), w, h, 3, p.algo, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (int i = 0; i < 3; i++) EXPECT_EQ(dst[i], 0);
}

TEST_P(HdrAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    HdrError err = process_hdr(src.data(), dst.data(), w, h, 1, p.algo, 8, HdrParams{});
    EXPECT_EQ(err, HdrError::InvalidChannels);
}

// ============================================================
//  Float32 (bit_depth=0) tests
// ============================================================

TEST(HdrFloatTest, ValidatesFloat32BitDepth) {
    EXPECT_TRUE(is_valid_bit_depth(0));
    EXPECT_TRUE(is_valid_bit_depth(8));
    EXPECT_TRUE(is_valid_bit_depth(16));
    EXPECT_FALSE(is_valid_bit_depth(-1));
}

TEST(HdrFloatTest, ReinhardCompressesHDR) {
    int w = 16, h = 16;
    // Create float HDR image: values range [0, 10] simulating scene with highlights
    std::vector<float> src_f(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            float v = static_cast<float>(x) / static_cast<float>(w) * 10.0f;
            src_f[idx + 0] = v;
            src_f[idx + 1] = v;
            src_f[idx + 2] = v;
        }
    }

    std::vector<float> dst_f(w * h * 3);
    auto* src_u8 = reinterpret_cast<const uint8_t*>(src_f.data());
    auto* dst_u8 = reinterpret_cast<uint8_t*>(dst_f.data());

    HdrParams params;
    params.gamma = 1.0f; // linear output
    HdrError err = process_reinhard(src_u8, dst_u8, w, h, 3, 0, params);
    EXPECT_EQ(err, HdrError::Ok);

    // Output must be in [0, 1]
    for (size_t i = 0; i < dst_f.size(); i++) {
        EXPECT_GE(dst_f[i], 0.0f) << "i=" << i;
        EXPECT_LE(dst_f[i], 1.0f) << "i=" << i;
    }

    // Darkest value should be near 0
    EXPECT_LT(dst_f[0], 0.01f);
    // Brightest HDR value (L≈10) should be compressed to < 1.0
    size_t last_r = (w - 1) * 3;
    EXPECT_LT(dst_f[last_r], 1.0f);
    EXPECT_GT(dst_f[last_r], 0.5f); // L=10 → L/(1+L)=0.909, with gamma=1 → ~0.91
}

TEST(HdrFloatTest, ExposureBrightensFloatImage) {
    int w = 8, h = 8;
    std::vector<float> src_f(w * h * 3, 0.5f);
    std::vector<float> dst_default(w * h * 3);
    std::vector<float> dst_bright(w * h * 3);

    auto* src_u8 = reinterpret_cast<const uint8_t*>(src_f.data());

    HdrParams p_default;
    HdrParams p_bright; p_bright.exposure = 2.0f;

    process_reinhard(src_u8, reinterpret_cast<uint8_t*>(dst_default.data()), w, h, 3, 0, p_default);
    process_reinhard(src_u8, reinterpret_cast<uint8_t*>(dst_bright.data()), w, h, 3, 0, p_bright);

    EXPECT_GT(dst_bright[0], dst_default[0]) << "Exposure should brighten";
}

TEST(HdrFloatTest, PQRoundTripFloat) {
    int w = 16, h = 16;
    std::vector<float> src_f(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        float v = static_cast<float>(i) / static_cast<float>(w * h); // [0, 1)
        src_f[i * 3 + 0] = v;
        src_f[i * 3 + 1] = v;
        src_f[i * 3 + 2] = v;
    }

    std::vector<float> pq_f(w * h * 3);
    std::vector<float> back_f(w * h * 3);

    auto* src_u8 = reinterpret_cast<const uint8_t*>(src_f.data());
    auto* pq_u8 = reinterpret_cast<uint8_t*>(pq_f.data());
    auto* back_u8 = reinterpret_cast<uint8_t*>(back_f.data());

    HdrError e1 = process_linear_to_pq(src_u8, pq_u8, w, h, 3, 0, HdrParams{});
    EXPECT_EQ(e1, HdrError::Ok);

    HdrError e2 = process_pq_to_linear(pq_u8, back_u8, w, h, 3, 0, HdrParams{});
    EXPECT_EQ(e2, HdrError::Ok);

    // Float round-trip should be very precise
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(back_f[i * 3], src_f[i * 3], 1e-4f) << "i=" << i;
    }
}

TEST_P(HdrAlgorithmTest, Float32OutputInRange) {
    auto p = GetParam();
    int w = 16, h = 16;
    // Create float data with HDR values (>1.0)
    std::vector<float> src_f(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            float v = static_cast<float>(x) / static_cast<float>(w) * 5.0f;
            src_f[idx + 0] = v;
            src_f[idx + 1] = v;
            src_f[idx + 2] = v;
        }
    }

    std::vector<float> dst_f(w * h * 3);
    auto* src_u8 = reinterpret_cast<const uint8_t*>(src_f.data());
    auto* dst_u8 = reinterpret_cast<uint8_t*>(dst_f.data());

    HdrError err = process_hdr(src_u8, dst_u8, w, h, 3, p.algo, 0, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);

    // All output values must be in SDR range [0, 1]
    for (size_t i = 0; i < dst_f.size(); i++) {
        EXPECT_GE(dst_f[i], 0.0f) << "algo=" << static_cast<int>(p.algo) << " i=" << i;
        EXPECT_LE(dst_f[i], 1.0f) << "algo=" << static_cast<int>(p.algo) << " i=" << i;
    }
}

TEST_P(HdrAlgorithmTest, Float32PreservesZero) {
    auto p = GetParam();
    int w = 8, h = 8;
    std::vector<float> src_f(w * h * 3, 0.0f);
    std::vector<float> dst_f(w * h * 3);

    auto* src_u8 = reinterpret_cast<const uint8_t*>(src_f.data());
    auto* dst_u8 = reinterpret_cast<uint8_t*>(dst_f.data());

    HdrError err = process_hdr(src_u8, dst_u8, w, h, 3, p.algo, 0, HdrParams{});
    EXPECT_EQ(err, HdrError::Ok);
    for (int i = 0; i < 3; i++) EXPECT_FLOAT_EQ(dst_f[i], 0.0f);
}

INSTANTIATE_TEST_SUITE_P(
    AllHdrAlgos,
    HdrAlgorithmTest,
    ::testing::Values(
        HdrTestParam{HdrAlgorithm::REINHARD},
        HdrTestParam{HdrAlgorithm::REINHARD_EXT},
        HdrTestParam{HdrAlgorithm::FILMIC_ACES},
        HdrTestParam{HdrAlgorithm::HABLE},
        HdrTestParam{HdrAlgorithm::DRAGO},
        HdrTestParam{HdrAlgorithm::ADAPTIVE_LOCAL},
        HdrTestParam{HdrAlgorithm::EXPONENTIAL},
        HdrTestParam{HdrAlgorithm::LOGARITHMIC},
        HdrTestParam{HdrAlgorithm::LINEAR_TO_PQ},
        HdrTestParam{HdrAlgorithm::PQ_TO_LINEAR},
        HdrTestParam{HdrAlgorithm::LINEAR_TO_HLG},
        HdrTestParam{HdrAlgorithm::HLG_TO_LINEAR}
    )
);
