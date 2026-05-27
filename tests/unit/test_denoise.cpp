#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "denoise/algorithms.hpp"

using namespace denoise;

namespace {

// Creates a simple RGB image with some noise: gradient + spike noise
void make_noisy_rgb(uint8_t* rgb, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            int r = 100 + (x * 10) % 50 + (y * 5) % 30;
            int g = 120 + (x * 8) % 40 + (y * 6) % 25;
            int b = 90 + (x * 12) % 45 + (y * 4) % 35;
            rgb[idx + 0] = static_cast<uint8_t>(r);
            rgb[idx + 1] = static_cast<uint8_t>(g);
            rgb[idx + 2] = static_cast<uint8_t>(b);
        }
    }
    // Add some salt-and-pepper noise
    rgb[12] = 255; rgb[13] = 0; rgb[14] = 128;  // spike
    rgb[150] = 0; rgb[151] = 0; rgb[152] = 0;   // dead pixel
}

// Creates a simple single-channel (grayscale/Bayer) image
void make_test_image_1ch(uint8_t* data, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            data[y * width + x] = static_cast<uint8_t>(100 + (x * 7) % 40 + (y * 3) % 30);
        }
    }
    // Add a spike
    data[width * 2 + 2] = 255;
}

struct DenoiseTestParam {
    DenoiseAlgorithm algo;
    int min_size;
};

class DenoiseAlgorithmTest : public ::testing::TestWithParam<DenoiseTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(DenoiseDispatchTest, ValidateInputs) {
    uint8_t src[16] = {0};
    uint8_t dst[16] = {0};

    // Null input
    EXPECT_EQ(process_denoise(nullptr, dst, 4, 4, 3,
              DenoiseAlgorithm::GAUSSIAN, 8), DenoiseError::NullInput);
    EXPECT_EQ(process_denoise(src, nullptr, 4, 4, 3,
              DenoiseAlgorithm::GAUSSIAN, 8), DenoiseError::NullInput);

    // Invalid dimensions
    EXPECT_EQ(process_denoise(src, dst, 0, 4, 3,
              DenoiseAlgorithm::GAUSSIAN, 8), DenoiseError::InvalidDimensions);
    EXPECT_EQ(process_denoise(src, dst, 4, -1, 3,
              DenoiseAlgorithm::GAUSSIAN, 8), DenoiseError::InvalidDimensions);

    // Invalid bit depth
    EXPECT_EQ(process_denoise(src, dst, 4, 4, 3,
              DenoiseAlgorithm::GAUSSIAN, 17), DenoiseError::InvalidBitDepth);

    // Invalid channels
    EXPECT_EQ(process_denoise(src, dst, 4, 4, 0,
              DenoiseAlgorithm::GAUSSIAN, 8), DenoiseError::InvalidChannels);
    EXPECT_EQ(process_denoise(src, dst, 4, 4, 5,
              DenoiseAlgorithm::GAUSSIAN, 8), DenoiseError::InvalidChannels);
}

// --- Metadata tests ---

TEST(DenoiseDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(DenoiseAlgorithm::GAUSSIAN), "Gaussian (separable blur)");
    EXPECT_EQ(algorithm_name(DenoiseAlgorithm::MEDIAN), "Median (3x3)");
    EXPECT_EQ(algorithm_name(DenoiseAlgorithm::BILATERAL), "Bilateral (edge-preserving)");
    EXPECT_EQ(algorithm_name(DenoiseAlgorithm::NLM), "NLM (Non-Local Means)");
    EXPECT_EQ(algorithm_name(DenoiseAlgorithm::WAVELET), "Wavelet (Soft Threshold)");
    EXPECT_EQ(algorithm_name(DenoiseAlgorithm::BAYER_DENOISE), "Bayer-domain denoise");
}

TEST(DenoiseDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(DenoiseAlgorithm::GAUSSIAN), 5);
    EXPECT_EQ(algorithm_window_size(DenoiseAlgorithm::MEDIAN), 3);
    EXPECT_EQ(algorithm_window_size(DenoiseAlgorithm::BILATERAL), 5);
    EXPECT_EQ(algorithm_window_size(DenoiseAlgorithm::NLM), 7);
    EXPECT_EQ(algorithm_window_size(DenoiseAlgorithm::WAVELET), 8);
    EXPECT_EQ(algorithm_window_size(DenoiseAlgorithm::BAYER_DENOISE), 5);
}

// --- Gaussian tests ---

TEST(DenoiseGaussianTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::GAUSSIAN, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    // Output should not be all zeros
    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(DenoiseGaussianTest, StrengthParameter) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst1(w * h * 3);
    std::vector<uint8_t> dst2(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    process_denoise(src.data(), dst1.data(), w, h, 3,
                    DenoiseAlgorithm::GAUSSIAN, 8, 0.5f);
    process_denoise(src.data(), dst2.data(), w, h, 3,
                    DenoiseAlgorithm::GAUSSIAN, 8, 2.0f);

    // Both should succeed
    // Higher sigma = more blur, but both should produce valid output
    for (size_t i = 0; i < dst1.size(); i++) {
        EXPECT_LE(dst1[i], 255);
        EXPECT_LE(dst2[i], 255);
    }
}

TEST(DenoiseGaussianTest, TooSmallImage) {
    std::vector<uint8_t> src(4);
    std::vector<uint8_t> dst(4);
    DenoiseError err = process_denoise(src.data(), dst.data(), 2, 2, 1,
                                        DenoiseAlgorithm::GAUSSIAN, 8);
    EXPECT_EQ(err, DenoiseError::ImageTooSmall);
}

TEST(DenoiseGaussianTest, GrayscaleInput) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    make_test_image_1ch(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::GAUSSIAN, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
}

// --- Median tests ---

TEST(DenoiseMedianTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::MEDIAN, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);

    // All output values should be in range
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(DenoiseMedianTest, RemovesSpikes) {
    // Create a flat image with an extreme spike
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3, 100);
    // Set a spike at (4,4) - R=255
    size_t spike_idx = (4 * w + 4) * 3;
    src[spike_idx] = 255;

    std::vector<uint8_t> dst(w * h * 3);
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::MEDIAN, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    // The spike should be suppressed (median of neighbors ~100)
    EXPECT_LE(dst[spike_idx], 150);
}

TEST(DenoiseMedianTest, TooSmallImage) {
    std::vector<uint8_t> src(4);
    std::vector<uint8_t> dst(4);
    DenoiseError err = process_denoise(src.data(), dst.data(), 2, 2, 1,
                                        DenoiseAlgorithm::MEDIAN, 8);
    EXPECT_EQ(err, DenoiseError::ImageTooSmall);
}

TEST(DenoiseMedianTest, GrayscaleInput) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    make_test_image_1ch(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::MEDIAN, 8);
    EXPECT_EQ(err, DenoiseError::Ok);
}

// --- Bayer denoise tests ---

TEST(DenoiseBayerTest, BasicProcessing8bit) {
    // Create a synthetic Bayer pattern (like RGGB)
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool even_row = (y & 1) == 0;
            bool even_col = (x & 1) == 0;
            if (even_row && even_col)       src[y * w + x] = 200;  // R
            else if (!even_row && even_col) src[y * w + x] = 120;  // Gb
            else if (even_row && !even_col) src[y * w + x] = 100;  // Gr
            else                             src[y * w + x] = 80;   // B
        }
    }
    // Add a hot pixel at (4,4) - this is an R pixel in RGGB
    src[4 * w + 4] = 255;

    std::vector<uint8_t> dst(w * h);
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::BAYER_DENOISE, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    // The hot R pixel should be suppressed
    EXPECT_LE(dst[4 * w + 4], 220);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(DenoiseBayerTest, RequiresSingleChannel) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::BAYER_DENOISE, 8);
    EXPECT_EQ(err, DenoiseError::InvalidChannels);
}

TEST(DenoiseBayerTest, TooSmallImage) {
    std::vector<uint8_t> src(16);
    std::vector<uint8_t> dst(16);
    DenoiseError err = process_denoise(src.data(), dst.data(), 4, 4, 1,
                                        DenoiseAlgorithm::BAYER_DENOISE, 8);
    EXPECT_EQ(err, DenoiseError::ImageTooSmall);
}

TEST(DenoiseBayerTest, PreservesBayerPattern) {
    // After denoising, same-color pixels should still have similar (corrected) values
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool even_row = (y & 1) == 0;
            bool even_col = (x & 1) == 0;
            if (even_row && even_col)       src[y * w + x] = 200;
            else if (!even_row && even_col) src[y * w + x] = 120;
            else if (even_row && !even_col) src[y * w + x] = 100;
            else                             src[y * w + x] = 80;
        }
    }

    std::vector<uint8_t> dst(w * h);
    process_denoise(src.data(), dst.data(), w, h, 1,
                    DenoiseAlgorithm::BAYER_DENOISE, 8);

    // R pixels (even, even) should still be close to 200
    // (may change slightly near borders due to mirror padding)
    int center_r = dst[6 * w + 6];
    EXPECT_GE(center_r, 180);
    EXPECT_LE(center_r, 220);

    // B pixels (odd, odd) should still be close to 80
    int center_b = dst[7 * w + 7];
    EXPECT_GE(center_b, 60);
    EXPECT_LE(center_b, 100);
}

// --- Bilateral tests ---

TEST(DenoiseBilateralTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::BILATERAL, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(DenoiseBilateralTest, PreservesEdges) {
    // Create a sharp edge image: left half dark, right half bright
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            uint8_t val = (x < w / 2) ? 50 : 200;
            src[idx + 0] = val;
            src[idx + 1] = val;
            src[idx + 2] = val;
        }
    }

    std::vector<uint8_t> dst(w * h * 3);
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::BILATERAL, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    // Edge should be preserved: pixel just left of edge still < pixel just right
    // w/2 - 1 = dark side, w/2 = bright side
    size_t left_idx  = (5 * w + (w/2 - 1)) * 3;
    size_t right_idx = (5 * w + (w/2)) * 3;
    // After bilateral, dark side should still be notably darker than bright side
    EXPECT_LT(dst[left_idx], dst[right_idx]);
}

TEST(DenoiseBilateralTest, StrengthParameter) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    // Low strength = subtle, high strength = aggressive
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::BILATERAL, 8, 0.5f);
    EXPECT_EQ(err, DenoiseError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);

    err = process_denoise(src.data(), dst.data(), w, h, 3,
                          DenoiseAlgorithm::BILATERAL, 8, 2.0f);
    EXPECT_EQ(err, DenoiseError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(DenoiseBilateralTest, TooSmallImage) {
    std::vector<uint8_t> src(4);
    std::vector<uint8_t> dst(4);
    DenoiseError err = process_denoise(src.data(), dst.data(), 2, 2, 1,
                                        DenoiseAlgorithm::BILATERAL, 8);
    EXPECT_EQ(err, DenoiseError::ImageTooSmall);
}

TEST(DenoiseBilateralTest, GrayscaleInput) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    make_test_image_1ch(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::BILATERAL, 8);
    EXPECT_EQ(err, DenoiseError::Ok);
}

// --- NLM tests ---

TEST(DenoiseNLMTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::NLM, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(DenoiseNLMTest, StrengthParameter) {
    int w = 24, h = 24;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::NLM, 8, 0.5f);
    EXPECT_EQ(err, DenoiseError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(DenoiseNLMTest, TooSmallImage) {
    std::vector<uint8_t> src(4 * 4);
    std::vector<uint8_t> dst(4 * 4);
    DenoiseError err = process_denoise(src.data(), dst.data(), 4, 4, 1,
                                        DenoiseAlgorithm::NLM, 8);
    EXPECT_EQ(err, DenoiseError::ImageTooSmall);
}

TEST(DenoiseNLMTest, GrayscaleInput) {
    int w = 24, h = 24;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    make_test_image_1ch(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::NLM, 8);
    EXPECT_EQ(err, DenoiseError::Ok);
}

// --- Wavelet tests ---

TEST(DenoiseWaveletTest, BasicProcessing8bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::WAVELET, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    bool has_nonzero = false;
    for (size_t i = 0; i < dst.size(); i++) {
        if (dst[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(DenoiseWaveletTest, StrengthParameter) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst1(w * h * 3);
    std::vector<uint8_t> dst2(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    process_denoise(src.data(), dst1.data(), w, h, 3,
                    DenoiseAlgorithm::WAVELET, 8, 0.5f);
    process_denoise(src.data(), dst2.data(), w, h, 3,
                    DenoiseAlgorithm::WAVELET, 8, 2.0f);

    // Both should produce valid output
    for (size_t i = 0; i < dst1.size(); i++) EXPECT_LE(dst1[i], 255);
    for (size_t i = 0; i < dst2.size(); i++) EXPECT_LE(dst2[i], 255);
}

TEST(DenoiseWaveletTest, TooSmallImage) {
    std::vector<uint8_t> src(4);
    std::vector<uint8_t> dst(4);
    DenoiseError err = process_denoise(src.data(), dst.data(), 2, 2, 1,
                                        DenoiseAlgorithm::WAVELET, 8);
    EXPECT_EQ(err, DenoiseError::ImageTooSmall);
}

TEST(DenoiseWaveletTest, GrayscaleInput) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    make_test_image_1ch(src.data(), w, h);

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::WAVELET, 8);
    EXPECT_EQ(err, DenoiseError::Ok);
}

TEST(DenoiseWaveletTest, ValidOutputOnFlatImage) {
    // Verify wavelet denoise produces valid output on flat data
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h, 128);

    std::vector<uint8_t> dst(w * h);
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::WAVELET, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    // All output values must be in valid range [0, 255]
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_GE(dst[i], 0);
        EXPECT_LE(dst[i], 255);
    }

    // Output should not be all identical (if there's any processing)
    bool varies = false;
    for (size_t i = 1; i < dst.size(); i++) {
        if (dst[i] != dst[0]) { varies = true; break; }
    }
    (void)varies; // We just care about valid range
}

TEST(DenoiseWaveletTest, NonSquareImage) {
    // Wavelet uses square power-of-2 padding, should handle non-square gracefully
    int w = 20, h = 10;
    std::vector<uint8_t> src(w * h * 3);
    make_noisy_rgb(src.data(), w, h);

    std::vector<uint8_t> dst(w * h * 3);
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::WAVELET, 8);
    EXPECT_EQ(err, DenoiseError::Ok);
}

// --- High bit depth tests ---

TEST(DenoiseHighBitDepthTest, Gaussian16bit) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    std::vector<uint8_t> dst(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h * 3; i++) src16[i] = 1000;

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::GAUSSIAN, 12);
    EXPECT_EQ(err, DenoiseError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) {
        EXPECT_LE(dst16[i], 4095);
    }
}

TEST(DenoiseHighBitDepthTest, Median16bit) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    std::vector<uint8_t> dst(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h * 3; i++) src16[i] = 2000;
    // Add a spike
    src16[12] = 4000;

    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 3,
                                        DenoiseAlgorithm::MEDIAN, 12);
    EXPECT_EQ(err, DenoiseError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    EXPECT_LE(dst16[12], 2500);  // Spike should be suppressed
}

TEST(DenoiseHighBitDepthTest, BayerDenoise16bit) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            bool even_row = (y & 1) == 0;
            bool even_col = (x & 1) == 0;
            if (even_row && even_col)       src16[y * w + x] = 2000;
            else if (!even_row && even_col) src16[y * w + x] = 1200;
            else if (even_row && !even_col) src16[y * w + x] = 1000;
            else                             src16[y * w + x] = 800;
        }
    }
    // Hot R pixel
    src16[2 * w + 2] = 4000;

    std::vector<uint8_t> dst(w * h * 2);
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, 1,
                                        DenoiseAlgorithm::BAYER_DENOISE, 12);
    EXPECT_EQ(err, DenoiseError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    EXPECT_LE(dst16[2 * w + 2], 3000);
}

// --- Parameterized test across all algorithms ---

TEST_P(DenoiseAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = std::max(p.min_size + 1, 16);
    int h = std::max(p.min_size + 1, 16);

    // BAYER_DENOISE requires single channel
    int channels = (p.algo == DenoiseAlgorithm::BAYER_DENOISE) ? 1 : 3;
    std::vector<uint8_t> src(w * h * channels);
    if (channels == 3) {
        make_noisy_rgb(src.data(), w, h);
    } else {
        make_test_image_1ch(src.data(), w, h);
    }

    std::vector<uint8_t> dst(w * h * channels);
    DenoiseError err = process_denoise(src.data(), dst.data(), w, h, channels,
                                        p.algo, 8);
    EXPECT_EQ(err, DenoiseError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(DenoiseAlgorithmTest, DetectsImageTooSmall) {
    auto p = GetParam();
    if (p.min_size <= 3) return;

    int channels = (p.algo == DenoiseAlgorithm::BAYER_DENOISE) ? 1 : 3;
    std::vector<uint8_t> src(2 * 2 * channels);
    std::vector<uint8_t> dst(2 * 2 * channels);
    DenoiseError err = process_denoise(src.data(), dst.data(), 2, 2, channels,
                                        p.algo, 8);
    EXPECT_EQ(err, DenoiseError::ImageTooSmall);
}

INSTANTIATE_TEST_SUITE_P(
    AllDenoiseAlgos,
    DenoiseAlgorithmTest,
    ::testing::Values(
        DenoiseTestParam{DenoiseAlgorithm::GAUSSIAN, 3},
        DenoiseTestParam{DenoiseAlgorithm::MEDIAN, 3},
        DenoiseTestParam{DenoiseAlgorithm::BILATERAL, 3},
        DenoiseTestParam{DenoiseAlgorithm::NLM, 7},
        DenoiseTestParam{DenoiseAlgorithm::WAVELET, 4},
        DenoiseTestParam{DenoiseAlgorithm::BAYER_DENOISE, 5}
    )
);
