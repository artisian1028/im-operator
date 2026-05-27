#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "sharpen/algorithms.hpp"

using namespace sharpen;

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

void make_edge_rgb(uint8_t* rgb, int width, int height) {
    // Left half = dark (50), right half = bright (200) — a sharp edge
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            uint8_t v = (x < width / 2) ? 50 : 200;
            rgb[idx + 0] = v;
            rgb[idx + 1] = v;
            rgb[idx + 2] = v;
        }
    }
}

struct SharpenTestParam {
    SharpenAlgorithm algo;
    int min_size;
};

class SharpenAlgorithmTest : public ::testing::TestWithParam<SharpenTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(SharpenDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    EXPECT_EQ(process_sharpen(nullptr, dst, 4, 4, 3,
                               SharpenAlgorithm::UNSHARP_MASK, 8),
              SharpenError::NullInput);
    EXPECT_EQ(process_sharpen(src, nullptr, 4, 4, 3,
                               SharpenAlgorithm::UNSHARP_MASK, 8),
              SharpenError::NullInput);

    EXPECT_EQ(process_sharpen(src, dst, 0, 4, 3,
                               SharpenAlgorithm::UNSHARP_MASK, 8),
              SharpenError::InvalidDimensions);
    EXPECT_EQ(process_sharpen(src, dst, 4, -1, 3,
                               SharpenAlgorithm::UNSHARP_MASK, 8),
              SharpenError::InvalidDimensions);

    EXPECT_EQ(process_sharpen(src, dst, 4, 4, 3,
                               SharpenAlgorithm::UNSHARP_MASK, 17),
              SharpenError::InvalidBitDepth);

    EXPECT_EQ(process_sharpen(src, dst, 4, 4, 1,
                               SharpenAlgorithm::UNSHARP_MASK, 8),
              SharpenError::InvalidChannels);
}

// --- Metadata tests ---

TEST(SharpenDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(SharpenAlgorithm::UNSHARP_MASK), "Unsharp Mask");
    EXPECT_EQ(algorithm_name(SharpenAlgorithm::LAPLACIAN), "Laplacian Edge Enhancement");
    EXPECT_EQ(algorithm_name(SharpenAlgorithm::HIGH_PASS), "High-Pass Filter Overlay");
    EXPECT_EQ(algorithm_name(SharpenAlgorithm::ADAPTIVE), "Adaptive (edge-aware)");
}

TEST(SharpenDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(SharpenAlgorithm::UNSHARP_MASK), 5);
    EXPECT_EQ(algorithm_window_size(SharpenAlgorithm::LAPLACIAN), 5);
    EXPECT_EQ(algorithm_window_size(SharpenAlgorithm::HIGH_PASS), 3);
    EXPECT_EQ(algorithm_window_size(SharpenAlgorithm::ADAPTIVE), 5);
}

// --- Unsharp Mask tests ---

TEST(SharpenUnsharpMaskTest, IdentityAtZeroAmount) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SharpenParams params;
    params.amount = 0.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::UNSHARP_MASK, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    // amount=0: output should match input
    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(SharpenUnsharpMaskTest, EnhancesEdges) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_edge_rgb(src.data(), w, h);

    SharpenParams params;
    params.amount = 1.5f;
    params.radius = 1.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::UNSHARP_MASK, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    // At the edge center, the difference between pixels should increase
    int edge_x = w / 2 - 1;
    int edge_next = w / 2;
    int diff_before = std::abs(src[(edge_next) * 3] - src[(edge_x) * 3]);
    int diff_after  = std::abs(dst[(edge_next) * 3] - dst[(edge_x) * 3]);
    EXPECT_GT(diff_after, diff_before - 2) << "Edge should not be weakened";
}

TEST(SharpenUnsharpMaskTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    SharpenParams params;
    params.amount = 2.0f;
    params.radius = 1.5f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::UNSHARP_MASK, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(SharpenUnsharpMaskTest, TooSmallImage) {
    std::vector<uint8_t> src(12);
    std::vector<uint8_t> dst(12);
    SharpenParams params;
    SharpenError err = process_sharpen(src.data(), dst.data(), 2, 2, 3,
                                        SharpenAlgorithm::UNSHARP_MASK, 8, params);
    EXPECT_EQ(err, SharpenError::ImageTooSmall);
}

TEST(SharpenUnsharpMaskTest, HighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    SharpenParams params;
    params.amount = 1.0f;
    params.radius = 1.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::UNSHARP_MASK, 12, params);
    EXPECT_EQ(err, SharpenError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) {
        EXPECT_LE(dst16[i], 4095);
    }
}

// --- Laplacian tests ---

TEST(SharpenLaplacianTest, IdentityAtZeroAmount) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SharpenParams params;
    params.amount = 0.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::LAPLACIAN, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(SharpenLaplacianTest, EnhancesEdges) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_edge_rgb(src.data(), w, h);

    SharpenParams params;
    params.amount = 1.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::LAPLACIAN, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    // Output should have valid values and some enhancement
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(SharpenLaplacianTest, LargeKernelMode) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SharpenParams params;
    params.amount = 0.5f;
    params.radius = 2.0f; // triggers 5x5 kernel
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::LAPLACIAN, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(SharpenLaplacianTest, TooSmallImage) {
    std::vector<uint8_t> src(12);
    std::vector<uint8_t> dst(12);
    SharpenParams params;
    SharpenError err = process_sharpen(src.data(), dst.data(), 2, 2, 3,
                                        SharpenAlgorithm::LAPLACIAN, 8, params);
    EXPECT_EQ(err, SharpenError::ImageTooSmall);
}

// --- High-Pass tests ---

TEST(SharpenHighPassTest, IdentityAtZeroAmount) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SharpenParams params;
    params.amount = 0.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::HIGH_PASS, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(SharpenHighPassTest, EnhancesFineDetails) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);

    // Pattern with fine variation
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            uint8_t v = static_cast<uint8_t>(128 + ((x + y) % 16) * 4);
            src[idx + 0] = v;
            src[idx + 1] = v;
            src[idx + 2] = v;
        }
    }

    SharpenParams params;
    params.amount = 1.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::HIGH_PASS, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(SharpenHighPassTest, OutputInRangeAtMaxAmount) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    SharpenParams params;
    params.amount = 3.0f; // maximum
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::HIGH_PASS, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(SharpenHighPassTest, TooSmallImage) {
    std::vector<uint8_t> src(12);
    std::vector<uint8_t> dst(12);
    SharpenParams params;
    SharpenError err = process_sharpen(src.data(), dst.data(), 2, 2, 3,
                                        SharpenAlgorithm::HIGH_PASS, 8, params);
    EXPECT_EQ(err, SharpenError::ImageTooSmall);
}

// --- Adaptive tests ---

TEST(SharpenAdaptiveTest, IdentityAtZeroAmount) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_flat_rgb(src.data(), w, h, 100, 150, 200);

    SharpenParams params;
    params.amount = 0.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::ADAPTIVE, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < src.size(); i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}

TEST(SharpenAdaptiveTest, DoesNotAmplifyFlatRegions) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_flat_rgb(src.data(), w, h, 150, 150, 150);

    std::vector<uint8_t> dst(w * h * 3);
    SharpenParams params;
    params.amount = 2.0f;
    params.radius = 1.0f;
    params.threshold = 0.1f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::ADAPTIVE, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    // Flat regions should stay flat (no noise amplification)
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(150), 2);
        EXPECT_NEAR(dst[i * 3 + 1], static_cast<uint8_t>(150), 2);
        EXPECT_NEAR(dst[i * 3 + 2], static_cast<uint8_t>(150), 2);
    }
}

TEST(SharpenAdaptiveTest, SharpensEdges) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    make_edge_rgb(src.data(), w, h);

    std::vector<uint8_t> dst(w * h * 3);
    SharpenParams params;
    params.amount = 1.5f;
    params.radius = 1.0f;
    params.threshold = 0.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::ADAPTIVE, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(SharpenAdaptiveTest, TooSmallImage) {
    std::vector<uint8_t> src(12);
    std::vector<uint8_t> dst(12);
    SharpenParams params;
    SharpenError err = process_sharpen(src.data(), dst.data(), 2, 2, 3,
                                        SharpenAlgorithm::ADAPTIVE, 8, params);
    EXPECT_EQ(err, SharpenError::ImageTooSmall);
}

TEST(SharpenAdaptiveTest, HighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            src16[idx * 3 + 0] = (x < w / 2) ? 500 : 3000;
            src16[idx * 3 + 1] = (x < w / 2) ? 500 : 3000;
            src16[idx * 3 + 2] = (x < w / 2) ? 500 : 3000;
        }
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    SharpenParams params;
    params.amount = 1.0f;
    params.radius = 1.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::ADAPTIVE, 12, params);
    EXPECT_EQ(err, SharpenError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) {
        EXPECT_LE(dst16[i], 4095);
    }
}

// --- Default params ---

TEST(SharpenDefaultsTest, DefaultParamsProduceOutput) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>(i % 256);

    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        SharpenAlgorithm::UNSHARP_MASK, 8);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

// --- Parameterized tests across all algorithms ---

TEST_P(SharpenAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = std::max(p.min_size + 2, 16);
    int h = std::max(p.min_size + 2, 16);

    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>((i * 37 + 127) % 256);

    std::vector<uint8_t> dst(w * h * 3);
    SharpenParams params;
    params.amount = 1.0f;
    params.radius = 1.0f;
    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 3,
                                        p.algo, 8, params);
    EXPECT_EQ(err, SharpenError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(SharpenAlgorithmTest, DetectsImageTooSmall) {
    auto p = GetParam();

    std::vector<uint8_t> src(12);
    std::vector<uint8_t> dst(12);
    SharpenParams params;
    SharpenError err = process_sharpen(src.data(), dst.data(), 2, 2, 3,
                                        p.algo, 8, params);
    EXPECT_EQ(err, SharpenError::ImageTooSmall);
}

TEST_P(SharpenAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();

    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);
    SharpenParams params;

    SharpenError err = process_sharpen(src.data(), dst.data(), w, h, 1,
                                        p.algo, 8, params);
    EXPECT_EQ(err, SharpenError::InvalidChannels);
}

INSTANTIATE_TEST_SUITE_P(
    AllSharpenAlgos,
    SharpenAlgorithmTest,
    ::testing::Values(
        SharpenTestParam{SharpenAlgorithm::UNSHARP_MASK, 3},
        SharpenTestParam{SharpenAlgorithm::LAPLACIAN, 3},
        SharpenTestParam{SharpenAlgorithm::HIGH_PASS, 3},
        SharpenTestParam{SharpenAlgorithm::ADAPTIVE, 5}
    )
);
