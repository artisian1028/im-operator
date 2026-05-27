#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "highlight_reconstruct/algorithms.hpp"

using namespace highlight_reconstruct;

void make_rgb_clip(uint8_t* rgb, int width, int height,
                   uint8_t r, uint8_t g, uint8_t b, int bit_depth) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (bit_depth <= 8) {
                size_t idx = (static_cast<size_t>(y) * width + x) * 3;
                rgb[idx + 0] = r; rgb[idx + 1] = g; rgb[idx + 2] = b;
            } else {
                auto* d16 = reinterpret_cast<uint16_t*>(rgb);
                size_t idx = (static_cast<size_t>(y) * width + x) * 3;
                d16[idx + 0] = r; d16[idx + 1] = g; d16[idx + 2] = b;
            }
        }
    }
}

TEST(HighlightReconstructTest, ValidatesInputs) {
    uint8_t src[12] = {0}, dst[12] = {0};
    EXPECT_EQ(process_highlight_reconstruct(nullptr, dst, 2, 2, 3,
              HighlightReconstructAlgorithm::CHANNEL_GUIDED, 8, HighlightReconstructParams{}),
              HighlightReconstructError::NullInput);
    EXPECT_EQ(process_highlight_reconstruct(src, dst, 0, 2, 3,
              HighlightReconstructAlgorithm::CHANNEL_GUIDED, 8, HighlightReconstructParams{}),
              HighlightReconstructError::InvalidDimensions);
    EXPECT_EQ(process_highlight_reconstruct(src, dst, 2, 2, 1,
              HighlightReconstructAlgorithm::CHANNEL_GUIDED, 8, HighlightReconstructParams{}),
              HighlightReconstructError::InvalidChannels);
}

TEST(HighlightReconstructTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(HighlightReconstructAlgorithm::CHANNEL_GUIDED),
              "Channel-Guided Highlight Reconstruct");
    EXPECT_EQ(algorithm_name(HighlightReconstructAlgorithm::GRADIENT_BASED),
              "Gradient-Based Highlight Reconstruct");
}

TEST(HighlightReconstructTest, UnclippedImageUnchanged) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_rgb_clip(src.data(), w, h, 100, 150, 200, 8);
    std::vector<uint8_t> dst(w * h * 3);

    auto err = process_channel_guided(src.data(), dst.data(), w, h, 3, 8, HighlightReconstructParams{});
    EXPECT_EQ(err, HighlightReconstructError::Ok);

    // Output should match input (no clipping)
    for (int i = 0; i < 3; i++) EXPECT_EQ(dst[i], src[i]);
}

TEST(HighlightReconstructTest, ReconstructsPartialClip) {
    int w = 4, h = 4;
    std::vector<uint8_t> src(w * h * 3);
    make_rgb_clip(src.data(), w, h, 255, 200, 150, 8); // R clipped, G,B fine
    std::vector<uint8_t> dst(w * h * 3);

    HighlightReconstructParams p;
    p.threshold = 0.9f;
    auto err = process_channel_guided(src.data(), dst.data(), w, h, 3, 8, p);
    EXPECT_EQ(err, HighlightReconstructError::Ok);

    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(HighlightReconstructTest, GradientBasedOutputInRange) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    make_rgb_clip(src.data(), w, h, 255, 255, 255, 8); // all clipped
    std::vector<uint8_t> dst(w * h * 3);

    HighlightReconstructParams p;
    p.threshold = 0.9f;
    auto err = process_gradient_based(src.data(), dst.data(), w, h, 3, 8, p);
    EXPECT_EQ(err, HighlightReconstructError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}
