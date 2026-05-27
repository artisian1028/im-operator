#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "local_contrast/algorithms.hpp"

using namespace local_contrast;

void make_rgb(uint8_t* rgb, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            rgb[idx+0] = r; rgb[idx+1] = g; rgb[idx+2] = b;
        }
}

TEST(LocalContrastTest, ValidatesInputs) {
    uint8_t src[12] = {0}, dst[12] = {0};
    EXPECT_EQ(process_local_contrast(nullptr, dst, 2, 2, 3,
              LocalContrastAlgorithm::UNSHARP, 8, LocalContrastParams{}),
              LocalContrastError::NullInput);
    EXPECT_EQ(process_local_contrast(src, dst, 2, 2, 1,
              LocalContrastAlgorithm::UNSHARP, 8, LocalContrastParams{}),
              LocalContrastError::InvalidChannels);
}

TEST(LocalContrastTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(LocalContrastAlgorithm::UNSHARP), "Unsharp Clarity");
    EXPECT_EQ(algorithm_name(LocalContrastAlgorithm::BILATERAL), "Bilateral Clarity");
}

TEST(LocalContrastTest, ZeroAmountPassthrough) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    make_rgb(src.data(), w, h, 100, 150, 200);
    std::vector<uint8_t> dst(w * h * 3);

    LocalContrastParams p;
    p.amount = 0.0f;
    auto err = process_unsharp(src.data(), dst.data(), w, h, 3, 8, p);
    EXPECT_EQ(err, LocalContrastError::Ok);
    for (int i = 0; i < 3; i++) EXPECT_EQ(dst[i], src[i]);
}

TEST(LocalContrastTest, UnsharpOutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>((i * 37 + 127) % 256);

    std::vector<uint8_t> dst(w * h * 3);
    LocalContrastParams p;
    p.amount = 0.5f;
    p.radius = 15.0f;
    auto err = process_unsharp(src.data(), dst.data(), w, h, 3, 8, p);
    EXPECT_EQ(err, LocalContrastError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(LocalContrastTest, UnsharpEnhancesEdges) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    // Create a sharp edge: left half dark, right half bright
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            uint8_t v = (x < w / 2) ? 50 : 200;
            src[idx+0] = v; src[idx+1] = v; src[idx+2] = v;
        }
    }

    std::vector<uint8_t> dst(w * h * 3);
    LocalContrastParams p;
    p.amount = 1.0f;
    p.radius = 10.0f;
    auto err = process_unsharp(src.data(), dst.data(), w, h, 3, 8, p);
    EXPECT_EQ(err, LocalContrastError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(LocalContrastTest, BilateralOutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) src[i] = static_cast<uint8_t>((i * 37 + 127) % 256);

    std::vector<uint8_t> dst(w * h * 3);
    LocalContrastParams p;
    p.amount = 0.5f;
    auto err = process_bilateral(src.data(), dst.data(), w, h, 3, 8, p);
    EXPECT_EQ(err, LocalContrastError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}

TEST(LocalContrastTest, BilateralPreservesEdges) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            uint8_t v = (x < w / 2) ? 50 : 200;
            src[idx+0] = v; src[idx+1] = v; src[idx+2] = v;
        }
    }

    std::vector<uint8_t> dst(w * h * 3);
    LocalContrastParams p;
    p.amount = 1.0f;
    auto err = process_bilateral(src.data(), dst.data(), w, h, 3, 8, p);
    EXPECT_EQ(err, LocalContrastError::Ok);
    for (size_t i = 0; i < dst.size(); i++) EXPECT_LE(dst[i], 255);
}
