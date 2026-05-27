#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "black_level/algorithms.hpp"

using namespace black_level;

TEST(BlackLevelTest, ValidatesInputs) {
    uint8_t data[16] = {0};
    EXPECT_EQ(process_black_level(nullptr, 4, 4, BayerPattern::RGGB,
                                   BlackLevelAlgorithm::GLOBAL, 8, BlackLevelParams{}),
              BlackLevelError::NullInput);
    EXPECT_EQ(process_black_level(data, 0, 4, BayerPattern::RGGB,
                                   BlackLevelAlgorithm::GLOBAL, 8, BlackLevelParams{}),
              BlackLevelError::InvalidDimensions);
    EXPECT_EQ(process_black_level(data, 4, 4, BayerPattern::RGGB,
                                   BlackLevelAlgorithm::GLOBAL, 0, BlackLevelParams{}),
              BlackLevelError::InvalidBitDepth);
}

TEST(BlackLevelTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(BlackLevelAlgorithm::PER_CHANNEL), "Per-Channel Black Level");
    EXPECT_EQ(algorithm_name(BlackLevelAlgorithm::GLOBAL), "Global Black Level");
}

TEST(BlackLevelTest, GlobalSubtractsConstant) {
    int w = 16, h = 16;
    std::vector<uint8_t> data(w * h, 200);
    BlackLevelParams p;
    p.r_offset = 64.0f;
    auto err = process_global(data.data(), w, h, 8, p);
    EXPECT_EQ(err, BlackLevelError::Ok);
    for (size_t i = 0; i < data.size(); i++) EXPECT_LE(data[i], 136);
}

TEST(BlackLevelTest, GlobalClampsToZero) {
    int w = 8, h = 8;
    std::vector<uint8_t> data(w * h, 40);
    BlackLevelParams p;
    p.r_offset = 100.0f;
    auto err = process_global(data.data(), w, h, 8, p);
    EXPECT_EQ(err, BlackLevelError::Ok);
    for (size_t i = 0; i < data.size(); i++) EXPECT_EQ(data[i], 0);
}

TEST(BlackLevelTest, PerChannelRGGB) {
    int w = 8, h = 8;
    std::vector<uint8_t> data(w * h, 100);
    BlackLevelParams p;
    p.r_offset = 10.0f;
    p.gr_offset = 20.0f;
    p.gb_offset = 30.0f;
    p.b_offset = 40.0f;
    auto err = process_per_channel(data.data(), w, h, BayerPattern::RGGB, 8, p);
    EXPECT_EQ(err, BlackLevelError::Ok);
    // R at (0,0): 100-10=90; Gr at (0,1): 100-20=80; Gb at (1,0): 100-30=70; B at (1,1): 100-40=60
    EXPECT_EQ(data[0], 90); // R
    EXPECT_EQ(data[1], 80); // Gr
    EXPECT_EQ(data[static_cast<size_t>(1)*w + 0], 70); // Gb
    EXPECT_EQ(data[static_cast<size_t>(1)*w + 1], 60); // B
}

TEST(BlackLevelTest, OutputInRange8Bit) {
    int w = 32, h = 32;
    std::vector<uint8_t> data(w * h);
    for (size_t i = 0; i < data.size(); i++) data[i] = static_cast<uint8_t>(i % 256);
    BlackLevelParams p;
    p.r_offset = 32.0f; p.gr_offset = 32.0f; p.gb_offset = 32.0f; p.b_offset = 32.0f;
    auto err = process_black_level(data.data(), w, h, BayerPattern::RGGB,
                                    BlackLevelAlgorithm::PER_CHANNEL, 8, p);
    EXPECT_EQ(err, BlackLevelError::Ok);
    for (size_t i = 0; i < data.size(); i++) EXPECT_LE(data[i], 255);
}
