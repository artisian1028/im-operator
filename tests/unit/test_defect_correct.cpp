#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include "defect_correct/algorithms.hpp"

using namespace defect_correct;

TEST(DefectCorrectTest, ValidatesInputs) {
    uint8_t data[100] = {0};
    EXPECT_EQ(process_defect_correct(nullptr, 10, 10, BayerPattern::RGGB,
                                      DefectCorrectAlgorithm::ADAPTIVE, 8, DefectCorrectParams{}),
              DefectCorrectError::NullInput);
    EXPECT_EQ(process_defect_correct(data, 0, 10, BayerPattern::RGGB,
                                      DefectCorrectAlgorithm::ADAPTIVE, 8, DefectCorrectParams{}),
              DefectCorrectError::InvalidDimensions);
}

TEST(DefectCorrectTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(DefectCorrectAlgorithm::ADAPTIVE), "Adaptive Defect Correction");
    EXPECT_EQ(algorithm_name(DefectCorrectAlgorithm::MAP_BASED), "Map-Based Defect Correction");
}

TEST(DefectCorrectTest, AdaptiveFixesHotPixel) {
    int w = 8, h = 8;
    std::vector<uint8_t> data(w * h, 50);
    // Insert a hot pixel at (4,4) — should be correctable
    data[static_cast<size_t>(4) * w + 4] = 255;

    DefectCorrectParams p;
    p.threshold = 0.3f;
    auto err = process_adaptive(data.data(), w, h, BayerPattern::RGGB, 8, p);
    EXPECT_EQ(err, DefectCorrectError::Ok);

    // The hot pixel should be suppressed (no longer 255)
    EXPECT_LT(data[static_cast<size_t>(4) * w + 4], 200);
}

TEST(DefectCorrectTest, AdaptivePreservesNormalPixels) {
    int w = 8, h = 8;
    std::vector<uint8_t> data(w * h, 128);
    DefectCorrectParams p;
    p.threshold = 0.3f;
    auto err = process_adaptive(data.data(), w, h, BayerPattern::RGGB, 8, p);
    EXPECT_EQ(err, DefectCorrectError::Ok);
    // All pixels should stay close to 128 (flat field)
    for (size_t i = 0; i < data.size(); i++) EXPECT_NEAR(data[i], 128, 5);
}

TEST(DefectCorrectTest, MapBasedFixesKnownDefects) {
    int w = 8, h = 8;
    std::vector<uint8_t> data(w * h, 100);
    // Mark (3,3) as defective
    data[static_cast<size_t>(3) * w + 3] = 255;

    DefectPoint defects[] = {{3, 3}};
    DefectCorrectParams p;
    p.map = defects;
    p.map_count = 1;
    auto err = process_map_based(data.data(), w, h, BayerPattern::RGGB, 8, p);
    EXPECT_EQ(err, DefectCorrectError::Ok);
    EXPECT_LT(data[static_cast<size_t>(3) * w + 3], 200);
}

TEST(DefectCorrectTest, OutputInRange) {
    int w = 16, h = 16;
    std::vector<uint8_t> data(w * h);
    for (size_t i = 0; i < data.size(); i++) data[i] = static_cast<uint8_t>(i % 256);
    auto err = process_defect_correct(data.data(), w, h, BayerPattern::RGGB,
                                       DefectCorrectAlgorithm::ADAPTIVE, 8, DefectCorrectParams{});
    EXPECT_EQ(err, DefectCorrectError::Ok);
    for (size_t i = 0; i < data.size(); i++) EXPECT_LE(data[i], 255);
}
