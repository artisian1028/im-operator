#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <cmath>
#include "lens_shading/algorithms.hpp"

using namespace lens_shading;

TEST(LensShadingTest, ValidatesInputs) {
    uint8_t data[16] = {0};
    EXPECT_EQ(process_lens_shading(nullptr, 4, 4, BayerPattern::RGGB,
              LensShadingAlgorithm::POLYNOMIAL, 8, LensShadingParams{}),
              LensShadingError::NullInput);
    EXPECT_EQ(process_lens_shading(data, 0, 4, BayerPattern::RGGB,
              LensShadingAlgorithm::POLYNOMIAL, 8, LensShadingParams{}),
              LensShadingError::InvalidDimensions);
}

TEST(LensShadingTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(LensShadingAlgorithm::POLYNOMIAL), "Polynomial Lens Shading Correction");
    EXPECT_EQ(algorithm_name(LensShadingAlgorithm::FLAT_FIELD), "Flat-Field Lens Shading Correction");
}

TEST(LensShadingTest, PolynomialIdentity) {
    int w = 16, h = 16;
    std::vector<uint8_t> data(w * h, 128);
    // All-zero coefficients = gain=1 everywhere
    auto err = process_polynomial(data.data(), w, h, BayerPattern::RGGB, 8, LensShadingParams{});
    EXPECT_EQ(err, LensShadingError::Ok);
    for (size_t i = 0; i < data.size(); i++) EXPECT_EQ(data[i], 128);
}

TEST(LensShadingTest, PolynomialBrightensCorners) {
    int w = 100, h = 100;
    std::vector<uint8_t> data(w * h, 120);
    LensShadingParams p;
    p.r_coef = {0.8f, 0.0f, 0.0f};  // strong a2 for vignetting correction
    p.gr_coef = {0.8f, 0.0f, 0.0f};
    p.gb_coef = {0.8f, 0.0f, 0.0f};
    p.b_coef = {0.8f, 0.0f, 0.0f};

    auto err = process_polynomial(data.data(), w, h, BayerPattern::RGGB, 8, p);
    EXPECT_EQ(err, LensShadingError::Ok);

    // Center pixel should stay ~120
    int cx = w / 2;
    EXPECT_NEAR(data[static_cast<size_t>(h/2) * w + cx], 120, 5);
    // Corner pixel should be brighter
    EXPECT_GT(data[0], 120);
}

TEST(LensShadingTest, PolynomialRespectsBayerChannels) {
    int w = 20, h = 20;
    std::vector<uint8_t> data(w * h, 100);
    // Only boost R channel
    LensShadingParams p;
    p.r_coef = {1.0f, 0.0f, 0.0f};
    p.gr_coef = {0.0f, 0.0f, 0.0f};
    p.gb_coef = {0.0f, 0.0f, 0.0f};
    p.b_coef = {0.0f, 0.0f, 0.0f};

    auto err = process_polynomial(data.data(), w, h, BayerPattern::RGGB, 8, p);
    EXPECT_EQ(err, LensShadingError::Ok);
    // R at (0,0) should change; Gr at (0,1) should stay 100
    EXPECT_GT(data[0], 100); // R at (0,0) — gain > 1 at corner
    EXPECT_EQ(data[1], 100); // Gr at (0,1) — gain = 1
}

TEST(LensShadingTest, OutputInRange) {
    int w = 32, h = 32;
    std::vector<uint8_t> data(w * h);
    for (size_t i = 0; i < data.size(); i++) data[i] = static_cast<uint8_t>(i % 256);

    LensShadingParams p;
    p.r_coef = {0.5f, 0.2f, 0.0f};
    p.gr_coef = {0.5f, 0.2f, 0.0f};
    p.gb_coef = {0.5f, 0.2f, 0.0f};
    p.b_coef = {0.5f, 0.2f, 0.0f};

    auto err = process_lens_shading(data.data(), w, h, BayerPattern::BGGR,
                                     LensShadingAlgorithm::POLYNOMIAL, 8, p);
    EXPECT_EQ(err, LensShadingError::Ok);
    for (size_t i = 0; i < data.size(); i++) EXPECT_LE(data[i], 255);
}

TEST(LensShadingTest, FlatFieldRequiresInput) {
    uint8_t data[16] = {0};
    auto err = process_flat_field(data, 4, 4, BayerPattern::RGGB, 8, LensShadingParams{});
    EXPECT_EQ(err, LensShadingError::NullInput);
}

TEST(LensShadingTest, FlatFieldIdentity) {
    int w = 32, h = 32;
    // Flat field with uniform value — gain should be ~1 everywhere
    std::vector<uint8_t> ff(w * h, 200);
    std::vector<uint8_t> data(w * h, 150);

    LensShadingParams p;
    p.flat_field = ff.data();
    p.flat_field_width = w;
    p.flat_field_height = h;

    auto err = process_flat_field(data.data(), w, h, BayerPattern::RGGB, 8, p);
    EXPECT_EQ(err, LensShadingError::Ok);
    // With uniform flat field, all gains should be ~1, output ≈ 150
    for (size_t i = 0; i < data.size(); i++) EXPECT_NEAR(data[i], 150, 2);
}
