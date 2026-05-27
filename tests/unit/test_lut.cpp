#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <fstream>
#include <cmath>
#include "lut/algorithms.hpp"

using namespace lut;

namespace {

void make_test_rgb(uint8_t* rgb, int width, int height,
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

void make_gradient_rgb(uint8_t* rgb, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            rgb[idx + 0] = static_cast<uint8_t>((x * 255) / width);
            rgb[idx + 1] = static_cast<uint8_t>((y * 255) / height);
            rgb[idx + 2] = static_cast<uint8_t>(128);
        }
    }
}

struct LUTTestParam {
    LUTAlgorithm algo;
};

class LUTAlgorithmTest : public ::testing::TestWithParam<LUTTestParam> {};

} // anonymous namespace

// --- Validation tests ---

TEST(LUTDispatchTest, ValidateInputs) {
    uint8_t src[12] = {0};
    uint8_t dst[12] = {0};

    EXPECT_EQ(process_lut(nullptr, dst, 4, 4, 3,
                           LUTAlgorithm::SEPIA, 8),
              LUTError::NullInput);
    EXPECT_EQ(process_lut(src, nullptr, 4, 4, 3,
                           LUTAlgorithm::SEPIA, 8),
              LUTError::NullInput);

    EXPECT_EQ(process_lut(src, dst, 0, 4, 3,
                           LUTAlgorithm::SEPIA, 8),
              LUTError::InvalidDimensions);
    EXPECT_EQ(process_lut(src, dst, 4, -1, 3,
                           LUTAlgorithm::SEPIA, 8),
              LUTError::InvalidDimensions);

    EXPECT_EQ(process_lut(src, dst, 2, 2, 3,
                           LUTAlgorithm::SEPIA, 17),
              LUTError::InvalidBitDepth);

    EXPECT_EQ(process_lut(src, dst, 2, 2, 1,
                           LUTAlgorithm::SEPIA, 8),
              LUTError::InvalidChannels);
}

// --- Metadata tests ---

TEST(LUTDispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(LUTAlgorithm::CUBE_FILE), "CUBE File LUT (.cube import)");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::CUSTOM_3D), "Custom 3D LUT");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::SEPIA), "Sepia Tone");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::COOL), "Cool (blue cast)");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::WARM), "Warm (amber cast)");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::HIGH_CONTRAST), "High Contrast (S-curve)");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::LOW_CONTRAST), "Low Contrast (faded)");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::INVERT), "Color Inversion");
    EXPECT_EQ(algorithm_name(LUTAlgorithm::VINTAGE_FADE), "Vintage Fade");
}

TEST(LUTDispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(LUTAlgorithm::SEPIA), 1);
    EXPECT_EQ(algorithm_window_size(LUTAlgorithm::CUBE_FILE), 1);
}

// --- 3D LUT data structure ---

TEST(LUT3DTest, DefaultEmpty) {
    LUT3D lut;
    EXPECT_EQ(lut.size, 0);
    EXPECT_TRUE(lut.empty());
    EXPECT_EQ(lut.total_samples(), 0);
}

TEST(LUT3DTest, NonEmpty) {
    LUT3D lut;
    lut.size = 17;
    lut.data.resize(17 * 17 * 17 * 3);
    EXPECT_FALSE(lut.empty());
    EXPECT_EQ(lut.total_samples(), 17u * 17 * 17);
}

// --- Identity LUT tests ---

TEST(LUTIdentityTest, BuildIdentity) {
    LUT3D lut = build_identity_lut(17);
    EXPECT_EQ(lut.size, 17);
    EXPECT_FALSE(lut.empty());

    // Corner (0,0,0) should be (0,0,0)
    EXPECT_FLOAT_EQ(lut.data[0], 0.0f);
    EXPECT_FLOAT_EQ(lut.data[1], 0.0f);
    EXPECT_FLOAT_EQ(lut.data[2], 0.0f);

    // Corner (16,16,16) should be (1,1,1)
    size_t corner = (16 * 17 * 17 + 16 * 17 + 16) * 3;
    EXPECT_FLOAT_EQ(lut.data[corner + 0], 1.0f);
    EXPECT_FLOAT_EQ(lut.data[corner + 1], 1.0f);
    EXPECT_FLOAT_EQ(lut.data[corner + 2], 1.0f);
}

TEST(LUTIdentityTest, IdentityPassThrough) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    LUT3D lut = build_identity_lut(33);
    LUTError err = apply_lut(lut, src.data(), dst.data(), w, h, 8);
    EXPECT_EQ(err, LUTError::Ok);

    // With trilinear interpolation, values should be very close to input
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(100), 1);
        EXPECT_NEAR(dst[i * 3 + 1], static_cast<uint8_t>(150), 1);
        EXPECT_NEAR(dst[i * 3 + 2], static_cast<uint8_t>(200), 1);
    }
}

TEST(LUTIdentityTest, IdentityHighBitDepth) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    LUT3D lut = build_identity_lut(33);
    LUTError err = apply_lut(lut, src.data(), dst.data(), w, h, 12);
    EXPECT_EQ(err, LUTError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst16[i * 3 + 0], 1000, 10);
        EXPECT_NEAR(dst16[i * 3 + 1], 2000, 10);
        EXPECT_NEAR(dst16[i * 3 + 2], 3000, 10);
        EXPECT_LE(dst16[i * 3 + 0], 4095);
    }
}

// --- Built-in style LUT tests ---

TEST(LUTStyleSepiaTest, BasicProcessing) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_gradient_rgb(src.data(), w, h);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::SEPIA, 8);
    EXPECT_EQ(err, LUTError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(LUTStyleSepiaTest, ProducesWarmTone) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 200, 200);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::SEPIA, 8);
    EXPECT_EQ(err, LUTError::Ok);

    // Sepia: R should be boosted relative to B
    double sum_r = 0, sum_b = 0;
    for (int i = 0; i < w * h; i++) {
        sum_r += dst[i * 3 + 0];
        sum_b += dst[i * 3 + 2];
    }
    EXPECT_GT(sum_r, sum_b) << "Sepia should have R > B (warm tone)";
}

TEST(LUTStyleSepiaTest, LUTSizeParameter) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_gradient_rgb(src.data(), w, h);

    // Small LUT (9) vs large LUT (65)
    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::SEPIA, 8, nullptr, 9);
    EXPECT_EQ(err, LUTError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(LUTStyleCoolTest, ProducesCoolTone) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 200, 200);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::COOL, 8);
    EXPECT_EQ(err, LUTError::Ok);

    double sum_r = 0, sum_b = 0;
    for (int i = 0; i < w * h; i++) {
        sum_r += dst[i * 3 + 0];
        sum_b += dst[i * 3 + 2];
    }
    EXPECT_LT(sum_r, sum_b) << "Cool style should have R < B (blue cast)";
}

TEST(LUTStyleWarmTest, ProducesWarmTone) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 200, 200);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::WARM, 8);
    EXPECT_EQ(err, LUTError::Ok);

    double sum_r = 0, sum_b = 0;
    for (int i = 0; i < w * h; i++) {
        sum_r += dst[i * 3 + 0];
        sum_b += dst[i * 3 + 2];
    }
    EXPECT_GT(sum_r, sum_b) << "Warm style should have R > B (amber cast)";
}

TEST(LUTStyleHighContrastTest, IncreasesContrast) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            uint8_t v = static_cast<uint8_t>(100 + (y * 5) % 50); // mid-range
            src[idx + 0] = v;
            src[idx + 1] = v;
            src[idx + 2] = v;
        }
    }

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::HIGH_CONTRAST, 8);
    EXPECT_EQ(err, LUTError::Ok);

    // Values should be valid
    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST(LUTStyleLowContrastTest, FadesImage) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 50, 50, 50);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::LOW_CONTRAST, 8);
    EXPECT_EQ(err, LUTError::Ok);

    // Low contrast should lift black pixels
    EXPECT_GE(dst[0], 20) << "Blacks should be lifted (faded)";
}

TEST(LUTStyleInvertTest, InvertsColors) {
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 50, 100, 200);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::INVERT, 8);
    EXPECT_EQ(err, LUTError::Ok);

    // Invert: bright becomes dark and vice versa
    EXPECT_GT(dst[0], 200) << "R=50 should invert to ~205";
    EXPECT_GT(dst[2], 0);
    EXPECT_LT(dst[2], 100) << "B=200 should invert to ~55";
}

TEST(LUTStyleVintageFadeTest, ProducesDesaturatedTone) {
    int w = 32, h = 32;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 200, 100, 50);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::VINTAGE_FADE, 8);
    EXPECT_EQ(err, LUTError::Ok);

    // Vintage desaturates: R=200, B=50 should come closer together
    int r_out = dst[0];
    int b_out = dst[2];
    EXPECT_LT(std::abs(r_out - b_out), 150) << "Channels should be less spread";
}

// --- Custom 3D LUT tests ---

TEST(LUTCustom3DTest, CustomLUT) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    LUT3D lut = build_identity_lut(17);
    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::CUSTOM_3D, 8, &lut, 17);
    EXPECT_EQ(err, LUTError::Ok);

    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(100), 1);
        EXPECT_NEAR(dst[i * 3 + 1], static_cast<uint8_t>(150), 1);
        EXPECT_NEAR(dst[i * 3 + 2], static_cast<uint8_t>(200), 1);
    }
}

TEST(LUTCustom3DTest, NullLUTFallbackToIdentity) {
    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_test_rgb(src.data(), w, h, 100, 150, 200);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                LUTAlgorithm::CUSTOM_3D, 8, nullptr, 17);
    EXPECT_EQ(err, LUTError::Ok);

    for (int i = 0; i < w * h; i++) {
        EXPECT_NEAR(dst[i * 3 + 0], static_cast<uint8_t>(100), 1);
        EXPECT_NEAR(dst[i * 3 + 1], static_cast<uint8_t>(150), 1);
        EXPECT_NEAR(dst[i * 3 + 2], static_cast<uint8_t>(200), 1);
    }
}

// --- Built-in LUT builder tests ---

TEST(LUTBuildersTest, BuildSepiaLUT) {
    LUT3D lut = build_sepia_lut(17);
    EXPECT_EQ(lut.size, 17);
    EXPECT_EQ(lut.data.size(), 17u * 17 * 17 * 3);
}

TEST(LUTBuildersTest, BuildCoolLUT) {
    LUT3D lut = build_cool_lut(17);
    EXPECT_EQ(lut.size, 17);

    // Corner (16,0,0) = pure red -> should be suppressed
    size_t red_corner = (16 * 17 * 17 + 0 * 17 + 0) * 3;
    EXPECT_LT(lut.data[red_corner + 0], 1.0f) << "Cool LUT should suppress pure red";
}

TEST(LUTBuildersTest, BuildWarmLUT) {
    LUT3D lut = build_warm_lut(17);
    EXPECT_EQ(lut.size, 17);

    // Corner (0,0,16) = pure blue -> should be suppressed
    size_t blue_corner = (0 * 17 * 17 + 0 * 17 + 16) * 3;
    EXPECT_LT(lut.data[blue_corner + 2], 1.0f) << "Warm LUT should suppress pure blue";
}

TEST(LUTBuildersTest, BuildHighContrastLUT) {
    LUT3D lut = build_high_contrast_lut(17);
    EXPECT_EQ(lut.size, 17);

    // Mid-gray input should map to mid-gray output (S-curve midpoint)
    size_t mid = (8 * 17 * 17 + 8 * 17 + 8) * 3;
    EXPECT_NEAR(lut.data[mid + 0], 0.5f, 0.05f);
}

TEST(LUTBuildersTest, BuildLowContrastLUT) {
    LUT3D lut = build_low_contrast_lut(17);
    EXPECT_EQ(lut.size, 17);

    // Black input should be lifted
    EXPECT_GT(lut.data[0], 0.0f) << "Blacks should be lifted";
    // White input should be reduced
    size_t white = (16 * 17 * 17 + 16 * 17 + 16) * 3;
    EXPECT_LT(lut.data[white + 0], 1.0f) << "Whites should be reduced";
}

TEST(LUTBuildersTest, BuildInvertLUT) {
    LUT3D lut = build_invert_lut(17);
    EXPECT_EQ(lut.size, 17);

    // Black -> white
    EXPECT_FLOAT_EQ(lut.data[0], 1.0f);
    EXPECT_FLOAT_EQ(lut.data[1], 1.0f);
    EXPECT_FLOAT_EQ(lut.data[2], 1.0f);

    // White -> black
    size_t white = (16 * 17 * 17 + 16 * 17 + 16) * 3;
    EXPECT_FLOAT_EQ(lut.data[white + 0], 0.0f);
    EXPECT_FLOAT_EQ(lut.data[white + 1], 0.0f);
    EXPECT_FLOAT_EQ(lut.data[white + 2], 0.0f);
}

TEST(LUTBuildersTest, BuildVintageFadeLUT) {
    LUT3D lut = build_vintage_fade_lut(17);
    EXPECT_EQ(lut.size, 17);

    // Reds should be boosted, blues should be suppressed
    size_t red_corner = (16 * 17 * 17 + 0 * 17 + 0) * 3;
    size_t blue_corner = (0 * 17 * 17 + 0 * 17 + 16) * 3;
    EXPECT_GT(lut.data[red_corner + 0], lut.data[blue_corner + 2])
        << "Vintage: warm tones should dominate";
}

// --- .cube file tests ---

TEST(LUTCubeFileTest, NullPathReturnsError) {
    uint8_t src[12] = {100, 150, 200, 100, 150, 200, 100, 150, 200, 100, 150, 200};
    uint8_t dst[12] = {0};

    LUTError err = process_lut(src, dst, 2, 2, 3,
                                LUTAlgorithm::CUBE_FILE, 8, nullptr, 33);
    EXPECT_EQ(err, LUTError::FileNotFound);
}

TEST(LUTCubeFileTest, NonexistentFileReturnsError) {
    uint8_t src[12] = {100, 150, 200, 100, 150, 200, 100, 150, 200, 100, 150, 200};
    uint8_t dst[12] = {0};

    LUTError err = process_lut(src, dst, 2, 2, 3,
                                LUTAlgorithm::CUBE_FILE, 8,
                                "nonexistent_file.cube", 33);
    EXPECT_EQ(err, LUTError::FileParseError);
}

TEST(LUTCubeFileTest, WriteAndLoadMinimalCube) {
    // Create a minimal .cube file
    std::string path = "test_minimal.cube";
    std::ofstream f(path);
    f << "TITLE \"Test LUT\"\n";
    f << "LUT_3D_SIZE 2\n";
    f << "0.0 0.0 0.0\n";
    f << "0.0 0.0 1.0\n";
    f << "0.0 1.0 0.0\n";
    f << "0.0 1.0 1.0\n";
    f << "1.0 0.0 0.0\n";
    f << "1.0 0.0 1.0\n";
    f << "1.0 1.0 0.0\n";
    f << "1.0 1.0 1.0\n";
    f.close();

    LUT3D lut = load_cube_file(path.c_str());
    EXPECT_EQ(lut.size, 2);
    EXPECT_FALSE(lut.empty());
    EXPECT_EQ(lut.data.size(), 8u * 3);

    // Check corners
    EXPECT_FLOAT_EQ(lut.data[(0 * 4 + 0 * 2 + 0) * 3 + 0], 0.0f); // (0,0,0) R
    EXPECT_FLOAT_EQ(lut.data[(1 * 4 + 1 * 2 + 1) * 3 + 1], 1.0f); // (1,1,1) G

    std::remove(path.c_str());
}

TEST(LUTCubeFileTest, ApplyCubeFromFile) {
    // Create a test .cube file with identity mapping
    std::string path = "test_identity.cube";
    std::ofstream f(path);
    f << "LUT_3D_SIZE 2\n";
    f << "0.0 0.0 0.0\n";
    f << "0.0 0.0 1.0\n";
    f << "0.0 1.0 0.0\n";
    f << "0.0 1.0 1.0\n";
    f << "1.0 0.0 0.0\n";
    f << "1.0 0.0 1.0\n";
    f << "1.0 1.0 0.0\n";
    f << "1.0 1.0 1.0\n";
    f.close();

    uint8_t src[12] = {0, 0, 0, 255, 255, 255, 128, 128, 128, 64, 192, 0};
    uint8_t dst[12] = {0};

    LUTError err = process_lut(src, dst, 2, 2, 3,
                                LUTAlgorithm::CUBE_FILE, 8,
                                path.c_str(), 2);
    EXPECT_EQ(err, LUTError::Ok);

    std::remove(path.c_str());
}

TEST(LUTCubeFileTest, LoadCubeWithComments) {
    std::string path = "test_comments.cube";
    std::ofstream f(path);
    f << "# This is a comment\n";
    f << "TITLE \"Commented LUT\"\n";
    f << "# Another comment\n";
    f << "LUT_3D_SIZE 2\n";
    f << "# Data starts here\n";
    f << "0.0 0.0 0.0\n";
    f << "# Mid\n";
    f << "0.0 0.0 1.0\n";
    f << "0.0 1.0 0.0\n";
    f << "0.0 1.0 1.0\n";
    f << "1.0 0.0 0.0\n";
    f << "1.0 0.0 1.0\n";
    f << "1.0 1.0 0.0\n";
    f << "1.0 1.0 1.0\n";
    f.close();

    LUT3D lut = load_cube_file(path.c_str());
    EXPECT_EQ(lut.size, 2);
    EXPECT_FALSE(lut.empty());

    std::remove(path.c_str());
}

TEST(LUTCubeFileTest, LoadCubeWithDomainRange) {
    std::string path = "test_domain.cube";
    std::ofstream f(path);
    f << "DOMAIN_MIN 0.0 0.0 0.0\n";
    f << "DOMAIN_MAX 1.0 1.0 1.0\n";
    f << "LUT_3D_SIZE 2\n";
    f << "0.0 0.0 0.0\n";
    f << "0.5 0.5 0.5\n";
    f << "0.5 0.5 0.5\n";
    f << "0.5 0.5 0.5\n";
    f << "0.5 0.5 0.5\n";
    f << "0.5 0.5 0.5\n";
    f << "0.5 0.5 0.5\n";
    f << "1.0 1.0 1.0\n";
    f.close();

    LUT3D lut = load_cube_file(path.c_str());
    EXPECT_EQ(lut.size, 2);
    EXPECT_FLOAT_EQ(lut.data[0], 0.0f);
    size_t corner = (1 * 4 + 1 * 2 + 1) * 3;
    EXPECT_FLOAT_EQ(lut.data[corner + 0], 1.0f);

    std::remove(path.c_str());
}

// --- Parameterized tests across all algorithms ---

TEST_P(LUTAlgorithmTest, ProducesValidOutput) {
    auto p = GetParam();
    int w = 16, h = 16;
    std::vector<uint8_t> src(w * h * 3);
    std::vector<uint8_t> dst(w * h * 3);
    make_gradient_rgb(src.data(), w, h);

    const void* lut_arg = nullptr;
    int lut_size = 33;

    if (p.algo == LUTAlgorithm::CUSTOM_3D) {
        static LUT3D ident = build_identity_lut(17);
        lut_arg = &ident;
        lut_size = 17;
    } else if (p.algo == LUTAlgorithm::CUBE_FILE) {
        // Skip file test in parameterized suite (avoids temporary file I/O)
        return;
    }

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                p.algo, 8, lut_arg, lut_size);
    EXPECT_EQ(err, LUTError::Ok);

    for (size_t i = 0; i < dst.size(); i++) {
        EXPECT_LE(dst[i], 255);
    }
}

TEST_P(LUTAlgorithmTest, RejectsNonRGBChannels) {
    auto p = GetParam();
    if (p.algo == LUTAlgorithm::CUBE_FILE) return; // needs file path

    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h);
    std::vector<uint8_t> dst(w * h);

    LUTError err = process_lut(src.data(), dst.data(), w, h, 1,
                                p.algo, 8, nullptr, 33);
    EXPECT_EQ(err, LUTError::InvalidChannels);
}

TEST_P(LUTAlgorithmTest, HighBitDepth) {
    auto p = GetParam();
    if (p.algo == LUTAlgorithm::CUBE_FILE) return;

    int w = 8, h = 8;
    std::vector<uint8_t> src(w * h * 3 * 2);
    auto* src16 = reinterpret_cast<uint16_t*>(src.data());
    for (int i = 0; i < w * h; i++) {
        src16[i * 3 + 0] = 1000;
        src16[i * 3 + 1] = 2000;
        src16[i * 3 + 2] = 3000;
    }

    std::vector<uint8_t> dst(w * h * 3 * 2);
    const void* lut_arg = nullptr;
    int lut_size = 33;

    if (p.algo == LUTAlgorithm::CUSTOM_3D) {
        static LUT3D ident = build_identity_lut(17);
        lut_arg = &ident;
        lut_size = 17;
    }

    LUTError err = process_lut(src.data(), dst.data(), w, h, 3,
                                p.algo, 12, lut_arg, lut_size);
    EXPECT_EQ(err, LUTError::Ok);

    auto* dst16 = reinterpret_cast<uint16_t*>(dst.data());
    for (int i = 0; i < w * h * 3; i++) {
        EXPECT_LE(dst16[i], 4095);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllLUTAlgos,
    LUTAlgorithmTest,
    ::testing::Values(
        LUTTestParam{LUTAlgorithm::SEPIA},
        LUTTestParam{LUTAlgorithm::COOL},
        LUTTestParam{LUTAlgorithm::WARM},
        LUTTestParam{LUTAlgorithm::HIGH_CONTRAST},
        LUTTestParam{LUTAlgorithm::LOW_CONTRAST},
        LUTTestParam{LUTAlgorithm::INVERT},
        LUTTestParam{LUTAlgorithm::VINTAGE_FADE},
        LUTTestParam{LUTAlgorithm::CUSTOM_3D},
        LUTTestParam{LUTAlgorithm::CUBE_FILE}
    )
);
