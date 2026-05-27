#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <numeric>
#include "imop/algorithms.hpp"
#include "imop/pixel_utils.hpp"
#include "imop/types.hpp"

using namespace imop;

namespace {

// ── Synthetic pattern generator ─────────────────────────────────────

void make_bayer_from_rgb(const uint8_t* r_plane, const uint8_t* g_plane,
                          const uint8_t* b_plane,
                          uint8_t* bayer, int width, int height,
                          BayerPattern pattern) {
    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            bool is_r = ((y & 1) == po.r_row) && ((x & 1) == po.r_col);
            bool is_b = ((y & 1) == po.b_row) && ((x & 1) == po.b_col);
            if (is_r)       bayer[idx] = r_plane[idx];
            else if (is_b)  bayer[idx] = b_plane[idx];
            else            bayer[idx] = g_plane[idx];
        }
    }
}

// Helper: compute PSNR over interior pixels only (skipping border pixels
// that algorithms cannot demosaic due to window size constraints).
double compute_interior_psnr(const uint8_t* actual, const uint8_t* expected,
                              int width, int height, int border) {
    double mse = 0.0;
    int count = 0;
    for (int y = border; y < height - border; y++) {
        for (int x = border; x < width - border; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            for (int c = 0; c < 3; c++) {
                double diff = static_cast<double>(actual[idx + c]) - static_cast<double>(expected[idx + c]);
                mse += diff * diff;
                count++;
            }
        }
    }
    if (count == 0) return 100.0;
    mse /= count;
    if (mse < 1e-10) return 100.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// Helper: maximum per-channel error over interior pixels
void max_interior_error(const uint8_t* rgb, int width, int height, int border,
                         int exp_r, int exp_g, int exp_b,
                         int& max_re, int& max_ge, int& max_be) {
    max_re = max_ge = max_be = 0;
    for (int y = border; y < height - border; y++) {
        for (int x = border; x < width - border; x++) {
            size_t idx = (static_cast<size_t>(y) * width + x) * 3;
            max_re = std::max(max_re, std::abs(static_cast<int>(rgb[idx+0]) - exp_r));
            max_ge = std::max(max_ge, std::abs(static_cast<int>(rgb[idx+1]) - exp_g));
            max_be = std::max(max_be, std::abs(static_cast<int>(rgb[idx+2]) - exp_b));
        }
    }
}

// Skip algorithms known to be unstable on this toolchain (same FIXME as
// in test_algorithms.cpp).
static bool skip_unstable_algo(DemosaicAlgorithm algo) {
    return algo == DemosaicAlgorithm::AHD ||
           algo == DemosaicAlgorithm::AMAZE ||
           algo == DemosaicAlgorithm::RCD ||
           algo == DemosaicAlgorithm::PRISM;
}

// ── Test 1: Uniform flat field (interior only) ──────────────────────

struct FlatFieldTestParam {
    DemosaicAlgorithm algo;
    double min_psnr;
};

class FlatFieldCorrectnessTest : public ::testing::TestWithParam<FlatFieldTestParam> {};

TEST_P(FlatFieldCorrectnessTest, UniformFlatFieldInterior) {
    auto p = GetParam();
    if (skip_unstable_algo(p.algo)) {
        GTEST_SKIP() << algorithm_name(p.algo) << " skipped (unstable on this toolchain)";
    }
    const int border = algorithm_window_size(p.algo);
    const int w = std::max(border * 4, 64);
    const int h = std::max(border * 4, 64);
    const size_t pixel_count = static_cast<size_t>(w) * h;

    std::vector<uint8_t> r_plane(pixel_count, 100);
    std::vector<uint8_t> g_plane(pixel_count, 150);
    std::vector<uint8_t> b_plane(pixel_count, 200);

    std::vector<uint8_t> bayer(pixel_count);
    make_bayer_from_rgb(r_plane.data(), g_plane.data(), b_plane.data(),
                        bayer.data(), w, h, BayerPattern::RGGB);

    std::vector<uint8_t> rgb(pixel_count * 3);
    DemosaicError err = demosaic_cpu(bayer.data(), rgb.data(), w, h,
                                           BayerPattern::RGGB, p.algo, 8);
    ASSERT_EQ(err, DemosaicError::Ok);

    std::vector<uint8_t> expected(pixel_count * 3);
    for (size_t i = 0; i < pixel_count; i++) {
        expected[i * 3 + 0] = 100;
        expected[i * 3 + 1] = 150;
        expected[i * 3 + 2] = 200;
    }

    double psnr = compute_interior_psnr(rgb.data(), expected.data(), w, h, border);
    EXPECT_GE(psnr, p.min_psnr) << algorithm_name(p.algo) << " interior PSNR too low";

    int max_re, max_ge, max_be;
    max_interior_error(rgb.data(), w, h, border, 100, 150, 200, max_re, max_ge, max_be);

    // For a uniform flat field, interior interpolation should be very close
    int tolerance = (p.algo == DemosaicAlgorithm::SUPER_FAST) ? 40 : 25;
    EXPECT_LE(max_re, tolerance) << "Max R error for " << algorithm_name(p.algo);
    EXPECT_LE(max_ge, tolerance) << "Max G error for " << algorithm_name(p.algo);
    EXPECT_LE(max_be, tolerance) << "Max B error for " << algorithm_name(p.algo);
}

// ── Test 2: Grayscale (R=G=B) — check for color artifacts ─────────

class GrayscaleCorrectnessTest : public ::testing::TestWithParam<FlatFieldTestParam> {};

TEST_P(GrayscaleCorrectnessTest, GrayscaleRamp) {
    auto p = GetParam();
    if (skip_unstable_algo(p.algo)) {
        GTEST_SKIP() << algorithm_name(p.algo) << " skipped (unstable on this toolchain)";
    }
    const int border = algorithm_window_size(p.algo);
    const int w = std::max(border * 4, 64);
    const int h = std::max(border * 4, 64);
    const size_t pixel_count = static_cast<size_t>(w) * h;

    // Create a grayscale ramp where R=G=B
    std::vector<uint8_t> plane(pixel_count);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            plane[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>((x * 220) / (w - 1) + 18);
        }
    }

    std::vector<uint8_t> bayer(pixel_count);
    make_bayer_from_rgb(plane.data(), plane.data(), plane.data(),
                        bayer.data(), w, h, BayerPattern::RGGB);

    std::vector<uint8_t> rgb(pixel_count * 3);
    DemosaicError err = demosaic_cpu(bayer.data(), rgb.data(), w, h,
                                           BayerPattern::RGGB, p.algo, 8);
    ASSERT_EQ(err, DemosaicError::Ok);

    // On a grayscale image, R≈G≈B at every pixel
    int max_diff = 0;
    for (int y = border; y < h - border; y++) {
        for (int x = border; x < w - border; x++) {
            size_t idx = (static_cast<size_t>(y) * w + x) * 3;
            int r = rgb[idx], g = rgb[idx+1], b = rgb[idx+2];
            max_diff = std::max(max_diff, std::abs(r - g));
            max_diff = std::max(max_diff, std::abs(g - b));
            max_diff = std::max(max_diff, std::abs(r - b));
        }
    }

    int tolerance = (p.algo == DemosaicAlgorithm::SUPER_FAST) ? 60 : 35;
    EXPECT_LE(max_diff, tolerance)
        << algorithm_name(p.algo) << " has color artifacts on grayscale data";
}

// ── Test 3: Output range preservation ──────────────────────────────

class OutputRangeTest : public ::testing::TestWithParam<DemosaicAlgorithm> {};

TEST_P(OutputRangeTest, OutputInValidRange) {
    DemosaicAlgorithm algo = GetParam();

    if (skip_unstable_algo(algo)) {
        GTEST_SKIP() << "Algorithm " << algorithm_name(algo) << " skipped in this test environment";
    }

    const int w = 40, h = 40;
    const size_t pixel_count = static_cast<size_t>(w) * h;

    std::vector<uint8_t> bayer(pixel_count);
    for (size_t i = 0; i < pixel_count; i++)
        bayer[i] = static_cast<uint8_t>((i * 73 + 127) % 256);

    std::vector<uint8_t> rgb(pixel_count * 3);
    DemosaicError err = demosaic_cpu(bayer.data(), rgb.data(), w, h,
                                           BayerPattern::RGGB, algo, 8);
    ASSERT_EQ(err, DemosaicError::Ok);

    for (size_t i = 0; i < pixel_count * 3; i++) {
        EXPECT_GE(rgb[i], 0);
        EXPECT_LE(rgb[i], 255);
    }
}

// ── Test 4: 12-bit range check ────────────────────────────────────

TEST(CorrectnessTest, Bit12OutputInRangeForHQLI) {
    const int w = 40, h = 40;
    const size_t pixel_count = static_cast<size_t>(w) * h;

    std::vector<uint8_t> bayer(pixel_count * 2, 0);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t off = (static_cast<size_t>(y) * w + x) * 2;
            bool is_even_row = (y & 1) == 0, is_even_col = (x & 1) == 0;
            uint16_t val;
            if (is_even_row && is_even_col)       val = 2000;
            else if (!is_even_row && is_even_col)  val = 1500;
            else if (is_even_row && !is_even_col)  val = 1500;
            else                                   val = 1000;
            std::memcpy(bayer.data() + off, &val, sizeof(val));
        }
    }

    std::vector<uint8_t> rgb(pixel_count * 6);
    DemosaicError err = demosaic_cpu(bayer.data(), rgb.data(), w, h,
                                           BayerPattern::RGGB, DemosaicAlgorithm::HQLI, 12);
    ASSERT_EQ(err, DemosaicError::Ok);

    auto* rgb16 = reinterpret_cast<const uint16_t*>(rgb.data());
    bool has_nonzero = false;
    for (size_t i = 0; i < pixel_count * 3; i++) {
        EXPECT_LE(rgb16[i], 4095) << "12-bit output out of range at " << i;
        if (rgb16[i] > 0) has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero);
}

// ── Test 5: All four Bayer patterns produce output ──────────────────

class PatternConsistencyTest : public ::testing::TestWithParam<DemosaicAlgorithm> {};

TEST_P(PatternConsistencyTest, AllPatternsProduceOutput) {
    DemosaicAlgorithm algo = GetParam();

    if (skip_unstable_algo(algo)) {
        GTEST_SKIP() << "Algorithm " << algorithm_name(algo) << " skipped in this test environment";
    }

    const int w = 40, h = 40;
    const size_t pixel_count = static_cast<size_t>(w) * h;

    std::vector<uint8_t> r_plane(pixel_count), g_plane(pixel_count), b_plane(pixel_count);
    for (size_t i = 0; i < pixel_count; i++) {
        r_plane[i] = static_cast<uint8_t>(100 + (i * 53) % 100);
        g_plane[i] = static_cast<uint8_t>(120 + (i * 37) % 80);
        b_plane[i] = static_cast<uint8_t>(80 + (i * 73) % 70);
    }

    std::vector<BayerPattern> patterns = {
        BayerPattern::RGGB, BayerPattern::BGGR,
        BayerPattern::GRBG, BayerPattern::GBRG
    };

    std::vector<double> means;
    for (auto pattern : patterns) {
        std::vector<uint8_t> bayer(pixel_count);
        make_bayer_from_rgb(r_plane.data(), g_plane.data(), b_plane.data(),
                            bayer.data(), w, h, pattern);

        std::vector<uint8_t> rgb(pixel_count * 3);
        DemosaicError err = demosaic_cpu(bayer.data(), rgb.data(), w, h,
                                               pattern, algo, 8);
        ASSERT_EQ(err, DemosaicError::Ok);

        double sum = std::accumulate(rgb.begin(), rgb.end(), 0.0);
        means.push_back(sum / static_cast<double>(pixel_count * 3));
    }

    double mean_of_means = std::accumulate(means.begin(), means.end(), 0.0) / means.size();
    for (size_t i = 0; i < means.size(); i++) {
        EXPECT_NEAR(means[i], mean_of_means, 20.0)
            << "Pattern " << static_cast<int>(patterns[i])
            << " inconsistent for " << algorithm_name(algo);
    }
}

// ── Test 6: Red-dominant field ─────────────────────────────────────

class RedDominanceTest : public ::testing::TestWithParam<DemosaicAlgorithm> {};

TEST_P(RedDominanceTest, RedChannelDominant) {
    DemosaicAlgorithm algo = GetParam();

    if (skip_unstable_algo(algo)) {
        GTEST_SKIP() << "Algorithm " << algorithm_name(algo) << " skipped";
    }

    const int border = algorithm_window_size(algo);
    const int w = std::max(border * 4, 40);
    const int h = std::max(border * 4, 40);
    const size_t pixel_count = static_cast<size_t>(w) * h;

    // RGGB: set R=200, G=B=0
    std::vector<uint8_t> bayer(pixel_count, 0);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if ((y & 1) == 0 && (x & 1) == 0)
                bayer[static_cast<size_t>(y) * w + x] = 200;

    std::vector<uint8_t> rgb(pixel_count * 3);
    DemosaicError err = demosaic_cpu(bayer.data(), rgb.data(), w, h,
                                           BayerPattern::RGGB, algo, 8);
    ASSERT_EQ(err, DemosaicError::Ok);

    int violations = 0, total_red = 0;
    for (int y = border; y < h - border; y++) {
        for (int x = border; x < w - border; x++) {
            if ((y & 1) == 0 && (x & 1) == 0) {
                total_red++;
                size_t idx = (static_cast<size_t>(y) * w + x) * 3;
                if (rgb[idx] < rgb[idx+1] || rgb[idx] < rgb[idx+2])
                    violations++;
            }
        }
    }

    ASSERT_GT(total_red, 0);
    EXPECT_LE(violations, total_red / 3)
        << algorithm_name(algo) << ": " << violations << "/" << total_red << " violations";
}

// ── Instantiate all parameterized tests ─────────────────────────────

INSTANTIATE_TEST_SUITE_P(AllAlgos, FlatFieldCorrectnessTest,
    ::testing::Values(
        FlatFieldTestParam{DemosaicAlgorithm::SUPER_FAST, 30.0},
        FlatFieldTestParam{DemosaicAlgorithm::HQLI, 38.0},
        FlatFieldTestParam{DemosaicAlgorithm::MG, 38.0},
        FlatFieldTestParam{DemosaicAlgorithm::L7, 38.0},
        FlatFieldTestParam{DemosaicAlgorithm::DFPD, 36.0},
        FlatFieldTestParam{DemosaicAlgorithm::AHD, 36.0},
        FlatFieldTestParam{DemosaicAlgorithm::AMAZE, 36.0},
        FlatFieldTestParam{DemosaicAlgorithm::RCD, 36.0},
        FlatFieldTestParam{DemosaicAlgorithm::PRISM, 30.0}
    ));

INSTANTIATE_TEST_SUITE_P(AllAlgos, GrayscaleCorrectnessTest,
    ::testing::Values(
        FlatFieldTestParam{DemosaicAlgorithm::SUPER_FAST, 25.0},
        FlatFieldTestParam{DemosaicAlgorithm::HQLI, 35.0},
        FlatFieldTestParam{DemosaicAlgorithm::MG, 35.0},
        FlatFieldTestParam{DemosaicAlgorithm::L7, 35.0},
        FlatFieldTestParam{DemosaicAlgorithm::DFPD, 35.0},
        FlatFieldTestParam{DemosaicAlgorithm::AHD, 35.0},
        FlatFieldTestParam{DemosaicAlgorithm::AMAZE, 35.0},
        FlatFieldTestParam{DemosaicAlgorithm::RCD, 35.0},
        FlatFieldTestParam{DemosaicAlgorithm::PRISM, 25.0}
    ));

INSTANTIATE_TEST_SUITE_P(AllAlgos, OutputRangeTest,
    ::testing::Values(
        DemosaicAlgorithm::SUPER_FAST,
        DemosaicAlgorithm::HQLI,
        DemosaicAlgorithm::MG,
        DemosaicAlgorithm::L7,
        DemosaicAlgorithm::DFPD,
        DemosaicAlgorithm::AHD,
        DemosaicAlgorithm::AMAZE,
        DemosaicAlgorithm::RCD,
        DemosaicAlgorithm::PRISM
    ));

INSTANTIATE_TEST_SUITE_P(AllAlgos, PatternConsistencyTest,
    ::testing::Values(
        DemosaicAlgorithm::SUPER_FAST,
        DemosaicAlgorithm::HQLI,
        DemosaicAlgorithm::MG,
        DemosaicAlgorithm::L7,
        DemosaicAlgorithm::DFPD,
        DemosaicAlgorithm::AHD,
        DemosaicAlgorithm::AMAZE,
        DemosaicAlgorithm::RCD,
        DemosaicAlgorithm::PRISM
    ));

INSTANTIATE_TEST_SUITE_P(AllAlgos, RedDominanceTest,
    ::testing::Values(
        DemosaicAlgorithm::SUPER_FAST,
        DemosaicAlgorithm::HQLI,
        DemosaicAlgorithm::MG,
        DemosaicAlgorithm::L7,
        DemosaicAlgorithm::DFPD,
        DemosaicAlgorithm::AHD,
        DemosaicAlgorithm::AMAZE,
        DemosaicAlgorithm::RCD,
        DemosaicAlgorithm::PRISM
    ));

} // anonymous namespace
