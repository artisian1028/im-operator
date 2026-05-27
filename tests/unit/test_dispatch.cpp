#include <gtest/gtest.h>
#include "imop/algorithms.hpp"
#include "imop/types.hpp"

using namespace imop;

TEST(DispatchTest, ValidateDemosaicInputs) {
    // We test via demosaic which internally calls validate
    uint8_t src[16] = {0};
    uint8_t dst[48] = {0};

    // Null input
    EXPECT_EQ(demosaic(nullptr, dst, 2, 2, BayerPattern::RGGB,
                              DemosaicAlgorithm::SUPER_FAST, 8), DemosaicError::NullInput);
    EXPECT_EQ(demosaic(src, nullptr, 2, 2, BayerPattern::RGGB,
                              DemosaicAlgorithm::SUPER_FAST, 8), DemosaicError::NullInput);

    // Invalid dimensions
    EXPECT_EQ(demosaic(src, dst, 0, 2, BayerPattern::RGGB,
                              DemosaicAlgorithm::SUPER_FAST, 8), DemosaicError::InvalidDimensions);
    EXPECT_EQ(demosaic(src, dst, 2, -1, BayerPattern::RGGB,
                              DemosaicAlgorithm::SUPER_FAST, 8), DemosaicError::InvalidDimensions);

    // Invalid bit depth
    EXPECT_EQ(demosaic(src, dst, 2, 2, BayerPattern::RGGB,
                              DemosaicAlgorithm::SUPER_FAST, 17), DemosaicError::InvalidBitDepth);

    // Image too small for algorithm
    EXPECT_EQ(demosaic(src, dst, 2, 2, BayerPattern::RGGB,
                              DemosaicAlgorithm::HQLI, 8), DemosaicError::ImageTooSmall);
}

TEST(DispatchTest, AlgorithmNames) {
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::SUPER_FAST), "SUPER_FAST (Nearest)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::HQLI), "HQLI (5x5)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::MG), "MG (Malvar-He-Cutler 5x5)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::L7), "L7 (7x7)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::DFPD), "DFPD (11x11)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::AHD), "AHD (Adaptive Homogeneity-Directed)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::AMAZE), "AMAZE (Adaptive Gradient)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::RCD), "RCD (Ratio Corrected 9x9)");
    EXPECT_EQ(algorithm_name(DemosaicAlgorithm::PRISM),
              "PRISM (Polar Ratio Interpolation Spectral Merging)");
}

TEST(DispatchTest, AlgorithmWindowSizes) {
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::SUPER_FAST), 1);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::HQLI), 5);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::MG), 5);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::L7), 7);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::DFPD), 11);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::AHD), 5);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::AMAZE), 5);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::RCD), 9);
    EXPECT_EQ(algorithm_window_size(DemosaicAlgorithm::PRISM), 9);
}

TEST(DispatchTest, HardwareConcurrency) {
    int n = compute_hardware_concurrency();
    EXPECT_GT(n, 0);
    EXPECT_LE(n, 512);
}

TEST(DispatchTest, HasAvx2ReturnsBool) {
    bool r = has_avx2();
    EXPECT_TRUE(r == true || r == false);
}

TEST(DispatchTest, HasCudaReturnsBool) {
    bool r = has_cuda();
    EXPECT_TRUE(r == true || r == false);
}

TEST(DispatchTest, SuperFastBasicProcessing) {
    // Create a minimal 4x4 RGGB pattern
    uint8_t raw[16] = {
        100, 50, 100, 50,
         50, 80,  50, 80,
        100, 50, 100, 50,
         50, 80,  50, 80
    };
    uint8_t rgb[48] = {0};

    DemosaicError err = demosaic(raw, rgb, 4, 4, BayerPattern::RGGB,
                                       DemosaicAlgorithm::SUPER_FAST, 8);
    EXPECT_EQ(err, DemosaicError::Ok);

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < 48; i++) {
        if (rgb[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(DispatchTest, CpuOnlyPath) {
    // Use 8x8 image: HQLI needs >= 6x6, SUPER_FAST needs >= 2x2
    uint8_t raw[64] = {
        100, 50, 100, 50, 100, 50, 100, 50,
         50, 80,  50, 80,  50, 80,  50, 80,
        100, 50, 100, 50, 100, 50, 100, 50,
         50, 80,  50, 80,  50, 80,  50, 80,
        100, 50, 100, 50, 100, 50, 100, 50,
         50, 80,  50, 80,  50, 80,  50, 80,
        100, 50, 100, 50, 100, 50, 100, 50,
         50, 80,  50, 80,  50, 80,  50, 80,
    };
    uint8_t rgb[192] = {0};

    DemosaicError err = demosaic_cpu(raw, rgb, 8, 8, BayerPattern::RGGB,
                                           DemosaicAlgorithm::HQLI, 8);
    EXPECT_EQ(err, DemosaicError::Ok);

    bool has_nonzero = false;
    for (int i = 0; i < 192; i++) {
        if (rgb[i] != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(DispatchTest, PackedFormatFallsBackGracefully) {
    // Packed 10-bit: 4x4 RGGB, each pixel is 0x100 (256)
    // 4x4 = 16 pixels, 10-bit packed = ceil(16*10/8) = 20 bytes
    uint8_t raw[20] = {0};
    // Fill with 0x100 pattern in 10-bit packing
    // 4 pixels per group, 5 bytes per group
    // pixel0: byte0=0x00, byte4&3=0x01 -> 0x100
    for (int i = 0; i < 5; i++) {
        raw[i * 5 + 0] = 0x00;  // pixel bottom 8 bits = 0x00
        raw[i * 5 + 1] = 0x00;
        raw[i * 5 + 2] = 0x00;
        raw[i * 5 + 3] = 0x00;
        raw[i * 5 + 4] = 0x55;  // all 4 pixels top 2 bits = 0b01 each
    }
    uint8_t rgb[48] = {0};

    // demosaic should auto-fallback from CUDA to CPU for packed
    DemosaicError err = demosaic(raw, rgb, 4, 4, BayerPattern::RGGB,
                                       DemosaicAlgorithm::SUPER_FAST, 10, true);
    EXPECT_EQ(err, DemosaicError::Ok);
}
