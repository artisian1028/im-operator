#include "imop/algorithms.hpp"
#include "imop/types.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

int main() {
    printf("=== Demosaic Example ===\n");

    const int width = 1920, height = 1080;
    const int bayer_size = width * height;
    const int rgb_size = width * height * 3;

    // Load bayer data
    std::vector<uint8_t> bayer_data(bayer_size);
    FILE* f = fopen("test_data/bayer_rggb_1920x1080_8bit.raw", "rb");
    if (!f) {
        printf("ERROR: Cannot open test_data/bayer_rggb_1920x1080_8bit.raw\n");
        return 1;
    }
    if (fread(bayer_data.data(), 1, bayer_size, f) != (size_t)bayer_size) {
        printf("ERROR: Failed to read full bayer data (%d bytes)\n", bayer_size);
        fclose(f);
        return 1;
    }
    fclose(f);
    printf("Loaded bayer data: %dx%d, %d bytes\n", width, height, bayer_size);

    // Allocate output
    std::vector<uint8_t> rgb_data(rgb_size);

    // Demo HQLI
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::HQLI, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("HQLI demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("HQLI demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_output.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_output.rgb\n");
        }
    }

    // Demo MG
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::MG, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("MG demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("MG demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_mg.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_mg.rgb\n");
        }
    }

    // Demo L7
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::L7, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("L7 demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("L7 demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_l7.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_l7.rgb\n");
        }
    }

    // Demo SUPER_FAST
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::SUPER_FAST, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("SUPER_FAST demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("SUPER_FAST demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_super_fast.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_super_fast.rgb\n");
        }
    }

    // Demo DFPD
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::DFPD, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("DFPD demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("DFPD demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_dfpd.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_dfpd.rgb\n");
        }
    }

    // Demo AHD
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::AHD, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("AHD demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("AHD demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_ahd.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_ahd.rgb\n");
        }
    }

    // Demo AMAZE
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::AMAZE, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("AMAZE demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("AMAZE demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_amaze.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_amaze.rgb\n");
        }
    }

    // Demo RCD
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::RCD, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("RCD demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("RCD demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_rcd.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_rcd.rgb\n");
        }
    }

    // Demo PRISM
    {
        auto start = std::chrono::steady_clock::now();
        auto err = imop::demosaic(bayer_data.data(), rgb_data.data(),
                                  width, height, imop::BayerPattern::RGGB,
                                  imop::DemosaicAlgorithm::PRISM, 8, false);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (!err) {
            printf("PRISM demosaic: ERROR - %s\n", imop::demosaic_error_message(err));
        } else {
            printf("PRISM demosaic: OK (%lld ms)\n", (long long)ms);
            FILE* out = fopen("demosaic_prism.rgb", "wb");
            if (out) { fwrite(rgb_data.data(), 1, rgb_size, out); fclose(out); }
            printf("  Saved demosaic_prism.rgb\n");
        }
    }

    printf("Done.\n");
    return 0;
}
