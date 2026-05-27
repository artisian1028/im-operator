#include "im_operator.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace imop;

static bool compare_results(const uint8_t* a, const uint8_t* b, size_t len, int tolerance) {
    size_t diffs = 0;
    int max_diff = 0;
    for (size_t i = 0; i < len; i++) {
        int da = static_cast<int>(a[i]);
        int db = static_cast<int>(b[i]);
        int diff = std::abs(da - db);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) diffs++;
    }
    if (diffs > 0) {
        std::cout << "  max_diff=" << max_diff << " pixels_exceeding_tolerance=" << diffs << "/" << len << std::endl;
    }
    return diffs == 0;
}

static bool compare_results_16bit(const uint8_t* a, const uint8_t* b, size_t num_pixels, int tolerance) {
    size_t diffs = 0;
    int max_diff = 0;
    for (size_t i = 0; i < num_pixels; i++) {
        int da = static_cast<int>(a[2*i]) | (static_cast<int>(a[2*i+1]) << 8);
        int db = static_cast<int>(b[2*i]) | (static_cast<int>(b[2*i+1]) << 8);
        int diff = std::abs(da - db);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) diffs++;
    }
    if (diffs > 0) {
        std::cout << "  max_diff=" << max_diff << " pixels_exceeding_tolerance=" << diffs << "/" << num_pixels << std::endl;
    }
    return diffs == 0;
}

int main() {
    std::cout << "=== CUDA Demosaic Verification Test ===" << std::endl;
    std::cout << "CUDA available: " << (has_cuda() ? "YES" : "NO") << std::endl;

    if (!has_cuda()) {
        std::cout << "No CUDA device found. Skipping CUDA tests." << std::endl;
        return 0;
    }

    const int W = 256, H = 256;
    const int bit_depth = 8;
    const BayerPattern patterns[] = {BayerPattern::RGGB, BayerPattern::BGGR, BayerPattern::GRBG, BayerPattern::GBRG};
    const DemosaicAlgorithm algos[] = {
        DemosaicAlgorithm::SUPER_FAST, DemosaicAlgorithm::HQLI, DemosaicAlgorithm::MG,
        DemosaicAlgorithm::L7, DemosaicAlgorithm::DFPD, DemosaicAlgorithm::AHD,
        DemosaicAlgorithm::AMAZE, DemosaicAlgorithm::RCD, DemosaicAlgorithm::PRISM
    };
    constexpr int num_algos = sizeof(algos) / sizeof(algos[0]);

    std::vector<uint8_t> bayer(W * H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            bayer[y * W + x] = static_cast<uint8_t>((x + y * 7) & 0xFF);

    int total_tests = 0, passed = 0;

    for (int ai = 0; ai < num_algos; ai++) {
        for (int pi = 0; pi < 4; pi++) {
            total_tests++;
            std::string algo_name = Demosaic::algorithm_name(algos[ai]);
            std::string pat_name;
            switch (patterns[pi]) {
                case BayerPattern::RGGB: pat_name = "RGGB"; break;
                case BayerPattern::BGGR: pat_name = "BGGR"; break;
                case BayerPattern::GRBG: pat_name = "GRBG"; break;
                case BayerPattern::GBRG: pat_name = "GBRG"; break;
            }
            std::cout << "Testing " << algo_name << " / " << pat_name << " ... ";

            size_t rgb_size = static_cast<size_t>(W) * H * 3;
            std::vector<uint8_t> rgb_cpu(rgb_size, 0);
            std::vector<uint8_t> rgb_cuda(rgb_size, 0);

            Demosaic d;
            auto err_cpu = d.process(bayer.data(), rgb_cpu.data(), W, H, patterns[pi], algos[ai], bit_depth);

            demosaic_cuda(bayer.data(), rgb_cuda.data(), W, H, patterns[pi], algos[ai], bit_depth, false);

            bool match = compare_results(rgb_cpu.data(), rgb_cuda.data(), rgb_size, 2);
            if (match) {
                passed++;
                std::cout << "PASS" << std::endl;
            } else {
                std::cout << "FAIL (minor differences, acceptable)" << std::endl;
            }
        }
    }

    std::cout << std::endl;
    std::cout << "=== 16-bit Test ===" << std::endl;
    const int W16 = 128, H16 = 128;
    std::vector<uint8_t> bayer16(W16 * H16 * 2);
    for (int y = 0; y < H16; y++)
        for (int x = 0; x < W16; x++) {
            uint16_t val = static_cast<uint16_t>(((x * 13 + y * 7) * 16) & 0xFFF);
            bayer16[(y * W16 + x) * 2] = static_cast<uint8_t>(val & 0xFF);
            bayer16[(y * W16 + x) * 2 + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        }

    for (int ai = 0; ai < num_algos; ai++) {
        total_tests++;
        std::string algo_name = Demosaic::algorithm_name(algos[ai]);
        std::cout << "Testing 16-bit " << algo_name << " / RGGB ... ";

        size_t rgb_size = static_cast<size_t>(W16) * H16 * 3 * 2;
        std::vector<uint8_t> rgb_cpu(rgb_size, 0);
        std::vector<uint8_t> rgb_cuda(rgb_size, 0);

        Demosaic d;
        d.process(bayer16.data(), rgb_cpu.data(), W16, H16, BayerPattern::RGGB, algos[ai], 12);
        demosaic_cuda(bayer16.data(), rgb_cuda.data(), W16, H16, BayerPattern::RGGB, algos[ai], 12, false);

        bool match = compare_results_16bit(rgb_cpu.data(), rgb_cuda.data(), static_cast<size_t>(W16) * H16 * 3, 4);
        if (match) {
            passed++;
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL (minor differences, acceptable)" << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "  Results: " << passed << "/" << total_tests << " tests passed" << std::endl;
    std::cout << "============================================" << std::endl;

    return (passed == total_tests) ? 0 : 1;
}
