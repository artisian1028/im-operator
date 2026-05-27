#ifndef IM_OPERATOR_H
#define IM_OPERATOR_H

#include "imop/types.hpp"
#include "imop/pixel_utils.hpp"
#include "imop/analyzer.hpp"
#include "imop/algorithms.hpp"

namespace imop {

class Demosaic final {
public:
    Demosaic() = default;
    ~Demosaic() = default;

    DemosaicError process(const uint8_t* bayer_data, uint8_t* rgb_data,
                         int width, int height, BayerPattern pattern,
                         DemosaicAlgorithm algorithm, int bit_depth = 8,
                         bool is_packed = false) {
        return demosaic(bayer_data, rgb_data, width, height, pattern, algorithm, bit_depth, is_packed);
    }

    DemosaicError process_cpu(const uint8_t* bayer_data, uint8_t* rgb_data,
                             int width, int height, BayerPattern pattern,
                             DemosaicAlgorithm algorithm, int bit_depth = 8,
                             bool is_packed = false) {
        return demosaic_cpu(bayer_data, rgb_data, width, height, pattern, algorithm, bit_depth, is_packed);
    }

    DemosaicError process(const ImageBuffer& bayer, ImageBuffer& rgb,
                         BayerPattern pattern, DemosaicAlgorithm algorithm) {
        if (bayer.empty() || bayer.channels != 1) return DemosaicError::InvalidDimensions;
        rgb.width = bayer.width;
        rgb.height = bayer.height;
        rgb.channels = 3;
        rgb.bit_depth = bayer.bit_depth;
        size_t rgb_size = static_cast<size_t>(bayer.width) * bayer.height * 3;
        if (bayer.bit_depth > 8) rgb_size *= 2;
        rgb.data.resize(rgb_size);
        return process(bayer.ptr(), rgb.ptr(), bayer.width, bayer.height, pattern, algorithm, bayer.bit_depth, bayer.is_packed);
    }

    static std::string algorithm_name(DemosaicAlgorithm algo) {
        return imop::algorithm_name(algo);
    }

    static int algorithm_window_size(DemosaicAlgorithm algo) {
        return imop::algorithm_window_size(algo);
    }

    static DataInfo analyze_data(const uint8_t* data, size_t byte_size) {
        return imop::analyze_data(data, byte_size);
    }

    static int detect_bit_depth(const uint8_t* data, size_t byte_size) {
        return imop::detect_bit_depth(data, byte_size);
    }

    static std::vector<std::pair<int, int>> suggest_dimensions(size_t pixel_count) {
        return imop::suggest_dimensions(pixel_count);
    }

    static BayerPattern guess_pattern(const uint8_t* data, int width, int height,
                                        int bit_depth = 8, bool is_packed = false) {
        return imop::guess_pattern(data, width, height, bit_depth, is_packed);
    }

    static int compute_hardware_concurrency() {
        return imop::compute_hardware_concurrency();
    }

    static void cuda_synchronize() {
        imop::cuda_synchronize();
    }
};

} // namespace imop

#endif // IM_OPERATOR_H
