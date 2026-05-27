#include "common.hpp"
#include "optimized.hpp"
#include "interp_core.hpp"

namespace imop {

DemosaicError process_hqli(const uint8_t* bayer, uint8_t* rgb,
                  int width, int height, BayerPattern pattern, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer, rgb, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (width < 6 || height < 6) return DemosaicError::ImageTooSmall;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);

    // AVX2 fast path for 8-bit unpacked
    if (bit_depth <= 8 && !is_packed && has_avx2()) {
        process_hqli_optimized(bayer, rgb, width, height, po, bit_depth, is_packed);
        detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::HQLI) / 2);
        return DemosaicError::Ok;
    }

    const int max_val = pixel::safe_max_val(bit_depth);

    if (bit_depth <= 8 && !is_packed) {
        // 8-bit unpacked: direct indexing
        auto pix = detail::Accessor8bit{bayer, width, height};
        auto write = detail::make_writer_8bit(rgb, width);
        detail::hqli_core(width, height, po, max_val, pix, write);
    } else if (is_packed) {
        // Packed data: use clamped accessor
        auto pix = detail::AccessorPacked{bayer, width, height, bit_depth};
        auto write = detail::make_writer_generic(rgb, width, bit_depth);
        detail::hqli_core(width, height, po, max_val, pix, write);
    } else {
        // 16-bit unpacked
        auto pix = detail::Accessor16bit{bayer, width, bit_depth};
        auto write = detail::make_writer_generic(rgb, width, bit_depth);
        detail::hqli_core(width, height, po, max_val, pix, write);
    }

    detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::HQLI) / 2);

    return DemosaicError::Ok;
}

} // namespace imop
