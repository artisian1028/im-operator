#include "common.hpp"

namespace hdr {

// ============================================================
//  HDR Transfer Functions — float32 + int paths
// ============================================================

namespace {
constexpr float pq_m1 = 0.1593017578125f;
constexpr float pq_m2 = 78.84375f;
constexpr float pq_c1 = 0.8359375f;
constexpr float pq_c2 = 18.8515625f;
constexpr float pq_c3 = 18.6875f;
constexpr float pq_n = 1.0f / pq_m2;
constexpr float pq_m1_inv = 1.0f / pq_m1;

constexpr float hlg_a = 0.17883277f;
constexpr float hlg_b_val = 1.0f - 4.0f * hlg_a;
const float hlg_c_val = 0.5f - hlg_a * std::log(4.0f * hlg_a);
}

// Per-channel processing helper: reads float, applies transfer, writes float/int
namespace {

template<typename Fn>
HdrError process_transfer_per_channel(const uint8_t* input, uint8_t* output,
                                       int width, int height, int channels,
                                       int bit_depth, const HdrParams& params, Fn transfer) {
    HdrError err = validate_hdr_inputs(input, output, width, height, channels, bit_depth);
    if (err != HdrError::Ok) return err;

    float exposure_mul = std::pow(2.0f, std::max(-8.0f, std::min(8.0f, params.exposure)));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float val = detail::read_pixel_f(input, x, y, width, channels, bit_depth, c);
                val *= exposure_mul;
                float result = transfer(val);
                if (bit_depth == 0) result = detail::clamp_val_f(result);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, c, result);
            }
        }
    }
    return HdrError::Ok;
}

} // anonymous namespace

HdrError process_linear_to_pq(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const HdrParams& params) {
    return process_transfer_per_channel(input, output, width, height, channels, bit_depth, params,
        [](float val) {
            float Lp = detail::safe_pow(val, pq_m1);
            float pq = detail::safe_pow((pq_c1 + pq_c2 * Lp) / (1.0f + pq_c3 * Lp), pq_m2);
            return std::max(0.0f, std::min(1.0f, pq));
        });
}

HdrError process_pq_to_linear(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const HdrParams& params) {
    return process_transfer_per_channel(input, output, width, height, channels, bit_depth, params,
        [](float val) {
            float vp = detail::safe_pow(val, pq_n);
            float linear = detail::safe_pow(
                std::max(0.0f, vp - pq_c1) / std::max(1e-10f, pq_c2 - pq_c3 * vp), pq_m1_inv);
            return std::max(0.0f, linear);
        });
}

HdrError process_linear_to_hlg(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const HdrParams& params) {
    return process_transfer_per_channel(input, output, width, height, channels, bit_depth, params,
        [](float E) {
            if (E <= 1.0f / 12.0f) return std::sqrt(3.0f * E);
            return std::max(0.0f, std::min(1.0f, hlg_a * std::log(12.0f * E - hlg_b_val) + hlg_c_val));
        });
}

HdrError process_hlg_to_linear(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const HdrParams& params) {
    return process_transfer_per_channel(input, output, width, height, channels, bit_depth, params,
        [](float sig) {
            if (sig <= 0.5f) return sig * sig / 3.0f;
            return std::max(0.0f, (std::exp((sig - hlg_c_val) / hlg_a) + hlg_b_val) / 12.0f);
        });
}

} // namespace hdr
