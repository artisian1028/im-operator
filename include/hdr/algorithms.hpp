#ifndef HDR_ALGORITHMS_HPP
#define HDR_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace hdr {

// Metadata helpers
std::string algorithm_name(HdrAlgorithm algo);
int algorithm_window_size(HdrAlgorithm algo);

// Input validation
HdrError validate_hdr_inputs(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth);

// Main dispatch: input RGB (channels=3) → output RGB (channels=3)
HdrError process_hdr(const uint8_t* input, uint8_t* output,
                      int width, int height, int channels,
                      HdrAlgorithm algorithm, int bit_depth = 16,
                      const HdrParams& params = {});

// Individual algorithm functions
HdrError process_reinhard(const uint8_t* input, uint8_t* output,
                           int width, int height, int channels,
                           int bit_depth, const HdrParams& params);

HdrError process_reinhard_ext(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const HdrParams& params);

HdrError process_filmic_aces(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, const HdrParams& params);

HdrError process_hable(const uint8_t* input, uint8_t* output,
                        int width, int height, int channels,
                        int bit_depth, const HdrParams& params);

HdrError process_drago(const uint8_t* input, uint8_t* output,
                        int width, int height, int channels,
                        int bit_depth, const HdrParams& params);

HdrError process_adaptive_local(const uint8_t* input, uint8_t* output,
                                 int width, int height, int channels,
                                 int bit_depth, const HdrParams& params);

HdrError process_exponential(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, const HdrParams& params);

HdrError process_logarithmic(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, const HdrParams& params);

HdrError process_linear_to_pq(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const HdrParams& params);

HdrError process_pq_to_linear(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const HdrParams& params);

HdrError process_linear_to_hlg(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const HdrParams& params);

HdrError process_hlg_to_linear(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const HdrParams& params);

// CUDA support (defined in cuda_dispatch.cpp when IM_OPERATOR_HAS_CUDA is set)
bool has_cuda();
HdrError process_hdr_cuda(const uint8_t* input, uint8_t* output,
                           int width, int height, int channels,
                           HdrAlgorithm algorithm, int bit_depth,
                           const HdrParams& params);

} // namespace hdr

#endif // HDR_ALGORITHMS_HPP
