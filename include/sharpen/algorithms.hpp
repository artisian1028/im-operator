#ifndef SHARPEN_ALGORITHMS_HPP
#define SHARPEN_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace sharpen {

// Metadata helpers
std::string algorithm_name(SharpenAlgorithm algo);
int algorithm_window_size(SharpenAlgorithm algo);

// Input validation
SharpenError validate_sharpen_inputs(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth);

// Main dispatch: input RGB (channels=3) -> output RGB (channels=3)
// params: amount/radius/threshold for fine-tuning
SharpenError process_sharpen(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              SharpenAlgorithm algorithm, int bit_depth = 8,
                              const SharpenParams& params = {});

// Individual algorithm functions
SharpenError process_unsharp_mask(const uint8_t* input, uint8_t* output,
                                   int width, int height, int channels,
                                   int bit_depth, const SharpenParams& params);

SharpenError process_laplacian(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const SharpenParams& params);

SharpenError process_high_pass(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, const SharpenParams& params);

SharpenError process_adaptive(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const SharpenParams& params);

} // namespace sharpen

#endif // SHARPEN_ALGORITHMS_HPP
