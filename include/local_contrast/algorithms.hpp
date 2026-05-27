#ifndef LOCAL_CONTRAST_ALGORITHMS_HPP
#define LOCAL_CONTRAST_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace local_contrast {

std::string algorithm_name(LocalContrastAlgorithm algo);
int algorithm_window_size(LocalContrastAlgorithm algo);

LocalContrastError validate_local_contrast_inputs(const uint8_t* input, uint8_t* output,
                                                   int width, int height,
                                                   int channels, int bit_depth);

LocalContrastError process_local_contrast(const uint8_t* input, uint8_t* output,
                                           int width, int height, int channels,
                                           LocalContrastAlgorithm algorithm,
                                           int bit_depth,
                                           const LocalContrastParams& params);

LocalContrastError process_unsharp(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    int bit_depth, const LocalContrastParams& params);

LocalContrastError process_bilateral(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, const LocalContrastParams& params);

} // namespace local_contrast

#endif
