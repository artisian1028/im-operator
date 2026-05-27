#ifndef HIGHLIGHT_RECONSTRUCT_ALGORITHMS_HPP
#define HIGHLIGHT_RECONSTRUCT_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace highlight_reconstruct {

std::string algorithm_name(HighlightReconstructAlgorithm algo);
int algorithm_window_size(HighlightReconstructAlgorithm algo);

HighlightReconstructError validate_highlight_reconstruct_inputs(const uint8_t* input,
                                                                  uint8_t* output,
                                                                  int width, int height,
                                                                  int channels, int bit_depth);

HighlightReconstructError process_highlight_reconstruct(const uint8_t* input, uint8_t* output,
                                                         int width, int height, int channels,
                                                         HighlightReconstructAlgorithm algorithm,
                                                         int bit_depth,
                                                         const HighlightReconstructParams& params);

HighlightReconstructError process_channel_guided(const uint8_t* input, uint8_t* output,
                                                  int width, int height, int channels,
                                                  int bit_depth,
                                                  const HighlightReconstructParams& params);

HighlightReconstructError process_gradient_based(const uint8_t* input, uint8_t* output,
                                                  int width, int height, int channels,
                                                  int bit_depth,
                                                  const HighlightReconstructParams& params);

} // namespace highlight_reconstruct

#endif
