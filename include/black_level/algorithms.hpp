#ifndef BLACK_LEVEL_ALGORITHMS_HPP
#define BLACK_LEVEL_ALGORITHMS_HPP

#include "types.hpp"
#include "imop/types.hpp" // for BayerPattern
#include <cstdint>
#include <string>

namespace black_level {

using imop::BayerPattern;

std::string algorithm_name(BlackLevelAlgorithm algo);
int algorithm_window_size(BlackLevelAlgorithm algo);

BlackLevelError validate_black_level_inputs(const uint8_t* data,
                                             int width, int height,
                                             int channels, int bit_depth);

BlackLevelError process_black_level(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     BlackLevelAlgorithm algorithm,
                                     int bit_depth,
                                     const BlackLevelParams& params);

BlackLevelError process_per_channel(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const BlackLevelParams& params);

BlackLevelError process_global(uint8_t* data,
                                int width, int height,
                                BayerPattern pattern,
                                int bit_depth,
                                const BlackLevelParams& params);

// CUDA support
bool has_cuda();
BlackLevelError process_black_level_cuda(uint8_t* data,
                                          int width, int height,
                                          BayerPattern pattern,
                                          BlackLevelAlgorithm algorithm,
                                          int bit_depth,
                                          const BlackLevelParams& params);

} // namespace black_level

#endif
