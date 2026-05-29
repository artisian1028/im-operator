#ifndef LENS_SHADING_ALGORITHMS_HPP
#define LENS_SHADING_ALGORITHMS_HPP

#include "types.hpp"
#include "imop/types.hpp"
#include <cstdint>
#include <string>

namespace lens_shading {

using imop::BayerPattern;

std::string algorithm_name(LensShadingAlgorithm algo);
int algorithm_window_size(LensShadingAlgorithm algo);

LensShadingError validate_lens_shading_inputs(const uint8_t* data,
                                               int width, int height,
                                               int channels, int bit_depth);

LensShadingError process_lens_shading(uint8_t* data,
                                       int width, int height,
                                       BayerPattern pattern,
                                       LensShadingAlgorithm algorithm,
                                       int bit_depth,
                                       const LensShadingParams& params);

LensShadingError process_polynomial(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const LensShadingParams& params);

LensShadingError process_flat_field(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const LensShadingParams& params);

// CUDA support
bool has_cuda();
LensShadingError process_lens_shading_cuda(uint8_t* data,
                                            int width, int height,
                                            BayerPattern pattern,
                                            LensShadingAlgorithm algorithm,
                                            int bit_depth,
                                            const LensShadingParams& params);

} // namespace lens_shading

#endif
