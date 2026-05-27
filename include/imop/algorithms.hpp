#ifndef IMOP_ALGORITHMS_HPP
#define IMOP_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace imop {

std::string algorithm_name(DemosaicAlgorithm algo);

int algorithm_window_size(DemosaicAlgorithm algo);

int compute_hardware_concurrency();

DemosaicError validate_demosaic_inputs(const uint8_t* bayer_data, uint8_t* rgb_data,
                                     int width, int height, int bit_depth);

DemosaicError demosaic(const uint8_t* bayer_data, uint8_t* rgb_data,
                             int width, int height, BayerPattern pattern,
                             DemosaicAlgorithm algorithm, int bit_depth = 8,
                             bool is_packed = false);

DemosaicError demosaic_cpu(const uint8_t* bayer_data, uint8_t* rgb_data,
                                 int width, int height, BayerPattern pattern,
                                 DemosaicAlgorithm algorithm, int bit_depth = 8,
                                 bool is_packed = false);

DemosaicError process_hqli(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, BayerPattern pattern, int bit_depth,
                          bool is_packed = false);

DemosaicError process_mg(const uint8_t* bayer, uint8_t* rgb,
                        int width, int height, BayerPattern pattern, int bit_depth,
                        bool is_packed = false);

DemosaicError process_l7(const uint8_t* bayer, uint8_t* rgb,
                        int width, int height, BayerPattern pattern, int bit_depth,
                        bool is_packed = false);

DemosaicError process_dfpd(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, BayerPattern pattern, int bit_depth,
                          bool is_packed = false);

DemosaicError process_ahd(const uint8_t* bayer, uint8_t* rgb,
                         int width, int height, BayerPattern pattern, int bit_depth,
                         bool is_packed = false);

DemosaicError process_amaze(const uint8_t* bayer, uint8_t* rgb,
                           int width, int height, BayerPattern pattern, int bit_depth,
                           bool is_packed = false);

DemosaicError process_rcd(const uint8_t* bayer, uint8_t* rgb,
                         int width, int height, BayerPattern pattern, int bit_depth,
                         bool is_packed = false);

DemosaicError process_prism(const uint8_t* bayer, uint8_t* rgb,
                           int width, int height, BayerPattern pattern, int bit_depth,
                           bool is_packed = false);

DemosaicError process_super_fast(const uint8_t* bayer, uint8_t* rgb,
                                int width, int height, BayerPattern pattern, int bit_depth,
                                bool is_packed = false);

bool has_avx2();

bool has_cuda();

const char* cuda_device_name();

void cuda_synchronize();

DemosaicError demosaic_cuda(const uint8_t* bayer_data, uint8_t* rgb_data,
                                  int width, int height, BayerPattern pattern,
                                  DemosaicAlgorithm algorithm, int bit_depth = 8,
                                  bool is_packed = false);

DemosaicError demosaic_cuda_batch(const uint8_t* const* bayer_data_array,
                                         uint8_t* const* rgb_data_array,
                                         int num_frames,
                                         int width, int height,
                                         BayerPattern pattern,
                                         DemosaicAlgorithm algorithm,
                                         int bit_depth = 8);

} // namespace imop

#endif // IMOP_ALGORITHMS_HPP
