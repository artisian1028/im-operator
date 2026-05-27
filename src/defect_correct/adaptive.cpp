#include "common.hpp"
#include "defect_correct/algorithms.hpp"
#include <vector>
#include <algorithm>

namespace defect_correct {

DefectCorrectError process_adaptive(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const DefectCorrectParams& params) {
    if (width < 5 || height < 5) return DefectCorrectError::ImageTooSmall;

    auto po = imop::PatternOffsets::from_pattern(pattern);
    int max_val = detail::safe_max_val(bit_depth);
    float threshold = std::max(0.05f, std::min(1.0f, params.threshold));
    int abs_thresh = static_cast<int>(threshold * static_cast<float>(max_val));

    // Process: detect and fix in place
    // For each pixel, compare to same-color neighbors at distance 2 (skip different colors)
    std::vector<int> neighbors;
    neighbors.reserve(12);

    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int center = detail::read_bayer(data, x, y, width, bit_depth);
            int color = detail::bayer_color(y, x, po);

            // Collect same-color neighbors in 5x5 window (skip=2 for Bayer)
            neighbors.clear();
            for (int dy = -2; dy <= 2; dy += 2) {
                for (int dx = -2; dx <= 2; dx += 2) {
                    if (dx == 0 && dy == 0) continue;
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 2 && nx < width - 2 && ny >= 2 && ny < height - 2) {
                        int nc = detail::bayer_color(ny, nx, po);
                        if (nc == color) {
                            neighbors.push_back(detail::read_bayer(data, nx, ny, width, bit_depth));
                        }
                    }
                }
            }

            if (neighbors.empty()) continue;

            // Compute median
            std::sort(neighbors.begin(), neighbors.end());
            int median = neighbors[neighbors.size() / 2];

            // If center deviates too much from median, replace with median
            int diff = center - median;
            if (diff < 0) diff = -diff;
            if (diff > abs_thresh) {
                detail::write_bayer(data, x, y, width, bit_depth, median);
            }
        }
    }

    return DefectCorrectError::Ok;
}

} // namespace defect_correct
