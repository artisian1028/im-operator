#include "common.hpp"
#include "defect_correct/algorithms.hpp"
#include <vector>
#include <algorithm>

namespace defect_correct {

DefectCorrectError process_map_based(uint8_t* data,
                                      int width, int height,
                                      BayerPattern pattern,
                                      int bit_depth,
                                      const DefectCorrectParams& params) {
    if (!params.map || params.map_count <= 0) return DefectCorrectError::Ok;

    auto po = imop::PatternOffsets::from_pattern(pattern);

    for (int i = 0; i < params.map_count; i++) {
        int x = params.map[i].x;
        int y = params.map[i].y;
        if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) continue;

        int color = detail::bayer_color(y, x, po);

        // Collect same-color neighbors within 5x5 (skip=2)
        std::vector<int> neighbors;
        for (int dy = -2; dy <= 2; dy += 2) {
            for (int dx = -2; dx <= 2; dx += 2) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx, ny = y + dy;
                if (nx >= 2 && nx < width - 2 && ny >= 2 && ny < height - 2) {
                    if (detail::bayer_color(ny, nx, po) == color) {
                        neighbors.push_back(detail::read_bayer(data, nx, ny, width, bit_depth));
                    }
                }
            }
        }

        if (!neighbors.empty()) {
            std::sort(neighbors.begin(), neighbors.end());
            int median = neighbors[neighbors.size() / 2];
            detail::write_bayer(data, x, y, width, bit_depth, median);
        }
    }

    return DefectCorrectError::Ok;
}

} // namespace defect_correct
