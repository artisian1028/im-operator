#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace calibration {

// ============================================================
//  Checkerboard corner detection with sub-pixel refinement
//
//  Based on the "Saddle Point" approach:
//  1. Compute corner response: product of two Hessian eigenvalues
//  2. Non-maximum suppression
//  3. Find quad patterns
//  4. Connect quads into grid
//  5. Sub-pixel refinement via iterative gradient descent
// ============================================================

namespace {

// Simple grayscale conversion
void to_gray(const uint8_t* rgb, int w, int h, int ch, int bd,
             std::vector<uint8_t>& gray) {
    gray.resize(static_cast<size_t>(w) * h);
    if (ch == 1 && bd <= 8) {
        std::memcpy(gray.data(), rgb, gray.size());
        return;
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int r, g, b;
            size_t idx;
            if (bd <= 8) {
                idx = (static_cast<size_t>(y) * w + x) * 3;
                r = rgb[idx]; g = rgb[idx+1]; b = rgb[idx+2];
            } else {
                const auto* d16 = reinterpret_cast<const uint16_t*>(rgb);
                idx = (static_cast<size_t>(y) * w + x) * 3;
                r = d16[idx] >> 8; g = d16[idx+1] >> 8; b = d16[idx+2] >> 8;
            }
            gray[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>((r*77 + g*150 + b*29) >> 8);
        }
    }
}

// Corner response: 2x2 gradient covariance matrix eigenvalues
// corner_score = det(M) - k * trace(M)^2  (Harris measure)
// For checkerboard, we want dark-light transitions in all directions
void compute_corner_scores(const std::vector<uint8_t>& gray, int w, int h,
                           std::vector<float>& scores) {
    scores.assign(static_cast<size_t>(w) * h, 0.0f);

    for (int y = 3; y < h - 3; y++) {
        for (int x = 3; x < w - 3; x++) {
            float ix = 0, iy = 0, ixx = 0, iyy = 0, ixy = 0;

            // 5x5 integration window
            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    // Simple gradient: central difference
                    int xp1 = x + dx + 1, xm1 = x + dx - 1;
                    int yp1 = y + dy + 1, ym1 = y + dy - 1;
                    float gx = static_cast<float>(gray[static_cast<size_t>(yp1)*w+x+dx]) -
                               static_cast<float>(gray[static_cast<size_t>(ym1)*w+x+dx]);
                    float gy = static_cast<float>(gray[static_cast<size_t>(y+dy)*w+xp1]) -
                               static_cast<float>(gray[static_cast<size_t>(y+dy)*w+xm1]);
                    ix += gx; iy += gy;
                    ixx += gx * gx; iyy += gy * gy; ixy += gx * gy;
                }
            }
            float det = ixx * iyy - ixy * ixy;
            float trace = ixx + iyy + 1e-6f;
            scores[static_cast<size_t>(y) * w + x] = det / (trace * trace + 1e-10f);
        }
    }
}

// Non-maximum suppression + threshold to get corner candidates
void find_candidates(const std::vector<float>& scores, int w, int h,
                     float threshold, int margin,
                     std::vector<int>& cx, std::vector<int>& cy) {
    cx.clear(); cy.clear();
    for (int y = margin; y < h - margin; y++) {
        for (int x = margin; x < w - margin; x++) {
            float s = scores[static_cast<size_t>(y) * w + x];
            if (s < threshold) continue;

            // Check if local maximum in 5x5
            bool is_max = true;
            for (int dy = -2; dy <= 2 && is_max; dy++)
                for (int dx = -2; dx <= 2 && is_max; dx++)
                    if (scores[static_cast<size_t>(y+dy)*w+(x+dx)] > s) is_max = false;

            if (is_max) { cx.push_back(x); cy.push_back(y); }
        }
    }
}

// Find a quadrilateral (4 corners forming a checkerboard cell)
struct Quad {
    int corners[4][2]; // 4 points: [x,y]
    bool valid = false;
};

bool find_quads(const std::vector<int>& cx, const std::vector<int>& cy,
                std::vector<Quad>& quads) {
    quads.clear();
    size_t n = cx.size();
    if (n < 4) return false;

    // Try each corner as potential top-left of a quad
    for (size_t i = 0; i < n; i++) {
        // Find closest candidate in each of 4 quadrants relative to (cx[i], cy[i])
        struct { int x, y; double dist; int idx; } best[4] = {
            {cx[i], cy[i], 1e30, -1}, // top-left
            {cx[i], cy[i], 1e30, -1}, // top-right
            {cx[i], cy[i], 1e30, -1}, // bottom-left
            {cx[i], cy[i], 1e30, -1}  // bottom-right
        };

        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            int dx = cx[j] - cx[i];
            int dy = cy[j] - cy[i];
            int q;
            if (dx < 0 && dy < 0) q = 0;       // top-left quadrant
            else if (dx > 0 && dy < 0) q = 1;  // top-right
            else if (dx < 0 && dy > 0) q = 2;  // bottom-left
            else if (dx > 0 && dy > 0) q = 3;  // bottom-right
            else continue;

            double d = std::sqrt(static_cast<double>(dx*dx + dy*dy));
            if (d < best[q].dist) { best[q].dist = d; best[q].idx = static_cast<int>(j); best[q].x = cx[j]; best[q].y = cy[j]; }
        }

        // All 4 quadrants must have a neighbor
        if (best[0].idx < 0 || best[1].idx < 0 || best[2].idx < 0 || best[3].idx < 0) continue;

        // Check rectangular shape: opposite corners should have similar distances
        double d01 = std::sqrt(std::pow(best[0].x-best[1].x,2.0) + std::pow(best[0].y-best[1].y,2.0));
        double d23 = std::sqrt(std::pow(best[2].x-best[3].x,2.0) + std::pow(best[2].y-best[3].y,2.0));
        double d02 = std::sqrt(std::pow(best[0].x-best[2].x,2.0) + std::pow(best[0].y-best[2].y,2.0));
        double d13 = std::sqrt(std::pow(best[1].x-best[3].x,2.0) + std::pow(best[1].y-best[3].y,2.0));

        double ratio1 = std::min(d01,d23)/std::max(d01,d23);
        double ratio2 = std::min(d02,d13)/std::max(d02,d13);
        if (ratio1 < 0.3 || ratio2 < 0.3) continue;

        Quad q;
        q.corners[0][0] = best[0].x; q.corners[0][1] = best[0].y;
        q.corners[1][0] = best[1].x; q.corners[1][1] = best[1].y;
        q.corners[2][0] = best[3].x; q.corners[2][1] = best[3].y; // bottom-right
        q.corners[3][0] = best[2].x; q.corners[3][1] = best[2].y; // bottom-left
        q.valid = true;
        quads.push_back(q);
    }

    return !quads.empty();
}

// Sub-pixel corner refinement
// For each corner, iteratively shift toward the saddle point using
// gradient-based method: the direction from current position to true
// corner minimizes the dot product of gradient and displacement.
void refine_sub_pixel(const std::vector<uint8_t>& gray, int w, int h,
                      int half_win, int max_iters, double eps,
                      std::vector<double>& pts_x, std::vector<double>& pts_y) {
    size_t n = pts_x.size();
    for (size_t i = 0; i < n; i++) {
        double cx = pts_x[i], cy = pts_y[i];

        for (int iter = 0; iter < max_iters; iter++) {
            double a = 0, b = 0, c = 0, bb1 = 0, bb2 = 0;
            int ix = static_cast<int>(cx + 0.5);
            int iy = static_cast<int>(cy + 0.5);

            for (int dy = -half_win; dy <= half_win; dy++) {
                for (int dx = -half_win; dx <= half_win; dx++) {
                    int px = ix + dx, py = iy + dy;
                    if (px < 1 || px >= w-1 || py < 1 || py >= h-1) continue;
                    double gx = (gray[static_cast<size_t>(py)*w+(px+1)] -
                                 gray[static_cast<size_t>(py)*w+(px-1)]) * 0.5;
                    double gy = (gray[static_cast<size_t>(py+1)*w+px] -
                                 gray[static_cast<size_t>(py-1)*w+px]) * 0.5;
                    double g = gx*gx + gy*gy;
                    if (g < 1e-6) continue;
                    double w2 = 1.0 / std::sqrt(g + 1.0);
                    a += gx * gx * w2;
                    b += gx * gy * w2;
                    c += gy * gy * w2;
                    double val = gray[static_cast<size_t>(py)*w+px] -
                                 (gx*(cx-px) + gy*(cy-py));
                    bb1 += gx * val * w2;
                    bb2 += gy * val * w2;
                }
            }

            double det = a*c - b*b;
            if (std::abs(det) < 1e-10) break;
            double dx = (c*bb1 - b*bb2) / det;
            double dy = (a*bb2 - b*bb1) / det;

            cx += dx; cy += dy;
            if (dx*dx + dy*dy < eps*eps) break;
        }
        pts_x[i] = cx; pts_y[i] = cy;
    }
}

} // anonymous namespace

// ============================================================
//  Public API
// ============================================================

CalibrationError process_checkerboard_detect(const uint8_t* image,
                                              int width, int height,
                                              int channels, int bit_depth,
                                              const CheckerboardConfig* config,
                                              CheckerboardCorners* corners) {
    if (!image || !config || !corners) return CalibrationError::NullInput;
    if (width < 20 || height < 20) return CalibrationError::InsufficientObservations;

    int cols = config->cols, rows = config->rows;
    if (cols < 2 || rows < 2) return CalibrationError::InsufficientObservations;

    // Convert to grayscale
    std::vector<uint8_t> gray;
    to_gray(image, width, height, channels, bit_depth, gray);

    // Compute corner scores
    std::vector<float> scores;
    compute_corner_scores(gray, width, height, scores);

    // Find candidates
    float threshold = 0.01f;
    std::vector<int> cx, cy;
    find_candidates(scores, width, height, threshold, 5, cx, cy);

    // If too few candidates, lower threshold
    int target = cols * rows + 10;
    while (static_cast<int>(cx.size()) < target && threshold > 0.0001f) {
        threshold *= 0.3f;
        find_candidates(scores, width, height, threshold, 5, cx, cy);
    }

    if (static_cast<int>(cx.size()) < cols * rows)
        return CalibrationError::InsufficientObservations;

    // Find quads
    std::vector<Quad> quads;
    if (!find_quads(cx, cy, quads))
        return CalibrationError::InsufficientObservations;

    // Extract corners from the best quad grid
    // Use the first valid quad's corners as initial estimates and spread into grid
    std::vector<double> pts_x(cols * rows), pts_y(cols * rows);
    bool found_grid = false;

    // Try each quad as a seed for the grid
    for (const auto& q : quads) {
        if (!q.valid) continue;

        // Estimate grid spacing and orientation from this quad
        double dx_r = (q.corners[1][0] - q.corners[0][0]) / (cols - 1.0);
        double dy_r = (q.corners[1][1] - q.corners[0][1]) / (cols - 1.0);
        double dx_d = (q.corners[3][0] - q.corners[0][0]) / (rows - 1.0);
        double dy_d = (q.corners[3][1] - q.corners[0][1]) / (rows - 1.0);

        // Project all expected corners
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                double ex = q.corners[0][0] + c * dx_r + r * dx_d;
                double ey = q.corners[0][1] + c * dy_r + r * dy_d;

                // Snap to nearest candidate
                double best_d = 1e30;
                for (size_t k = 0; k < cx.size(); k++) {
                    double d = std::pow(cx[k]-ex,2.0) + std::pow(cy[k]-ey,2.0);
                    if (d < best_d) { best_d = d; pts_x[idx] = cx[k]; pts_y[idx] = cy[k]; }
                }
                if (best_d > 100.0) { pts_x[idx] = ex; pts_y[idx] = ey; } // fallback
            }
        }
        found_grid = true;
        break;
    }

    if (!found_grid) return CalibrationError::InsufficientObservations;

    // Sub-pixel refinement
    if (config->sub_pixel) {
        refine_sub_pixel(gray, width, height,
                         config->sub_pixel_window, config->sub_pixel_iters,
                         config->sub_pixel_eps, pts_x, pts_y);
    }

    // Allocate and fill output
    corners->rows = rows;
    corners->cols = cols;
    corners->points = new Point2D[static_cast<size_t>(cols) * rows];
    for (int i = 0; i < cols * rows; i++) {
        corners->points[i].x = pts_x[i];
        corners->points[i].y = pts_y[i];
    }
    corners->valid = true;

    return CalibrationError::Ok;
}

} // namespace calibration
