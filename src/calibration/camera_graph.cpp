#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace calibration {

// ============================================================
//  Camera Graph: build visibility/connectivity structure
//
//  For each camera pair, count how many checkerboard corners they
//  both observe. This builds a connectivity graph for incremental
//  SfM and provides initial estimates for relative poses.
// ============================================================

CalibrationError process_camera_graph(const SfMView* views, int view_count,
                                       const CheckerboardConfig* config,
                                       CameraGraph* graph) {
    if (!views || !graph || !config) return CalibrationError::NullInput;
    if (view_count < 2) return CalibrationError::InsufficientObservations;

    int n = view_count;
    int cols = config->cols, rows = config->rows;
    int total_pts = cols * rows;

    // For each camera, get its corner observations
    std::vector<const CheckerboardCorners*> cviews(n);
    for (int i = 0; i < n; i++) cviews[i] = views[i].corners;

    // Count pairwise common observations
    // For checkerboard-based calibration, each view is observing the SAME board
    // So all cameras seeing board pose j share all cols*rows points
    // Build visibility: which cameras see which 3D points
    // In the checkerboard case, point_idx = frame * (cols*rows) + row*cols+col

    // Simplified: group views by board pose index
    // For N views, typically each view is a different board position
    // Camera pair sharing = sum over shared board positions × points_per_board

    graph->total_points = total_pts * n; // each view's corners = separate 3D instance
    graph->camera_count = n;

    // Count common observations per pair
    std::vector<CameraPair> pairs;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            // For checkerboard: if cameras i,j both see same board pose, all corners match
            int common = 0;
            // In the general case: cameras see different board positions
            // Compute shared via point matching (simplified: use same position = same view idx)
            if (cviews[i] && cviews[j]) {
                // Compare first corner position to determine if same view
                double dx = cviews[i]->points[0].x - cviews[j]->points[0].x;
                double dy = cviews[i]->points[0].y - cviews[j]->points[0].y;
                double dist = std::sqrt(dx*dx + dy*dy);
                if (dist < 100.0) common = total_pts; // same view → all points match
            }
            if (common > 0) {
                pairs.push_back({i, j, common, static_cast<double>(common)});
            }
        }
    }

    // Sort by common points descending
    std::sort(pairs.begin(), pairs.end(),
              [](const CameraPair& a, const CameraPair& b) { return a.common_pts > b.common_pts; });

    graph->pair_count = static_cast<int>(pairs.size());
    if (pairs.empty()) return CalibrationError::InsufficientObservations;

    graph->pairs = new CameraPair[pairs.size()];
    for (size_t i = 0; i < pairs.size(); i++) graph->pairs[i] = pairs[i];

    // Build visibility matrix: each camera sees its own checkerboard points.
    // Cameras viewing the same board position also share those points.
    graph->vis_counts = new int[n]();
    graph->visibility = new int*[n];
    int per_view = total_pts;
    for (int ci = 0; ci < n; ci++) {
        graph->visibility[ci] = new int[graph->total_points]();
        // Own view's points
        int start = ci * per_view;
        int end = start + per_view;
        for (int p = start; p < end && p < graph->total_points; p++) {
            graph->visibility[ci][p] = 1;
        }
        // Points from cameras that observe the same board position
        for (size_t k = 0; k < pairs.size(); k++) {
            int other = -1;
            if (pairs[k].cam_i == ci && pairs[k].common_pts > 0) other = pairs[k].cam_j;
            else if (pairs[k].cam_j == ci && pairs[k].common_pts > 0) other = pairs[k].cam_i;
            if (other >= 0) {
                int o_start = other * per_view;
                int o_end = o_start + per_view;
                for (int p = o_start; p < o_end && p < graph->total_points; p++) {
                    graph->visibility[ci][p] = 1;
                }
            }
        }
        // Count visible points
        int cnt = 0;
        for (int p = 0; p < graph->total_points; p++) {
            if (graph->visibility[ci][p]) cnt++;
        }
        graph->vis_counts[ci] = cnt;
    }

    return CalibrationError::Ok;
}

} // namespace calibration
