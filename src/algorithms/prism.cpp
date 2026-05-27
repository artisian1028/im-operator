#include "common.hpp"
#include <cmath>
#include <algorithm>

namespace imop {

namespace prism_detail {

inline float median_3x3(float* w) {
#define CSWAP(a, b) do { float _t = w[a]; if (_t > w[b]) { w[a] = w[b]; w[b] = _t; } } while(0)
    CSWAP(0,1); CSWAP(3,4); CSWAP(6,7);
    CSWAP(1,2); CSWAP(4,5); CSWAP(7,8);
    CSWAP(0,1); CSWAP(3,4); CSWAP(6,7);
    CSWAP(0,3); CSWAP(4,7); CSWAP(1,4);
    CSWAP(3,6); CSWAP(2,5); CSWAP(5,8);
    CSWAP(1,3); CSWAP(4,6); CSWAP(2,5);
    CSWAP(2,3); CSWAP(5,6); CSWAP(2,4);
    CSWAP(3,5); CSWAP(3,4);
    return w[4];
#undef CSWAP
}

}

DemosaicError process_prism(const uint8_t* bayer, uint8_t* rgb,
                           int width, int height, BayerPattern pattern, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer, rgb, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (width < 10 || height < 10) return DemosaicError::ImageTooSmall;

    using namespace pixel;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    int max_val = safe_max_val(bit_depth);
    float scale = static_cast<float>(max_val);
    size_t total = static_cast<size_t>(width) * height;

    auto fc = [&](int r, int c) -> int {
        if ((r & 1) == po.r_row && (c & 1) == po.r_col) return 0;
        if ((r & 1) == po.b_row && (c & 1) == po.b_col) return 2;
        return 1;
    };

    auto& pool = detail::thread_local_pool();
    auto& cfa = pool.get(total, 0);
    auto& rgb0 = pool.get(total, 1);
    auto& rgb1 = pool.get(total, 2);
    auto& rgb2 = pool.get(total, 3);
    auto& VH_Dir = pool.get(total, 4);
    auto& lpf = pool.get(total, 5);
    auto& PQ_Dir = pool.get(total, 6);
    auto& L = pool.get(total, 7);
    auto& M = pool.get(total, 8);
    auto& L_tmp = pool.get(total, 9);
    auto& M_tmp = pool.get(total, 10);
    std::fill(cfa.begin(), cfa.end(), 0.0f);
    std::fill(rgb0.begin(), rgb0.end(), 0.0f);
    std::fill(rgb1.begin(), rgb1.end(), 0.0f);
    std::fill(rgb2.begin(), rgb2.end(), 0.0f);
    std::fill(VH_Dir.begin(), VH_Dir.end(), 0.5f);
    std::fill(lpf.begin(), lpf.end(), 0.0f);
    std::fill(PQ_Dir.begin(), PQ_Dir.end(), 0.5f);
    std::fill(L.begin(), L.end(), 0.0f);
    std::fill(M.begin(), M.end(), 0.0f);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            size_t idx = static_cast<size_t>(row) * width + col;
            float val = static_cast<float>(get_clamped(bayer, col, row, width, height, bit_depth, is_packed)) / scale;
            cfa[idx] = val;
            int ch = fc(row, col);
            if (ch == 0) rgb0[idx] = val;
            else if (ch == 1) rgb1[idx] = val;
            else rgb2[idx] = val;
        }
    }

    const int w = width;
    const float eps = 1e-5f;
    const float epssq = 1e-10f;

    DLOOP(row, 4, height - 4)
    {
        for (int col = 4; col < width - 4; col++) {
            size_t indx = static_cast<size_t>(row) * w + col;
            float V_Stat = 0.0f, H_Stat = 0.0f;
            for (int k = -4; k <= 4; k++) {
                float cv = cfa[static_cast<size_t>(row + k) * w + col];
                float ch = cfa[static_cast<size_t>(row) * w + (col + k)];
                V_Stat += std::abs(cfa[indx] - cv);
                H_Stat += std::abs(cfa[indx] - ch);
            }
            V_Stat = std::max(V_Stat, epssq);
            H_Stat = std::max(H_Stat, epssq);
            VH_Dir[indx] = V_Stat / (V_Stat + H_Stat);
        }
    }

    DLOOP(row, 2, height - 2)
    {
        int start_col = 2 + (fc(row, 0) & 1);
        for (int col = start_col; col < width - 2; col += 2) {
            size_t indx = static_cast<size_t>(row) * w + col;
            lpf[indx] = 0.25f * cfa[indx]
                + 0.125f * (cfa[static_cast<size_t>(row-1)*w+col] + cfa[static_cast<size_t>(row+1)*w+col]
                          + cfa[static_cast<size_t>(row)*w+(col-1)] + cfa[static_cast<size_t>(row)*w+(col+1)])
                + 0.0625f * (cfa[static_cast<size_t>(row-1)*w+(col-1)] + cfa[static_cast<size_t>(row-1)*w+(col+1)]
                           + cfa[static_cast<size_t>(row+1)*w+(col-1)] + cfa[static_cast<size_t>(row+1)*w+(col+1)]);
        }
    }

    DLOOP(row, 4, height - 4)
    {
        int start_col = 4 + (fc(row, 0) & 1);
        for (int col = start_col; col < width - 4; col += 2) {
            size_t indx = static_cast<size_t>(row) * w + col;
            float VH_Central = VH_Dir[indx];
            float VH_Neighbour = 0.25f * (
                VH_Dir[static_cast<size_t>(row-1)*w+(col-1)] + VH_Dir[static_cast<size_t>(row-1)*w+(col+1)] +
                VH_Dir[static_cast<size_t>(row+1)*w+(col-1)] + VH_Dir[static_cast<size_t>(row+1)*w+(col+1)]);
            float VH_Disc = (std::abs(0.5f - VH_Central) < std::abs(0.5f - VH_Neighbour)) ? VH_Neighbour : VH_Central;

            float N_Grad = eps + std::abs(cfa[static_cast<size_t>(row-1)*w+col] - cfa[static_cast<size_t>(row+1)*w+col])
                + std::abs(cfa[indx] - cfa[static_cast<size_t>(row-2)*w+col])
                + std::abs(cfa[static_cast<size_t>(row-1)*w+col] - cfa[static_cast<size_t>(row-3)*w+col]);
            float S_Grad = eps + std::abs(cfa[static_cast<size_t>(row+1)*w+col] - cfa[static_cast<size_t>(row-1)*w+col])
                + std::abs(cfa[indx] - cfa[static_cast<size_t>(row+2)*w+col])
                + std::abs(cfa[static_cast<size_t>(row+1)*w+col] - cfa[static_cast<size_t>(row+3)*w+col]);
            float W_Grad = eps + std::abs(cfa[static_cast<size_t>(row)*w+(col-1)] - cfa[static_cast<size_t>(row)*w+(col+1)])
                + std::abs(cfa[indx] - cfa[static_cast<size_t>(row)*w+(col-2)])
                + std::abs(cfa[static_cast<size_t>(row)*w+(col-1)] - cfa[static_cast<size_t>(row)*w+(col-3)]);
            float E_Grad = eps + std::abs(cfa[static_cast<size_t>(row)*w+(col+1)] - cfa[static_cast<size_t>(row)*w+(col-1)])
                + std::abs(cfa[indx] - cfa[static_cast<size_t>(row)*w+(col+2)])
                + std::abs(cfa[static_cast<size_t>(row)*w+(col+1)] - cfa[static_cast<size_t>(row)*w+(col+3)]);

            auto safe_lpf = [&](int r, int c) -> float {
                int sr = std::max(2, std::min(r, height - 3));
                int sc = std::max(2, std::min(c, width - 3));
                return lpf[static_cast<size_t>(sr) * w + sc];
            };

            float N_Est = cfa[static_cast<size_t>(row-1)*w+col] * (1.0f + (safe_lpf(row,col) - safe_lpf(row-2,col))
                / (eps + safe_lpf(row,col) + safe_lpf(row-2,col)));
            float S_Est = cfa[static_cast<size_t>(row+1)*w+col] * (1.0f + (safe_lpf(row,col) - safe_lpf(row+2,col))
                / (eps + safe_lpf(row,col) + safe_lpf(row+2,col)));
            float W_Est = cfa[static_cast<size_t>(row)*w+(col-1)] * (1.0f + (safe_lpf(row,col) - safe_lpf(row,col-2))
                / (eps + safe_lpf(row,col) + safe_lpf(row,col-2)));
            float E_Est = cfa[static_cast<size_t>(row)*w+(col+1)] * (1.0f + (safe_lpf(row,col) - safe_lpf(row,col+2))
                / (eps + safe_lpf(row,col) + safe_lpf(row,col+2)));

            float V_Est = (S_Grad * N_Est + N_Grad * S_Est) / (N_Grad + S_Grad);
            float H_Est = (W_Grad * E_Est + E_Grad * W_Est) / (E_Grad + W_Grad);

            float laplacianV = (2.0f * cfa[indx] - cfa[static_cast<size_t>(row-2)*w+col] - cfa[static_cast<size_t>(row+2)*w+col]) * 0.25f;
            float laplacianH = (2.0f * cfa[indx] - cfa[static_cast<size_t>(row)*w+(col-2)] - cfa[static_cast<size_t>(row)*w+(col+2)]) * 0.25f;
            V_Est += laplacianV;
            H_Est += laplacianH;

            rgb1[indx] = std::max(0.0f, std::min(1.0f, VH_Disc * H_Est + (1.0f - VH_Disc) * V_Est));
        }
    }

    DLOOP(row, 4, height - 4)
    {
        int start_col = 4 + (fc(row, 0) & 1);
        for (int col = start_col; col < width - 4; col += 2) {
            size_t indx = static_cast<size_t>(row) * w + col;
            float P_Stat = 0.0f, Q_Stat = 0.0f;
            for (int k = -4; k <= 4; k++) {
                float cp = cfa[static_cast<size_t>(row+k)*w+(col+k)];
                float cq = cfa[static_cast<size_t>(row+k)*w+(col-k)];
                P_Stat += std::abs(cfa[indx] - cp); Q_Stat += std::abs(cfa[indx] - cq);
            }
            P_Stat = std::max(P_Stat, epssq); Q_Stat = std::max(Q_Stat, epssq);
            PQ_Dir[indx] = P_Stat / (P_Stat + Q_Stat);
        }
    }

    DLOOP(row, 4, height - 4)
    {
        int start_col = 4 + (fc(row, 0) & 1);
        for (int col = start_col; col < width - 4; col += 2) {
            size_t indx = static_cast<size_t>(row) * w + col;
            int c = 2 - fc(row, col);
            float PQ_Central = PQ_Dir[indx];
            float PQ_Neighbour = 0.25f * (
                PQ_Dir[static_cast<size_t>(row-1)*w+(col-1)] + PQ_Dir[static_cast<size_t>(row-1)*w+(col+1)] +
                PQ_Dir[static_cast<size_t>(row+1)*w+(col-1)] + PQ_Dir[static_cast<size_t>(row+1)*w+(col+1)]);
            float PQ_Disc = (std::abs(0.5f - PQ_Central) < std::abs(0.5f - PQ_Neighbour)) ? PQ_Neighbour : PQ_Central;

            auto get_rgb_c = [&](int r, int cc, int channel) -> float {
                if (r < 0 || r >= height || cc < 0 || cc >= width) return 0.0f;
                size_t i = static_cast<size_t>(r) * w + cc;
                if (channel == 0) return rgb0[i];
                if (channel == 1) return rgb1[i];
                return rgb2[i];
            };

            float NW_Grad = eps + std::abs(get_rgb_c(row-1,col-1,c) - get_rgb_c(row+1,col+1,c))
                + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row-2)*w+(col-2)])
                + std::abs(get_rgb_c(row-1,col-1,c) - get_rgb_c(row-3,col-3,c));
            float NE_Grad = eps + std::abs(get_rgb_c(row-1,col+1,c) - get_rgb_c(row+1,col-1,c))
                + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row-2)*w+(col+2)])
                + std::abs(get_rgb_c(row-1,col+1,c) - get_rgb_c(row-3,col+3,c));
            float SW_Grad = eps + std::abs(get_rgb_c(row+1,col-1,c) - get_rgb_c(row-1,col+1,c))
                + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row+2)*w+(col-2)])
                + std::abs(get_rgb_c(row+1,col-1,c) - get_rgb_c(row+3,col-3,c));
            float SE_Grad = eps + std::abs(get_rgb_c(row+1,col+1,c) - get_rgb_c(row-1,col-1,c))
                + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row+2)*w+(col+2)])
                + std::abs(get_rgb_c(row+1,col+1,c) - get_rgb_c(row+3,col+3,c));

            float NW_Est = get_rgb_c(row-1,col-1,c) - rgb1[static_cast<size_t>(row-1)*w+(col-1)];
            float NE_Est = get_rgb_c(row-1,col+1,c) - rgb1[static_cast<size_t>(row-1)*w+(col+1)];
            float SW_Est = get_rgb_c(row+1,col-1,c) - rgb1[static_cast<size_t>(row+1)*w+(col-1)];
            float SE_Est = get_rgb_c(row+1,col+1,c) - rgb1[static_cast<size_t>(row+1)*w+(col+1)];

            float P_Est = (NW_Grad * SE_Est + SE_Grad * NW_Est) / (NW_Grad + SE_Grad);
            float Q_Est = (NE_Grad * SW_Est + SW_Grad * NE_Est) / (NE_Grad + SW_Grad);

            float& target = (c == 0) ? rgb0[indx] : rgb2[indx];
            target = std::max(0.0f, std::min(1.0f, rgb1[indx] + (1.0f - PQ_Disc) * P_Est + PQ_Disc * Q_Est));
        }
    }

    DLOOP(row, 4, height - 4)
    {
        int start_col = 4 + (fc(row, 1) & 1);
        for (int col = start_col; col < width - 4; col += 2) {
            size_t indx = static_cast<size_t>(row) * w + col;
            float VH_Central = VH_Dir[indx];
            float VH_Neighbour = 0.25f * (
                VH_Dir[static_cast<size_t>(row-1)*w+(col-1)] + VH_Dir[static_cast<size_t>(row-1)*w+(col+1)] +
                VH_Dir[static_cast<size_t>(row+1)*w+(col-1)] + VH_Dir[static_cast<size_t>(row+1)*w+(col+1)]);
            float VH_Disc = (std::abs(0.5f - VH_Central) < std::abs(0.5f - VH_Neighbour)) ? VH_Neighbour : VH_Central;

            for (int c = 0; c <= 2; c += 2) {
                float N_Grad = eps + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row-2)*w+col])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row-1)*w+col] - (c==0?rgb0:rgb2)[static_cast<size_t>(row+1)*w+col])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row-1)*w+col] - (c==0?rgb0:rgb2)[static_cast<size_t>(row-3)*w+col]);
                float S_Grad = eps + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row+2)*w+col])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row+1)*w+col] - (c==0?rgb0:rgb2)[static_cast<size_t>(row-1)*w+col])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row+1)*w+col] - (c==0?rgb0:rgb2)[static_cast<size_t>(row+3)*w+col]);
                float W_Grad = eps + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row)*w+(col-2)])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col-1)] - (c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col+1)])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col-1)] - (c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col-3)]);
                float E_Grad = eps + std::abs(rgb1[indx] - rgb1[static_cast<size_t>(row)*w+(col+2)])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col+1)] - (c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col-1)])
                    + std::abs((c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col+1)] - (c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col+3)]);

                float N_CDiff = (c==0?rgb0:rgb2)[static_cast<size_t>(row-1)*w+col] - rgb1[static_cast<size_t>(row-1)*w+col];
                float S_CDiff = (c==0?rgb0:rgb2)[static_cast<size_t>(row+1)*w+col] - rgb1[static_cast<size_t>(row+1)*w+col];
                float W_CDiff = (c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col-1)] - rgb1[static_cast<size_t>(row)*w+(col-1)];
                float E_CDiff = (c==0?rgb0:rgb2)[static_cast<size_t>(row)*w+(col+1)] - rgb1[static_cast<size_t>(row)*w+(col+1)];

                float V_Est = (S_Grad * N_CDiff + N_Grad * S_CDiff) / (N_Grad + S_Grad);
                float H_Est = (W_Grad * E_CDiff + E_Grad * W_CDiff) / (E_Grad + W_Grad);

                float& target = (c == 0) ? rgb0[indx] : rgb2[indx];
                target = std::max(0.0f, std::min(1.0f, rgb1[indx] + VH_Disc * H_Est + (1.0f - VH_Disc) * V_Est));
            }
        }
    }

    for (size_t i = 0; i < total; i++) {
        L[i] = rgb0[i] - rgb1[i];
        M[i] = rgb2[i] - rgb1[i];
    }

    // Apply median filter to color-difference planes using separate output buffers
    // to avoid in-place corruption (subsequent neighbors would read already-filtered values)
    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            float wL[9], wM[9];
            int vi = 0;
            for (int dy = -1; dy <= 1; dy++) {
                size_t row_off = static_cast<size_t>(row + dy) * width;
                for (int dx = -1; dx <= 1; dx++) {
                    wL[vi] = L[row_off + (col + dx)];
                    wM[vi] = M[row_off + (col + dx)];
                    vi++;
                }
            }
            size_t idx = static_cast<size_t>(row) * width + col;
            L_tmp[idx] = prism_detail::median_3x3(wL);
            M_tmp[idx] = prism_detail::median_3x3(wM);
        }
    }
    // Copy filtered results back to L and M
    std::copy(L_tmp.begin(), L_tmp.end(), L.begin());
    std::copy(M_tmp.begin(), M_tmp.end(), M.begin());

    for (int row = 1; row < height - 1; row++) {
        for (int col = 1; col < width - 1; col++) {
            size_t idx = static_cast<size_t>(row) * width + col;
            rgb0[idx] = std::max(0.0f, std::min(1.0f, rgb1[idx] + L[idx]));
            rgb2[idx] = std::max(0.0f, std::min(1.0f, rgb1[idx] + M[idx]));
        }
    }

    detail::fill_intermediate_borders(rgb0.data(), rgb1.data(), rgb2.data(), width, height, 4);

    DLOOP(y, 0, height)
    {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            int r_out = std::max(0, std::min(static_cast<int>(rgb0[idx] * scale + 0.5f), max_val));
            int g_out = std::max(0, std::min(static_cast<int>(rgb1[idx] * scale + 0.5f), max_val));
            int b_out = std::max(0, std::min(static_cast<int>(rgb2[idx] * scale + 0.5f), max_val));
            set_rgb_raw(rgb, x, y, width, r_out, g_out, b_out, bit_depth);
        }
    }

    return DemosaicError::Ok;
}

} // namespace imop
