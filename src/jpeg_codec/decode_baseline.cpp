#include "common.hpp"

namespace jpeg_codec {

// ============================================================
//  Baseline DCT JPEG Decoder
//
//  Pipeline:
//    JFIF Bitstream -> Parse Headers -> Huffman Decode -> Inverse Zigzag
//    -> Inverse Quantization -> 8x8 Block IDCT -> YCbCr to RGB Output
// ============================================================

// Forward declaration of decode helpers
struct JpegDecoderState {
    int width = 0;
    int height = 0;
    int num_components = 0;
    int restart_interval = 0;

    int luma_qtable[64];
    int chroma_qtable[64];

    detail::HuffmanTable dc_luma;
    detail::HuffmanTable dc_chroma;
    detail::HuffmanTable ac_luma;
    detail::HuffmanTable ac_chroma;

    int h_samp[3] = {1, 1, 1};
    int v_samp[3] = {1, 1, 1};
    int qtable_id[3] = {0, 1, 1};

    bool dqt_seen[4] = {};
    bool sof_seen = false;
    bool sos_seen = false;
};

// Decode one 8x8 block from the bitstream
static bool decode_block(detail::BitReader& br, int block[64],
                          int& prev_dc,
                          const detail::HuffmanTable& dc_table,
                          const detail::HuffmanTable& ac_table,
                          const int qtable[64]) {
    // Clear block
    for (int i = 0; i < 64; i++) block[i] = 0;

    // Decode DC coefficient
    int dc_cat = 0;
    int code = 0;
    bool found = false;
    for (int len = 1; len <= 16; len++) {
        code = (code << 1) | br.read_bits(1);
        for (int v = 0; v < 256; v++) {
            if (dc_table.huffsize[v] == len && dc_table.huffcode[v] == code) {
                dc_cat = v;
                found = true;
                break;
            }
        }
        if (found) break;
    }
    if (!found) return false;

    if (dc_cat == 0) {
        block[0] = prev_dc;
    } else {
        int bits = br.read_bits(dc_cat);
        if (bits < (1 << (dc_cat - 1))) {
            bits -= (1 << dc_cat) - 1;
        }
        block[0] = prev_dc + bits;
    }
    prev_dc = block[0];

    // Decode AC coefficients
    int k = 1;
    while (k < 64) {
        code = 0;
        found = false;
        int symbol = 0;
        for (int len = 1; len <= 16; len++) {
            code = (code << 1) | br.read_bits(1);
            for (int v = 0; v < 256; v++) {
                if (ac_table.huffsize[v] == len && ac_table.huffcode[v] == code) {
                    symbol = v;
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        if (!found) return false;

        int rrrr = (symbol >> 4) & 0x0F;
        int ssss = symbol & 0x0F;

        if (ssss == 0) {
            if (rrrr == 0) {
                // EOB
                break;
            } else if (rrrr == 15) {
                // ZRL
                k += 16;
                continue;
            }
        }

        k += rrrr;
        if (k >= 64) break;

        int bits = br.read_bits(ssss);
        if (bits < (1 << (ssss - 1))) {
            bits -= (1 << ssss) - 1;
        }
        block[detail::kZigzag[k]] = bits;
        k++;
    }

    // Inverse quantization (both block and qtable are in 2D order)
    for (int i = 0; i < 64; i++) {
        block[i] = block[i] * qtable[i];
    }

    return true;
}

// Parse JPEG markers and build decoder state
static JpegError parse_jpeg_headers(detail::BitReader& br, JpegDecoderState& state) {
    // Check SOI marker
    if (br.read_byte() != 0xFF || br.read_byte() != detail::kMarkerSOI) {
        return JpegError::InvalidJpegData;
    }

    bool parsing = true;
    while (parsing && !br.eof()) {
        uint8_t b = br.read_byte();
        if (b != 0xFF) {
            // Should be at a marker
            continue;
        }

        // Skip any padding 0xFF bytes
        uint8_t marker = br.read_byte();
        while (marker == 0xFF && !br.eof()) marker = br.read_byte();

        switch (marker) {
            case detail::kMarkerSOF0: {
                // Start of Frame (Baseline DCT)
                uint16_t length = br.read_uint16_be();
                size_t start = br.position();
                uint8_t precision = br.read_byte();
                if (precision != 8) return JpegError::InvalidJpegData;

                state.height = br.read_uint16_be();
                state.width = br.read_uint16_be();
                state.num_components = br.read_byte();

                if (state.num_components < 1 || state.num_components > 4)
                    return JpegError::InvalidJpegData;

                for (int i = 0; i < state.num_components; i++) {
                    uint8_t comp_id = br.read_byte();
                    uint8_t sampling = br.read_byte();
                    uint8_t qt_id = br.read_byte();

                    int idx = comp_id - 1;
                    if (idx >= 0 && idx < 3) {
                        state.h_samp[idx] = (sampling >> 4) & 0x0F;
                        state.v_samp[idx] = sampling & 0x0F;
                        state.qtable_id[idx] = qt_id;
                    }
                }
                state.sof_seen = true;

                // Skip any remaining bytes in this segment
                size_t consumed = br.position() - start;
                if (consumed + 2 < length) br.skip(length - 2 - consumed);
                break;
            }
            case detail::kMarkerDQT: {
                uint16_t length = br.read_uint16_be();
                size_t start = br.position();
                size_t end = start + length - 2;

                while (br.position() < end) {
                    uint8_t info = br.read_byte();
                    uint8_t table_id = info & 0x0F;
                    bool is_16bit = (info & 0xF0) != 0;

                    int* dst = (table_id == 0) ? state.luma_qtable : state.chroma_qtable;
                    state.dqt_seen[table_id] = true;

                    if (is_16bit) {
                        for (int i = 0; i < 64; i++) {
                            dst[i] = br.read_uint16_be();
                        }
                    } else {
                        for (int i = 0; i < 64; i++) {
                            dst[detail::kZigzag[i]] = br.read_byte();
                        }
                    }
                }
                break;
            }
            case detail::kMarkerDHT: {
                uint16_t length = br.read_uint16_be();
                size_t start = br.position();
                size_t end = start + length - 2;

                while (br.position() < end) {
                    uint8_t info = br.read_byte();
                    uint8_t table_class = (info >> 4) & 0x0F;
                    uint8_t table_id = info & 0x0F;

                    uint8_t bits[16];
                    int total = 0;
                    for (int i = 0; i < 16; i++) {
                        bits[i] = br.read_byte();
                        total += bits[i];
                    }

                    uint8_t huffval[256] = {};
                    for (int i = 0; i < total && i < 256; i++) {
                        huffval[i] = br.read_byte();
                    }

                    detail::HuffmanTable* tbl = nullptr;
                    if (table_class == 0 && table_id == 0) {
                        tbl = &state.dc_luma;
                    } else if (table_class == 0 && table_id == 1) {
                        tbl = &state.dc_chroma;
                    } else if (table_class == 1 && table_id == 0) {
                        tbl = &state.ac_luma;
                    } else if (table_class == 1 && table_id == 1) {
                        tbl = &state.ac_chroma;
                    }

                    if (tbl) {
                        std::memcpy(tbl->bits, bits, 16);
                        for (int i = 0; i < total; i++) tbl->huffval[i] = huffval[i];
                        detail::build_huffman_lookup(*tbl);
                    }
                }
                break;
            }
            case detail::kMarkerSOS: {
                // Start of Scan — we have all headers, ready to decode image data
                uint16_t length = br.read_uint16_be();
                size_t sos_start = br.position();
                uint8_t ncomp = br.read_byte();

                for (int i = 0; i < ncomp; i++) {
                    br.read_byte(); // component ID
                    br.read_byte(); // table selector
                }

                // Skip spectral selection bytes
                br.read_byte(); // Ss
                br.read_byte(); // Se
                br.read_byte(); // Ah/Al

                // Consume remaining bytes in SOS header
                size_t consumed = br.position() - sos_start;
                if (consumed + 2 < length) br.skip(length - 2 - consumed);

                // Image data follows directly after SOS
                parsing = false;
                state.sos_seen = true;
                break;
            }
            case detail::kMarkerAPP0:
            case detail::kMarkerCOM: {
                // Skip application/comment segments
                uint16_t length = br.read_uint16_be();
                if (length >= 2) br.skip(length - 2);
                break;
            }
            case 0xDD: // DRI (restart interval)
            {
                uint16_t length = br.read_uint16_be();
                if (length >= 2) br.skip(length - 2);
                break;
            }
            case detail::kMarkerEOI:
                // End of Image — nothing more to parse
                parsing = false;
                break;
            case 0x01: // TEM
                break;
            case 0xD0: case 0xD1: case 0xD2: case 0xD3:
            case 0xD4: case 0xD5: case 0xD6: case 0xD7:
                // RST markers — skip during header parsing
                break;
            default: {
                // Unknown marker with length
                if (br.eof()) {
                    parsing = false;
                } else {
                    uint16_t length = br.read_uint16_be();
                    if (length >= 2) br.skip(length - 2);
                }
                break;
            }
        }
    }

    if (!state.sof_seen) return JpegError::InvalidJpegData;
    if (!state.sos_seen) return JpegError::InvalidJpegData;
    if (state.width <= 0 || state.height <= 0) return JpegError::InvalidJpegData;

    return JpegError::Ok;
}

// ============================================================
//  Main decoder function
// ============================================================

JpegError process_decode_baseline(const uint8_t* input, size_t input_size,
                                   uint8_t* output,
                                   int* width, int* height, int* channels) {
    if (!input || !output || !width || !height || !channels)
        return JpegError::NullInput;
    if (input_size < 2)
        return JpegError::InvalidJpegData;

    // Check SOI marker
    if (input[0] != 0xFF || input[1] != detail::kMarkerSOI)
        return JpegError::InvalidJpegData;

    detail::BitReader br(input, input_size);

    // Use default tables initially; they'll be overridden by stream markers
    JpegDecoderState state;
    // Init default quantization tables from standard
    for (int i = 0; i < 64; i++) {
        state.luma_qtable[i] = detail::kStdLumaQTable50[detail::kZigzag[i]];
        state.chroma_qtable[i] = detail::kStdChromaQTable50[detail::kZigzag[i]];
    }
    detail::init_std_huffman_tables(state.dc_luma, state.dc_chroma,
                                     state.ac_luma, state.ac_chroma);

    // Parse headers
    JpegError err = parse_jpeg_headers(br, state);
    if (err != JpegError::Ok) return err;

    // Re-init quantization tables in zigzag order for decode use
    int luma_dq[64], chroma_dq[64];
    for (int i = 0; i < 64; i++) {
        luma_dq[detail::kZigzag[i]] = state.luma_qtable[i];
        chroma_dq[detail::kZigzag[i]] = state.chroma_qtable[i];
    }

    int w = state.width;
    int h = state.height;

    *width = w;
    *height = h;
    *channels = 3;

    // Get sampling factors
    int h_max = std::max(state.h_samp[0], std::max(state.h_samp[1], state.h_samp[2]));
    int v_max = std::max(state.v_samp[0], std::max(state.v_samp[1], state.v_samp[2]));
    int mcu_w = h_max * 8;
    int mcu_h = v_max * 8;
    int padded_w = ((w + mcu_w - 1) / mcu_w) * mcu_w;
    int padded_h = ((h + mcu_h - 1) / mcu_h) * mcu_h;

    // Allocate component planes
    std::vector<int> plane_y(static_cast<size_t>(padded_w) * padded_h, 128);
    std::vector<int> plane_cb(static_cast<size_t>(padded_w / h_max * state.h_samp[1]) *
                              (padded_h / v_max * state.v_samp[1]), 128);
    std::vector<int> plane_cr(static_cast<size_t>(padded_w / h_max * state.h_samp[2]) *
                              (padded_h / v_max * state.v_samp[2]), 128);

    int prev_dc[3] = {0, 0, 0};

    // Decode MCUs
    int mcu_count_x = padded_w / mcu_w;
    int mcu_count_y = padded_h / mcu_h;

    for (int mcu_y = 0; mcu_y < mcu_count_y; mcu_y++) {
        for (int mcu_x = 0; mcu_x < mcu_count_x; mcu_x++) {
            // Decode each component's blocks within this MCU
            for (int comp = 0; comp < 3; comp++) {
                int h_s = state.h_samp[comp];
                int v_s = state.v_samp[comp];
                const int* qt = (comp == 0) ? luma_dq : chroma_dq;
                detail::HuffmanTable* dc_tbl = (comp == 0) ? &state.dc_luma : &state.dc_chroma;
                detail::HuffmanTable* ac_tbl = (comp == 0) ? &state.ac_luma : &state.ac_chroma;

                for (int by = 0; by < v_s; by++) {
                    for (int bx = 0; bx < h_s; bx++) {
                        int block[64];
                        if (!decode_block(br, block, prev_dc[comp], *dc_tbl, *ac_tbl, qt)) {
                            return JpegError::DecodeFailed;
                        }

                        // IDCT
                        float fblock[64];
                        for (int i = 0; i < 64; i++) fblock[i] = static_cast<float>(block[i]);
                        detail::idct_8x8(fblock);

                        // Store in component plane
                        int base_x = mcu_x * mcu_w / h_max + bx * 8;
                        int base_y = mcu_y * mcu_h / v_max + by * 8;
                        int comp_w = padded_w / h_max * h_s;

                        for (int j = 0; j < 8; j++) {
                            for (int i = 0; i < 8; i++) {
                                int val = detail::clamp_val(
                                    static_cast<int>(fblock[j * 8 + i] + 128.5f), 255);
                                int px = base_x + i;
                                int py = base_y + j;
                                if (comp == 0) {
                                    plane_y[static_cast<size_t>(py) * padded_w + px] = val;
                                } else if (comp == 1) {
                                    plane_cb[static_cast<size_t>(py) * comp_w + px] = val;
                                } else {
                                    plane_cr[static_cast<size_t>(py) * comp_w + px] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // YCbCr to RGB conversion
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Map pixel position to component plane coordinates
            int cb_x = x * state.h_samp[1] / h_max;
            int cb_y = y * state.v_samp[1] / v_max;
            int cr_x = x * state.h_samp[2] / h_max;
            int cr_y = y * state.v_samp[2] / v_max;

            int comp_w_cb = padded_w / h_max * state.h_samp[1];
            int comp_w_cr = padded_w / h_max * state.h_samp[2];

            int yv = plane_y[static_cast<size_t>(y) * padded_w + x];
            int cbv = plane_cb[static_cast<size_t>(cb_y) * comp_w_cb + cb_x];
            int crv = plane_cr[static_cast<size_t>(cr_y) * comp_w_cr + cr_x];

            int r, g, b;
            detail::ycbcr_to_rgb(yv, cbv, crv, r, g, b);

            detail::write_pixel(output, x, y, w, 3, 8, 0, detail::clamp_val(r, 255));
            detail::write_pixel(output, x, y, w, 3, 8, 1, detail::clamp_val(g, 255));
            detail::write_pixel(output, x, y, w, 3, 8, 2, detail::clamp_val(b, 255));
        }
    }

    return JpegError::Ok;
}

} // namespace jpeg_codec
