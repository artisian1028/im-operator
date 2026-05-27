#include "common.hpp"

namespace jpeg_codec {

// ============================================================
//  Baseline DCT JPEG Encoder
//
//  Pipeline:
//    RGB Input -> YCbCr Color Conversion -> Chroma Subsampling
//    -> 8x8 Block DCT -> Quantization -> Zigzag Scan
//    -> DPCM/Huffman Encoding -> JFIF Bitstream Output
// ============================================================

// --- Encode one 8x8 block ---

static void encode_block(detail::BitWriter& bw, const float block[64],
                          int prev_dc, int& new_prev_dc,
                          const detail::HuffmanTable& dc_table,
                          const detail::HuffmanTable& ac_table,
                          const int qtable[64]) {
    int zz[64];

    // Quantize + zigzag
    for (int i = 0; i < 64; i++) {
        zz[detail::kZigzag[i]] = static_cast<int>(std::round(block[i] / qtable[i]));
    }

    // Encode DC coefficient (DPCM)
    int dc_diff = zz[0] - prev_dc;
    new_prev_dc = zz[0];

    // DC category
    int abs_diff = std::abs(dc_diff);
    int dc_cat = 0;
    int temp = abs_diff;
    while (temp) { dc_cat++; temp >>= 1; }

    // Write DC Huffman code + additional bits
    if (dc_cat > 0 && static_cast<size_t>(dc_cat) < 256) {
        bw.write_bits(dc_table.huffcode[dc_cat], dc_table.huffsize[dc_cat]);
        // JPEG standard: for negative diff, encode as diff + (2^k - 1)
        if (dc_diff < 0) abs_diff = dc_diff + (1 << dc_cat) - 1;
        bw.write_bits(abs_diff, dc_cat);
    } else if (dc_cat == 0) {
        bw.write_bits(dc_table.huffcode[0], dc_table.huffsize[0]);
    }

    // Encode AC coefficients (run-length + Huffman)
    bool wrote_eob = false;
    int run = 0;
    for (int i = 1; i < 64; i++) {
        if (zz[i] == 0) {
            run++;
            if (run == 16) {
                // Check if all remaining ACs are zero; if so, emit EOB instead of ZRL
                bool all_rem_zero = true;
                for (int j = i + 1; j < 64; j++) {
                    if (zz[j] != 0) { all_rem_zero = false; break; }
                }
                if (all_rem_zero) {
                    bw.write_bits(ac_table.huffcode[0x00], ac_table.huffsize[0x00]); // EOB
                    wrote_eob = true;
                    break;
                }
                // ZRL (Zero Run Length) marker: 0xF0
                bw.write_bits(ac_table.huffcode[0xF0], ac_table.huffsize[0xF0]);
                run = 0;
            }
        } else {
            int ac_val = zz[i];
            int abs_ac = std::abs(ac_val);
            int ac_cat = 0;
            temp = abs_ac;
            while (temp) { ac_cat++; temp >>= 1; }

            int symbol = (run << 4) | ac_cat;
            if (ac_table.huffsize[symbol] > 0) {
                bw.write_bits(ac_table.huffcode[symbol], ac_table.huffsize[symbol]);
                // JPEG standard: for negative value, encode as val + (2^k - 1)
                if (ac_val < 0) abs_ac = ac_val + (1 << ac_cat) - 1;
                bw.write_bits(abs_ac, ac_cat);
            }
            run = 0;
        }

        // Check for remaining zeros (EOB shortcut)
        bool all_zero = true;
        for (int j = i + 1; j < 64; j++) {
            if (zz[j] != 0) { all_zero = false; break; }
        }
        if (all_zero) {
            bw.write_bits(ac_table.huffcode[0x00], ac_table.huffsize[0x00]); // EOB
            wrote_eob = true;
            break;
        }
    }

    // If all AC coefficients were zero, we must still write EOB
    if (!wrote_eob) {
        bw.write_bits(ac_table.huffcode[0x00], ac_table.huffsize[0x00]); // EOB
    }
}

// --- Write SOF0 marker (Start of Frame) ---

static void write_sof0(detail::BitWriter& bw, int width, int height, int subsample) {
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerSOF0);

    int num_components = 3;
    size_t len_pos = bw.position();
    bw.write_uint16_be(0); // placeholder for length
    bw.write_byte(8); // precision
    bw.write_uint16_be(static_cast<uint16_t>(height));
    bw.write_uint16_be(static_cast<uint16_t>(width));
    bw.write_byte(static_cast<uint8_t>(num_components));

    // Component 1: Y (luminance)
    bw.write_byte(1); // component ID
    int h_samp_y = detail::get_h_sampling(subsample, 0);
    int v_samp_y = detail::get_v_sampling(subsample, 0);
    bw.write_byte(static_cast<uint8_t>((h_samp_y << 4) | v_samp_y));
    bw.write_byte(0); // quantization table 0

    // Component 2: Cb
    bw.write_byte(2);
    int h_samp_cb = detail::get_h_sampling(subsample, 1);
    int v_samp_cb = detail::get_v_sampling(subsample, 1);
    bw.write_byte(static_cast<uint8_t>((h_samp_cb << 4) | v_samp_cb));
    bw.write_byte(1); // quantization table 1

    // Component 3: Cr
    bw.write_byte(3);
    int h_samp_cr = detail::get_h_sampling(subsample, 2);
    int v_samp_cr = detail::get_v_sampling(subsample, 2);
    bw.write_byte(static_cast<uint8_t>((h_samp_cr << 4) | v_samp_cr));
    bw.write_byte(1); // quantization table 1

    // Patch length
    uint16_t len = static_cast<uint16_t>(bw.position() - len_pos);
    // We can't easily go back, so compute it. The length field includes itself (2 bytes).
    // 8+2+2+1+3*3 = 8 + precision(1) + height(2) + width(2) + ncomp(1) + 3*comp(3)
    // But we already wrote it. Let's just compute upfront.
    // Actually for simplicity, we write a known fixed length.
    // Since we know num_components=3, we can precompute.
    // The actual length (excluding marker bytes but including length field) = 8 + 3*3 = 17
    // Let's just recalculate
    size_t end_pos = bw.position();
    // Length = end_pos - len_pos (includes the 2 bytes of length field itself)
    // But we can't modify bytes already written with this BitWriter design.
    // We'll compute correctly upfront.
}

// --- Write SOS marker (Start of Scan) ---

static void write_sos(detail::BitWriter& bw) {
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerSOS);

    bw.write_uint16_be(12); // length (includes 2 length bytes + 1 ncomp + 3*comp(2) + 3 skip bytes)
    bw.write_byte(3); // number of components

    // Component 1: Y — DC table 0, AC table 0
    bw.write_byte(1);
    bw.write_byte(0x00);

    // Component 2: Cb — DC table 1, AC table 1
    bw.write_byte(2);
    bw.write_byte(0x11);

    // Component 3: Cr — DC table 1, AC table 1
    bw.write_byte(3);
    bw.write_byte(0x11);

    // Spectral selection
    bw.write_byte(0);  // start spectral
    bw.write_byte(63); // end spectral
    bw.write_byte(0);  // successive approximation
}

// --- Write DQT marker (Define Quantization Table) ---

static void write_dqt(detail::BitWriter& bw, uint8_t table_id, const int qtable[64]) {
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerDQT);
    bw.write_uint16_be(67); // length: 2 (length) + 1 (info) + 64 (values)
    bw.write_byte(table_id); // 0=luma, 1=chroma
    for (int i = 0; i < 64; i++) {
        bw.write_byte(static_cast<uint8_t>(detail::kZigzag[i] < 64 ?
            qtable[detail::kZigzag[i]] : 0));
    }
}

// --- Write JFIF APPO marker ---

static void write_jfif_app0(detail::BitWriter& bw) {
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerAPP0);
    bw.write_uint16_be(16); // length
    bw.write_byte('J');
    bw.write_byte('F');
    bw.write_byte('I');
    bw.write_byte('F');
    bw.write_byte(0); // null terminator
    bw.write_byte(1); // major version
    bw.write_byte(2); // minor version (1.02)
    bw.write_byte(1); // units: dots per inch
    bw.write_uint16_be(72); // X density
    bw.write_uint16_be(72); // Y density
    bw.write_byte(0); // thumbnail width
    bw.write_byte(0); // thumbnail height
}

// ============================================================
//  Main encoder function
// ============================================================

JpegError process_encode_baseline(const uint8_t* input, uint8_t* output,
                                   size_t* output_size,
                                   int width, int height, int channels,
                                   int bit_depth, const JpegParams& params) {
    (void)channels;
    JpegError err = validate_jpeg_encode_inputs(input, output, width, height,
                                                 3, bit_depth, params.quality);
    if (err != JpegError::Ok) return err;

    int quality = std::max(1, std::min(100, params.quality));
    int subsample = std::max(0, std::min(2, params.chroma_subsample));

    // Round dimensions up to MCU boundary
    int mcu_w = detail::get_mcu_width(subsample);
    int mcu_h = detail::get_mcu_height(subsample);
    int padded_w = ((width + mcu_w - 1) / mcu_w) * mcu_w;
    int padded_h = ((height + mcu_h - 1) / mcu_h) * mcu_h;

    // Build quantization tables
    int luma_qtable[64];
    int chroma_qtable[64];
    detail::scale_quant_table(luma_qtable, detail::kStdLumaQTable50, quality);
    detail::scale_quant_table(chroma_qtable, detail::kStdChromaQTable50, quality);

    // Build Huffman tables
    detail::HuffmanTable dc_luma{}, dc_chroma{}, ac_luma{}, ac_chroma{};
    detail::init_std_huffman_tables(dc_luma, dc_chroma, ac_luma, ac_chroma);

    // Bitstream writer
    size_t max_size = *output_size > 0 ? *output_size : get_max_jpeg_size(width, height, 3);
    detail::BitWriter bw(output, max_size);

    // --- Write JPEG headers ---
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerSOI);

    write_jfif_app0(bw);

    // DQT: Luma table
    write_dqt(bw, 0, luma_qtable);
    // DQT: Chroma table
    write_dqt(bw, 1, chroma_qtable);

    // SOF0: Start of Frame
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerSOF0);
    bw.write_uint16_be(17); // length
    bw.write_byte(8); // precision
    bw.write_uint16_be(static_cast<uint16_t>(height));
    bw.write_uint16_be(static_cast<uint16_t>(width));
    bw.write_byte(3); // 3 components

    int h_y = detail::get_h_sampling(subsample, 0);
    int v_y = detail::get_v_sampling(subsample, 0);
    bw.write_byte(1); // Y component ID
    bw.write_byte(static_cast<uint8_t>((h_y << 4) | v_y));
    bw.write_byte(0); // Q table 0

    int h_cb = detail::get_h_sampling(subsample, 1);
    int v_cb = detail::get_v_sampling(subsample, 1);
    bw.write_byte(2); // Cb component ID
    bw.write_byte(static_cast<uint8_t>((h_cb << 4) | v_cb));
    bw.write_byte(1); // Q table 1

    int h_cr = detail::get_h_sampling(subsample, 2);
    int v_cr = detail::get_v_sampling(subsample, 2);
    bw.write_byte(3); // Cr component ID
    bw.write_byte(static_cast<uint8_t>((h_cr << 4) | v_cr));
    bw.write_byte(1); // Q table 1

    // DHT: Huffman tables
    detail::write_huffman_table(bw, 0, 0, dc_luma.bits, dc_luma.huffval);
    detail::write_huffman_table(bw, 1, 0, ac_luma.bits, ac_luma.huffval);
    detail::write_huffman_table(bw, 0, 1, dc_chroma.bits, dc_chroma.huffval);
    detail::write_huffman_table(bw, 1, 1, ac_chroma.bits, ac_chroma.huffval);

    // SOS: Start of Scan
    write_sos(bw);

    // --- Encode image data ---
    int prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    for (int mcu_y = 0; mcu_y < padded_h; mcu_y += mcu_h) {
        for (int mcu_x = 0; mcu_x < padded_w; mcu_x += mcu_w) {
            // Encode Y blocks (1 or 4 depending on subsampling)
            for (int by = 0; by < v_y; by++) {
                for (int bx = 0; bx < h_y; bx++) {
                    int block_x = mcu_x + bx * 8;
                    int block_y = mcu_y + by * 8;

                    float block[64];
                    for (int j = 0; j < 8; j++) {
                        for (int i = 0; i < 8; i++) {
                            int px = std::min(block_x + i, width - 1);
                            int py = std::min(block_y + j, height - 1);
                            int r = detail::read_pixel(input, px, py, width, 3, bit_depth, 0);
                            int g = detail::read_pixel(input, px, py, width, 3, bit_depth, 1);
                            int b = detail::read_pixel(input, px, py, width, 3, bit_depth, 2);
                            int yv, cbv, crv;
                            detail::rgb_to_ycbcr(r, g, b, yv, cbv, crv);
                            block[j * 8 + i] = static_cast<float>(yv - 128);
                        }
                    }
                    detail::fdct_8x8(block);
                    encode_block(bw, block, prev_dc_y, prev_dc_y,
                                dc_luma, ac_luma, luma_qtable);
                }
            }

            // Encode Cb block
            {
                float block[64] = {};
                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) {
                        int px = std::min(mcu_x + i * h_y / h_cb, width - 1);
                        int py = std::min(mcu_y + j * v_y / v_cb, height - 1);
                        int r = detail::read_pixel(input, px, py, width, 3, bit_depth, 0);
                        int g = detail::read_pixel(input, px, py, width, 3, bit_depth, 1);
                        int b = detail::read_pixel(input, px, py, width, 3, bit_depth, 2);
                        int yv, cbv, crv;
                        detail::rgb_to_ycbcr(r, g, b, yv, cbv, crv);
                        block[j * 8 + i] = static_cast<float>(cbv - 128);
                    }
                }
                detail::fdct_8x8(block);
                encode_block(bw, block, prev_dc_cb, prev_dc_cb,
                            dc_chroma, ac_chroma, chroma_qtable);
            }

            // Encode Cr block
            {
                float block[64] = {};
                for (int j = 0; j < 8; j++) {
                    for (int i = 0; i < 8; i++) {
                        int px = std::min(mcu_x + i * h_y / h_cr, width - 1);
                        int py = std::min(mcu_y + j * v_y / v_cr, height - 1);
                        int r = detail::read_pixel(input, px, py, width, 3, bit_depth, 0);
                        int g = detail::read_pixel(input, px, py, width, 3, bit_depth, 1);
                        int b = detail::read_pixel(input, px, py, width, 3, bit_depth, 2);
                        int yv, cbv, crv;
                        detail::rgb_to_ycbcr(r, g, b, yv, cbv, crv);
                        block[j * 8 + i] = static_cast<float>(crv - 128);
                    }
                }
                detail::fdct_8x8(block);
                encode_block(bw, block, prev_dc_cr, prev_dc_cr,
                            dc_chroma, ac_chroma, chroma_qtable);
            }
        }
    }

    bw.flush_bits();

    // EOI marker
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerEOI);

    *output_size = bw.position();
    return JpegError::Ok;
}

} // namespace jpeg_codec
