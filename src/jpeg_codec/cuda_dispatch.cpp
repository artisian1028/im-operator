#include "jpeg_codec/algorithms.hpp"
#include "common.hpp"
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef IM_OPERATOR_HAS_CUDA

#include <cuda_runtime.h>
#include "jpeg_codec_kernels.cuh"

namespace jpeg_codec {

static bool s_cuda_available = false;
static std::once_flag s_cuda_once;

bool has_cuda() {
    std::call_once(s_cuda_once, [] {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        s_cuda_available = (err == cudaSuccess && device_count > 0);
    });
    return s_cuda_available;
}

const char* cuda_device_name() {
    if (!has_cuda()) return "N/A";
    static char name_buf[256];
    static bool cached = false;
    if (!cached) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) return "Unknown";
        size_t len = 0;
        while (len < 255 && prop.name[len]) { name_buf[len] = prop.name[len]; len++; }
        name_buf[len] = '\0';
        cached = true;
    }
    return name_buf;
}

void cuda_synchronize() {
    cudaDeviceSynchronize();
}

static thread_local JpegCudaWorkspace t_cuda_ws;

// ============================================================
//  Helper: Write JPEG headers to output buffer
// ============================================================

static void write_jpeg_headers(detail::BitWriter& bw, int width, int height,
                                 int quality, int subsample,
                                 const int luma_qt[64], const int chroma_qt[64]) {
    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerSOI);

    // JFIF APP0
    bw.write_byte(0xFF); bw.write_byte(detail::kMarkerAPP0);
    bw.write_uint16_be(16);
    bw.write_byte('J'); bw.write_byte('F'); bw.write_byte('I'); bw.write_byte('F');
    bw.write_byte(0); bw.write_byte(1); bw.write_byte(2); bw.write_byte(1);
    bw.write_uint16_be(72); bw.write_uint16_be(72);
    bw.write_byte(0); bw.write_byte(0);

    // DQT Luma
    bw.write_byte(0xFF); bw.write_byte(detail::kMarkerDQT);
    bw.write_uint16_be(67); bw.write_byte(0);
    for (int i = 0; i < 64; i++)
        bw.write_byte(static_cast<uint8_t>(luma_qt[detail::kZigzag[i]]));

    // DQT Chroma
    bw.write_byte(0xFF); bw.write_byte(detail::kMarkerDQT);
    bw.write_uint16_be(67); bw.write_byte(1);
    for (int i = 0; i < 64; i++)
        bw.write_byte(static_cast<uint8_t>(chroma_qt[detail::kZigzag[i]]));

    // SOF0
    int h_y = detail::get_h_sampling(subsample, 0);
    int v_y = detail::get_v_sampling(subsample, 0);
    int h_cb = detail::get_h_sampling(subsample, 1);
    int v_cb = detail::get_v_sampling(subsample, 1);
    int h_cr = detail::get_h_sampling(subsample, 2);
    int v_cr = detail::get_v_sampling(subsample, 2);

    bw.write_byte(0xFF); bw.write_byte(detail::kMarkerSOF0);
    bw.write_uint16_be(17); bw.write_byte(8);
    bw.write_uint16_be(static_cast<uint16_t>(height));
    bw.write_uint16_be(static_cast<uint16_t>(width));
    bw.write_byte(3);
    bw.write_byte(1); bw.write_byte(static_cast<uint8_t>((h_y << 4) | v_y)); bw.write_byte(0);
    bw.write_byte(2); bw.write_byte(static_cast<uint8_t>((h_cb << 4) | v_cb)); bw.write_byte(1);
    bw.write_byte(3); bw.write_byte(static_cast<uint8_t>((h_cr << 4) | v_cr)); bw.write_byte(1);

    // DHT tables (standard)
    detail::HuffmanTable dc_luma, dc_chroma, ac_luma, ac_chroma;
    detail::init_std_huffman_tables(dc_luma, dc_chroma, ac_luma, ac_chroma);
    detail::write_huffman_table(bw, 0, 0, dc_luma.bits, dc_luma.huffval);
    detail::write_huffman_table(bw, 1, 0, ac_luma.bits, ac_luma.huffval);
    detail::write_huffman_table(bw, 0, 1, dc_chroma.bits, dc_chroma.huffval);
    detail::write_huffman_table(bw, 1, 1, ac_chroma.bits, ac_chroma.huffval);

    // SOS
    bw.write_byte(0xFF); bw.write_byte(detail::kMarkerSOS);
    bw.write_uint16_be(12); bw.write_byte(3);
    bw.write_byte(1); bw.write_byte(0x00);
    bw.write_byte(2); bw.write_byte(0x11);
    bw.write_byte(3); bw.write_byte(0x11);
    bw.write_byte(0); bw.write_byte(63); bw.write_byte(0);
}

// ============================================================
//  CUDA JPEG Encoder — Optimized Pipeline
//
//  GPU: RGB->YCbCr -> DCT+Quant -> DC diffs -> Huffman Pass1
//       -> CPU Prefix Sum -> Huffman Pass2 -> GPU Byte Stuff
//  CPU: Write JPEG headers + copy stuffed data + EOI
// ============================================================

JpegError process_encode_cuda(const uint8_t* input, uint8_t* output,
                               size_t* output_size,
                               int width, int height, int channels,
                               int bit_depth, const JpegParams& params) {
    (void)channels;
    if (!has_cuda()) return JpegError::CudaNotAvailable;

    JpegError err = validate_jpeg_encode_inputs(input, output, width, height,
                                                 3, bit_depth, params.quality);
    if (err != JpegError::Ok) return err;

    int quality = std::max(1, std::min(100, params.quality));
    int subsample = std::max(0, std::min(2, params.chroma_subsample));
    size_t raw_size = static_cast<size_t>(width) * height * 3;

    if (!t_cuda_ws.ensure(width, height, bit_depth))
        return JpegError::InternalError;

    cudaError_t cuda_err;
    cudaGetLastError(); // clear any stale errors

    // Step 1: Upload input
    cuda_err = cudaMemcpy(t_cuda_ws.d_input, input, raw_size, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "[GPU] cudaMemcpy H2D failed: %s\n", cudaGetErrorString(cuda_err));
        return JpegError::InternalError;
    }

    // Build & upload quantization tables
    int luma_qt[64], chroma_qt[64];
    detail::scale_quant_table(luma_qt, detail::kStdLumaQTable50, quality);
    detail::scale_quant_table(chroma_qt, detail::kStdChromaQTable50, quality);

    cuda_err = cudaMemcpy(t_cuda_ws.d_quant_tables, luma_qt, 64 * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) return JpegError::InternalError;
    cuda_err = cudaMemcpy(t_cuda_ws.d_quant_tables + 64, chroma_qt, 64 * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) return JpegError::InternalError;

    int blocks_x = t_cuda_ws.blocks_x;
    int blocks_y = t_cuda_ws.blocks_y_dim;
    int bpc = t_cuda_ws.blocks_per_channel;
    int padded_w = ((width + 7) / 8) * 8;
    int padded_h = ((height + 7) / 8) * 8;

    // Step 2-4: Launch all GPU kernels in a single wave (no intermediate syncs)
    // Default stream ensures sequential execution order on GPU.
    cuda_jpeg_rgb_to_ycbcr(t_cuda_ws.d_input,
                            t_cuda_ws.d_y_plane, t_cuda_ws.d_cb_plane, t_cuda_ws.d_cr_plane,
                            width, height, bit_depth);

    // DCT+Quantize in MCU order: global_idx = mcu_idx * 3 + component
    cuda_jpeg_fdct_quantize_mcu(t_cuda_ws.d_y_plane, t_cuda_ws.d_quant_coeffs,
                                 t_cuda_ws.d_quant_tables,
                                 padded_w, padded_h, blocks_x, blocks_y, 0);
    cuda_jpeg_fdct_quantize_mcu(t_cuda_ws.d_cb_plane, t_cuda_ws.d_quant_coeffs,
                                 t_cuda_ws.d_quant_tables + 64,
                                 padded_w, padded_h, blocks_x, blocks_y, 1);
    cuda_jpeg_fdct_quantize_mcu(t_cuda_ws.d_cr_plane, t_cuda_ws.d_quant_coeffs,
                                 t_cuda_ws.d_quant_tables + 64,
                                 padded_w, padded_h, blocks_x, blocks_y, 2);

    cuda_huffman_pass1_lengths(t_cuda_ws.d_quant_coeffs,
                                t_cuda_ws.d_block_lengths,
                                t_cuda_ws.total_blocks, bpc, bpc, bpc);

    // GPU parallel prefix sum: lengths → offsets (no CPU round-trip!)
    // d_block_offsets and d_total_bits are allocated in workspace
    cuda_gpu_prefix_sum(t_cuda_ws.d_block_lengths, t_cuda_ws.d_block_offsets,
                         t_cuda_ws.d_block_sums, t_cuda_ws.d_total_bits,
                         t_cuda_ws.total_blocks);

    // Read total bits from GPU (implicit sync after prefix sum; 4 bytes only)
    int total_bits = 0;
    cuda_err = cudaMemcpy(&total_bits, t_cuda_ws.d_total_bits, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) return JpegError::InternalError;

    // Clear bitstream & launch Huffman Pass 2
    size_t bitstream_words = ((size_t)total_bits + 31) / 32 + 1;
    if (bitstream_words > t_cuda_ws.d_bitstream_capacity)
        bitstream_words = t_cuda_ws.d_bitstream_capacity;
    cuda_err = cudaMemset(t_cuda_ws.d_bitstream, 0, bitstream_words * sizeof(uint32_t));
    if (cuda_err != cudaSuccess) return JpegError::InternalError;

    cuda_huffman_pass2_write(t_cuda_ws.d_quant_coeffs,
                              t_cuda_ws.d_block_offsets, t_cuda_ws.d_bitstream,
                              t_cuda_ws.total_blocks, bpc, bpc, bpc);

    // GPU parallel byte stuffing
    {
        int words = (total_bits + 31) / 32;
        int num_threads = 4096;
        int wpt = (words + num_threads - 1) / num_threads;
        if (wpt < 1) wpt = 1;
        cuda_byte_stuff_parallel(
            t_cuda_ws.d_bitstream, t_cuda_ws.d_total_bits,
            t_cuda_ws.d_stuffed, t_cuda_ws.d_block_lengths,
            t_cuda_ws.d_block_offsets, t_cuda_ws.d_block_sums,
            t_cuda_ws.d_total_bits, t_cuda_ws.d_output,
            wpt, num_threads);
    }

    int stuffed_size = 0;
    cuda_err = cudaMemcpy(&stuffed_size, t_cuda_ws.d_total_bits, sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) return JpegError::InternalError;

    size_t max_output = *output_size;
    if (max_output == 0) max_output = get_max_jpeg_size(width, height, 3);

    detail::BitWriter bw(output, max_output);
    write_jpeg_headers(bw, width, height, quality, subsample, luma_qt, chroma_qt);

    if (stuffed_size > 0) {
        std::vector<uint8_t> h_stuffed(stuffed_size);
        cuda_err = cudaMemcpy(h_stuffed.data(), t_cuda_ws.d_output, stuffed_size, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) return JpegError::InternalError;
        bw.write_bytes(h_stuffed.data(), stuffed_size);
    }

    bw.write_byte(0xFF);
    bw.write_byte(detail::kMarkerEOI);
    *output_size = bw.position();
    return JpegError::Ok;
}

// ============================================================
//  CUDA JPEG Decoder — falls back to CPU for now
// ============================================================

JpegError process_decode_cuda(const uint8_t* input, size_t input_size,
                               uint8_t* output,
                               int* width, int* height, int* channels) {
    if (!has_cuda()) return JpegError::CudaNotAvailable;
    if (!input || !output || !width || !height || !channels)
        return JpegError::NullInput;
    if (input_size < 2 || input[0] != 0xFF || input[1] != detail::kMarkerSOI)
        return JpegError::InvalidJpegData;
    return process_decode_baseline(input, input_size, output, width, height, channels);
}

} // namespace jpeg_codec

#else

namespace jpeg_codec {

bool has_cuda() { return false; }
const char* cuda_device_name() { return "N/A"; }
void cuda_synchronize() {}

JpegError process_encode_cuda(const uint8_t*, uint8_t*, size_t*, int, int, int, int,
                               const JpegParams&) {
    return JpegError::CudaNotAvailable;
}

JpegError process_decode_cuda(const uint8_t*, size_t, uint8_t*, int*, int*, int*) {
    return JpegError::CudaNotAvailable;
}

} // namespace jpeg_codec

#endif // IM_OPERATOR_HAS_CUDA
