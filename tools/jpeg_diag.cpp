#include "jpeg_codec/algorithms.hpp"
#include <cstdio>
#include <vector>

using namespace jpeg_codec;

int main() {
    // Test gradient with different subsampling
    for (int sub : {0, 1, 2}) {
        const char* subname = (sub==0)?"4:4:4":(sub==1?"4:2:0":"4:2:2");
        printf("=== Subsampling %s ===\n", subname);
        for (int sz : {16, 24, 32, 40, 48, 56, 64}) {
            int mw = sz, mh = sz;
            std::vector<uint8_t> msrc(static_cast<size_t>(mw) * mh * 3);
            for (int y = 0; y < mh; y++) {
                for (int x = 0; x < mw; x++) {
                    size_t idx = (static_cast<size_t>(y) * mw + x) * 3;
                    msrc[idx + 0] = static_cast<uint8_t>((x * 255) / (mw - 1));
                    msrc[idx + 1] = static_cast<uint8_t>((y * 255) / (mh - 1));
                    msrc[idx + 2] = static_cast<uint8_t>(128);
                }
            }
            size_t mxs = get_max_jpeg_size(mw, mh, 3);
            std::vector<uint8_t> mjpg(mxs);
            size_t mjs = mxs;
            JpegParams params;
            params.quality = 95;
            params.chroma_subsample = sub;
            JpegError e = process_jpeg_encode(msrc.data(), mjpg.data(), &mjs, mw, mh, 3,
                                               JpegAlgorithm::ENCODE_BASELINE, 8, params);
            if (e != JpegError::Ok) { printf("  %dx%d: encode FAILED\n", mw, mh); continue; }

            std::vector<uint8_t> mdec(static_cast<size_t>(mw) * mh * 3);
            int ow, oh, oc;
            e = process_jpeg_decode(mjpg.data(), mjs, mdec.data(), &ow, &oh, &oc,
                                     JpegAlgorithm::DECODE_BASELINE);
            printf("  %dx%d: %s (%zu bytes)\n", mw, mh, jpeg_error_message(e), mjs);
        }
    }
    return 0;
}
