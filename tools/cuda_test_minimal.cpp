#include "jpeg_codec/algorithms.hpp"
#include <cstdio>
int main() {
    printf("Starting...\n");
    bool cuda = jpeg_codec::has_cuda();
    printf("has_cuda: %s\n", cuda ? "YES" : "NO");
    if (cuda) printf("Device: %s\n", jpeg_codec::cuda_device_name());
    printf("Done.\n");
    return 0;
}
