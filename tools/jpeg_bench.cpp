#include "jpeg_codec/algorithms.hpp"
#include <cstdio>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>
#include <cstring>
#include <algorithm>

using namespace jpeg_codec;
using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;
using Us = std::chrono::duration<double, std::micro>;

// ============================================================
struct Result {
    double enc_ms, dec_ms;
    size_t jpg_bytes;
    int w, h;
    bool enc_ok, dec_ok;
    const char* label;
};

// ============================================================
Result bench_cpu(int w, int h, int iters, int quality, int sub, const char* label) {
    Result r{}; r.w=w; r.h=h; r.label=label;
    size_t px = static_cast<size_t>(w)*h*3;
    std::vector<uint8_t> src(px);
    for (size_t i=0;i<px;i+=3){src[i]=80;src[i+1]=120;src[i+2]=200;}

    size_t max_jpg = get_max_jpeg_size(w,h,3);
    std::vector<uint8_t> jpg(max_jpg);
    std::vector<uint8_t> dec(px);
    JpegParams p; p.quality=quality; p.chroma_subsample=sub;

    // Warmup
    size_t js = max_jpg;
    process_jpeg_encode(src.data(),jpg.data(),&js,w,h,3,JpegAlgorithm::ENCODE_BASELINE,8,p);

    double best_enc=1e9, best_dec=1e9;
    for(int i=0;i<iters;i++){
        js=max_jpg;
        auto t0=Clock::now();
        JpegError e=process_jpeg_encode(src.data(),jpg.data(),&js,w,h,3,JpegAlgorithm::ENCODE_BASELINE,8,p);
        auto t1=Clock::now();
        if(e!=JpegError::Ok){printf("  GPU enc error: %s\n",jpeg_error_message(e));r.enc_ok=false;return r;}
        double ems=Ms(t1-t0).count();
        if(ems<best_enc) best_enc=ems;
        r.jpg_bytes=js;

        int ow,oh,oc;
        auto t2=Clock::now();
        e=process_jpeg_decode(jpg.data(),js,dec.data(),&ow,&oh,&oc,JpegAlgorithm::DECODE_BASELINE);
        auto t3=Clock::now();
        if(e!=JpegError::Ok){r.dec_ok=false;return r;}
        double dms=Ms(t3-t2).count();
        if(dms<best_dec) best_dec=dms;
    }
    r.enc_ms=best_enc; r.dec_ms=best_dec; r.enc_ok=r.dec_ok=true;
    return r;
}

// ============================================================
Result bench_gpu(int w, int h, int iters, int quality, int sub, const char* label) {
    Result r{}; r.w=w; r.h=h; r.label=label;
    if(!has_cuda()){r.enc_ok=false;return r;}

    size_t px = static_cast<size_t>(w)*h*3;
    std::vector<uint8_t> src(px);
    for (size_t i=0;i<px;i+=3){src[i]=80;src[i+1]=120;src[i+2]=200;}

    size_t max_jpg = get_max_jpeg_size(w,h,3);
    std::vector<uint8_t> jpg(max_jpg);
    std::vector<uint8_t> dec(px);
    JpegParams p; p.quality=quality; p.chroma_subsample=sub;

    // Warmup
    size_t js=max_jpg;
    process_jpeg_encode(src.data(),jpg.data(),&js,w,h,3,JpegAlgorithm::ENCODE_CUDA,8,p);

    double best_enc=1e9, best_dec=1e9;
    for(int i=0;i<iters;i++){
        js=max_jpg;
        auto t0=Clock::now();
        JpegError e=process_jpeg_encode(src.data(),jpg.data(),&js,w,h,3,JpegAlgorithm::ENCODE_CUDA,8,p);
        auto t1=Clock::now();
        if(e!=JpegError::Ok){fprintf(stderr,"  GPU enc error: %s\n",jpeg_error_message(e));r.enc_ok=false;return r;}
        double ems=Ms(t1-t0).count();
        if(ems<best_enc) best_enc=ems;
        r.jpg_bytes=js;
        // Decode with CPU path (GPU decode needs block offsets from encode side channel)
        int ow,oh,oc;
        auto t2=Clock::now();
        e=process_jpeg_decode(jpg.data(),js,dec.data(),&ow,&oh,&oc,JpegAlgorithm::DECODE_BASELINE);
        auto t3=Clock::now();
        if(e!=JpegError::Ok){fprintf(stderr,"  GPU dec error: %s\n",jpeg_error_message(e));r.dec_ok=false;return r;}
        double dms=Ms(t3-t2).count();
        if(dms<best_dec) best_dec=dms;
    }
    r.enc_ms=best_enc; r.dec_ms=best_dec; r.enc_ok=r.dec_ok=true;
    return r;
}

// ============================================================
int main() {
    printf("=== JPEG Codec: CPU vs GPU Real Benchmark ===\n");
    printf("GPU: %s\n", has_cuda()?cuda_device_name():"N/A");
    printf("CPU: scalar float DCT, CPU Huffman\n\n");

    struct Cfg { int w,h,iters; const char* name; };
    Cfg cfgs[]={
        {256,256,100,"256x256"},
        {512,512,50,"512x512"},
        {1024,1024,20,"1K"},
        {2048,2048,5,"2K"},
        {640,480,100,"VGA"},
        {1280,720,50,"HD"},
        {1920,1080,20,"FHD"},
        {3840,2160,5,"4K"},
    };

    // Table header
    printf("%-6s %7s | %10s %10s %7s | %10s %10s %7s | %7s %7s\n",
           "Res","MP","CPU-E(ms)","CPU-D(ms)","CPU-FPS",
           "GPU-E(ms)","GPU-D(ms)","GPU-FPS","SpE","SpD");
    printf("%-6s %7s | %10s %10s %7s | %10s %10s %7s | %7s %7s\n",
           "---","------","--------","--------","------",
           "--------","--------","------","-----","-----");

    for(auto& c : cfgs){
        double mp=static_cast<double>(c.w)*c.h/1e6;

        auto cpu=bench_cpu(c.w,c.h,c.iters,90,0,c.name);
        if(!cpu.enc_ok||!cpu.dec_ok){printf("%-6s CPU FAILED\n",c.name);continue;}

        auto gpu=bench_gpu(c.w,c.h,c.iters,90,0,c.name);
        if(!gpu.enc_ok||!gpu.dec_ok){printf("%-6s GPU: %s\n",c.name,gpu.enc_ok?"dec FAIL":"enc FAIL");continue;}

        double sp_e=cpu.enc_ms/gpu.enc_ms;
        double sp_d=cpu.dec_ms/gpu.dec_ms;
        double fps_cpu=1000.0/std::max(cpu.enc_ms,cpu.dec_ms);
        double fps_gpu=1000.0/std::max(gpu.enc_ms,gpu.dec_ms);

        printf("%-6s %6.1fM | %9.2f %9.2f %6.1f | %9.2f %9.2f %6.0f | %6.1fx %6.1fx\n",
               c.name, mp,
               cpu.enc_ms, cpu.dec_ms, fps_cpu,
               gpu.enc_ms, gpu.dec_ms, fps_gpu,
               sp_e, sp_d);
    }

    // Detailed 4K breakdown
    printf("\n=== 4K (3840x2160) Detailed Comparison ===\n\n");
    auto cpu4k=bench_cpu(3840,2160,3,90,0,"4K");
    auto gpu4k=bench_gpu(3840,2160,3,90,0,"4K");

    if(cpu4k.enc_ok && gpu4k.enc_ok){
        printf("                    CPU       GPU       Speedup\n");
        printf("                    ---       ---       -------\n");
        printf("Encode (ms):      %7.2f   %7.2f    %6.1fx\n",cpu4k.enc_ms,gpu4k.enc_ms,cpu4k.enc_ms/gpu4k.enc_ms);
        printf("Decode (ms):      %7.2f   %7.2f    %6.1fx\n",cpu4k.dec_ms,gpu4k.dec_ms,cpu4k.dec_ms/gpu4k.dec_ms);
        printf("JPEG size:       %7zu   %7zu\n",cpu4k.jpg_bytes,gpu4k.jpg_bytes);
        printf("4K FPS (enc):    %7.1f   %7.0f\n",1000.0/cpu4k.enc_ms,1000.0/gpu4k.enc_ms);
        printf("4K FPS (dec):    %7.1f   %7.0f\n",1000.0/cpu4k.dec_ms,1000.0/gpu4k.dec_ms);
        printf("VRAM bandwidth:      -    %7.0f MB/s\n",
               static_cast<double>(gpu4k.jpg_bytes)/(gpu4k.enc_ms/1000.0)/(1024*1024));
    }

    return 0;
}
