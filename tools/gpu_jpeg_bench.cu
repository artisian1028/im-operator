// Standalone GPU JPEG kernel benchmark - real hardware measurement
// Compile: nvcc -std=c++17 -O2 -arch=sm_89 gpu_jpeg_bench.cu -o gpu_jpeg_bench.exe

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(c) do { \
    cudaError_t e = c; \
    if (e != cudaSuccess) { printf("CUDA error at %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

// Kernel 1: RGB -> YCbCr
__global__ void krgb2ycbcr(const uint8_t* rgb, float* y, float* cb, float* cr, int w, int h) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, yi=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||yi>=h)return;
    int i=(yi*w+x)*3, p=yi*w+x;
    float r=rgb[i],g=rgb[i+1],b=rgb[i+2];
    y[p]=0.299f*r+0.587f*g+0.114f*b;
    cb[p]=-0.168736f*r-0.331264f*g+0.5f*b+128.0f;
    cr[p]=0.5f*r-0.418688f*g-0.081312f*b+128.0f;
}

// Kernel 2: 2D DCT on 8x8 blocks
__global__ void kdct(float* blocks, int nb_x, int nb_y, int iw, int ih) {
    int bx=blockIdx.x,by=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
    if(bx>=nb_x||by>=nb_y)return;
    __shared__ float sd[64];
    int px=min(bx*8+tx,iw-1),py=min(by*8+ty,ih-1);
    sd[ty*8+tx]=blocks[py*iw+px]-128.0f;
    __syncthreads();
    float sum=0.0f;
    for(int y=0;y<8;y++) for(int x=0;x<8;x++)
        sum+=sd[y*8+x]*__cosf((2*x+1)*tx*3.14159265f/16)*__cosf((2*y+1)*ty*3.14159265f/16);
    float s1=(tx==0)?0.35355339f:0.5f, s2=(ty==0)?0.35355339f:0.5f;
    blocks[(by*nb_x+bx)*64+ty*8+tx]=sum*s1*s2*0.25f;
}

// Kernel 3: Quantization
__global__ void kquant(float* blocks, const int* qt, int total) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total*64)return;
    int c=i%64;
    blocks[i]=roundf(blocks[i]/qt[c]);
}

// Kernel 4: Dequantization
__global__ void kdequant(float* blocks, const int* qt, int total) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total*64)return;
    int c=i%64;
    blocks[i]=blocks[i]*qt[c];
}

// Kernel 5: 2D IDCT on 8x8 blocks (same structure as DCT, different formula)
__global__ void kidct(float* blocks, int nb_x, int nb_y, int iw, int ih) {
    int bx=blockIdx.x,by=blockIdx.y,tx=threadIdx.x,ty=threadIdx.y;
    if(bx>=nb_x||by>=nb_y)return;
    __shared__ float tmp[64];
    tmp[ty*8+tx]=blocks[(by*nb_x+bx)*64+ty*8+tx];
    __syncthreads();
    float sum=0.0f;
    for(int v=0;v<8;v++) for(int u=0;u<8;u++) {
        float s1=(u==0)?0.35355339f:0.5f, s2=(v==0)?0.35355339f:0.5f;
        sum+=tmp[v*8+u]*s1*s2*__cosf((2*tx+1)*u*3.14159265f/16)*__cosf((2*ty+1)*v*3.14159265f/16);
    }
    sum=sum*0.25f+128.0f;
    int px=bx*8+tx,py=by*8+ty;
    if(px<iw&&py<ih) blocks[py*iw+px]=sum;
}

// Kernel 6: YCbCr -> RGB
__global__ void kycbcr2rgb(const float* y, const float* cb, const float* cr, uint8_t* rgb, int w, int h) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, yi=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||yi>=h)return;
    int p=yi*w+x,i=(yi*w+x)*3;
    float yv=y[p],cbv=cb[p]-128,crv=cr[p]-128;
    float r=yv+1.402f*crv,g=yv-0.344136f*cbv-0.714136f*crv,b=yv+1.772f*cbv;
    r=fminf(fmaxf(r,0),255); g=fminf(fmaxf(g,0),255); b=fminf(fmaxf(b,0),255);
    rgb[i]=(uint8_t)(r+0.5f); rgb[i+1]=(uint8_t)(g+0.5f); rgb[i+2]=(uint8_t)(b+0.5f);
}

double time_kernel(int iters, int warmup) {
    cudaEvent_t s,t;
    cudaEventCreate(&s); cudaEventCreate(&t);
    for(int i=0;i<warmup;i++){cudaEventRecord(s);cudaEventRecord(t);cudaEventSynchronize(t);}
    cudaEventRecord(s);
    for(int i=0;i<iters;i++){cudaEventRecord(t);cudaEventSynchronize(t);}
    // Wait for all to finish
    cudaDeviceSynchronize();
    cudaEventRecord(t);
    cudaEventSynchronize(t);
    float ms;
    cudaEventElapsedTime(&ms,s,t);
    cudaEventDestroy(s); cudaEventDestroy(t);
    return ms*1000.0/iters; // us per iteration
}

int main() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    printf("=== GPU JPEG Kernel Benchmark (real HW) ===\nGPU: %s (%d SMs, %.0f GB/s VRAM)\n\n",p.name,p.multiProcessorCount,
           p.memoryClockRate*2.0*(p.memoryBusWidth/8)/1e6);

    struct{int w,h;const char* n;} cfgs[]={
        {640,480,"VGA"},{1280,720,"HD"},{1920,1080,"FHD"},{3840,2160,"4K"},{7680,4320,"8K"}
    };

    printf("%-6s %7s | %8s %8s %8s %8s %8s %8s | %8s %8s %8s\n",
           "Res","MP","RGB2YCC","DCT","Quant","Dequant","IDCT","YCC2RGB","Enc(us)","Dec(us)","FPS");
    printf("%-6s %7s | %8s %8s %8s %8s %8s %8s | %8s %8s %8s\n",
           "---","------","------","------","------","------","------","------","------","------","---");

    for(auto& c:cfgs){
        double mp=(double)c.w*c.h/1e6;
        size_t rb=(size_t)c.w*c.h*3, fb=(size_t)c.w*c.h*4;
        int pw=((c.w+7)/8)*8, ph=((c.h+7)/8)*8;
        int bx=pw/8, by=ph/8, tb=bx*by*3;

        uint8_t *drgb,*drgb2; float *dy,*dcb,*dcr,*dblk; int *dqt;
        CUDA_CHECK(cudaMalloc(&drgb,rb)); CUDA_CHECK(cudaMalloc(&drgb2,rb));
        CUDA_CHECK(cudaMalloc(&dy,fb)); CUDA_CHECK(cudaMalloc(&dcb,fb)); CUDA_CHECK(cudaMalloc(&dcr,fb));
        CUDA_CHECK(cudaMalloc(&dblk,(size_t)tb*64*4)); CUDA_CHECK(cudaMalloc(&dqt,64*4));

        uint8_t* hr=new uint8_t[rb];
        for(size_t i=0;i<rb;i+=3){hr[i]=80;hr[i+1]=120;hr[i+2]=200;}
        CUDA_CHECK(cudaMemcpy(drgb,hr,rb,cudaMemcpyHostToDevice));
        int hq[64]; for(int i=0;i<64;i++) hq[i]=1;
        CUDA_CHECK(cudaMemcpy(dqt,hq,256,cudaMemcpyHostToDevice));
        delete[] hr;

        dim3 b16(16,16), g16((c.w+15)/16,(c.h+15)/16);
        dim3 b8(8,8), g8(bx,by);
        dim3 b256(256), gq((tb*64+255)/256);

        int wu=2, it=5;

        // Measure each kernel
        cudaEvent_t st,sp;
        cudaEventCreate(&st); cudaEventCreate(&sp);

        // RGB->YCbCr
        for(int i=0;i<wu;i++) krgb2ycbcr<<<g16,b16>>>(drgb,dy,dcb,dcr,c.w,c.h);
        cudaEventRecord(st); for(int i=0;i<it;i++) krgb2ycbcr<<<g16,b16>>>(drgb,dy,dcb,dcr,c.w,c.h);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float t1; cudaEventElapsedTime(&t1,st,sp); t1/=it;

        // DCT (Y plane)
        for(int i=0;i<wu;i++) kdct<<<g8,b8>>>(dy,bx,by,c.w,c.h);
        cudaEventRecord(st); for(int i=0;i<it;i++) kdct<<<g8,b8>>>(dy,bx,by,c.w,c.h);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float td; cudaEventElapsedTime(&td,st,sp); td/=it;

        // Quant
        for(int i=0;i<wu;i++) kquant<<<gq,b256>>>(dblk,dqt,tb);
        cudaEventRecord(st); for(int i=0;i<it;i++) kquant<<<gq,b256>>>(dblk,dqt,tb);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float tq; cudaEventElapsedTime(&tq,st,sp); tq/=it;

        // Dequant
        for(int i=0;i<wu;i++) kdequant<<<gq,b256>>>(dblk,dqt,tb);
        cudaEventRecord(st); for(int i=0;i<it;i++) kdequant<<<gq,b256>>>(dblk,dqt,tb);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float tdq; cudaEventElapsedTime(&tdq,st,sp); tdq/=it;

        // IDCT
        for(int i=0;i<wu;i++) kidct<<<g8,b8>>>(dy,bx,by,c.w,c.h);
        cudaEventRecord(st); for(int i=0;i<it;i++) kidct<<<g8,b8>>>(dy,bx,by,c.w,c.h);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float ti; cudaEventElapsedTime(&ti,st,sp); ti/=it;

        // YCbCr->RGB
        for(int i=0;i<wu;i++) kycbcr2rgb<<<g16,b16>>>(dy,dcb,dcr,drgb2,c.w,c.h);
        cudaEventRecord(st); for(int i=0;i<it;i++) kycbcr2rgb<<<g16,b16>>>(dy,dcb,dcr,drgb2,c.w,c.h);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float t2; cudaEventElapsedTime(&t2,st,sp); t2/=it;

        cudaEventDestroy(st); cudaEventDestroy(sp);

        double enc_us = t1*1000 + td*1000*3 + tq*1000*3;
        double dec_us = tdq*1000*3 + ti*1000*3 + t2*1000;
        double fps = 1e6 / std::max(enc_us, dec_us);

        printf("%-6s %6.1fM | %7.0f %7.0f %7.0f %7.0f %7.0f %7.0f | %8.0f %8.0f %7.0f\n",
               c.n,mp,t1*1000,td*1000,tq*1000,tdq*1000,ti*1000,t2*1000,enc_us,dec_us,fps);

        CUDA_CHECK(cudaFree(drgb)); CUDA_CHECK(cudaFree(drgb2));
        CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dcb)); CUDA_CHECK(cudaFree(dcr));
        CUDA_CHECK(cudaFree(dblk)); CUDA_CHECK(cudaFree(dqt));
    }

    printf("\nEnc(us) = RGB2YCbCr + DCT*3 + Quant*3  (GPU-only, no PCIe, no Huffman)\n");
    printf("Dec(us) = Dequant*3 + IDCT*3 + YCbCr2RGB\n");
    return 0;
}
