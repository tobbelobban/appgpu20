
// reminder: use -arch=sm_52 for my gpu
// add CUDA11.1 to PATH: export PATH=/usr/local/cuda-11.1/bin/${PATH:+:${PATH}}

#include <stdio.h>

#define N 256
#define TPB 32

__global__ void hello_kernel() {
    const int tid = blockIdx.y*blockDim.x + threadIdx.x;
    printf("Hello world! from thread %i\n", tid);
}

int main(int argc, char* argv[]) {
    hello_kernel<<<N/TPB,TPB>>>();
    cudaDeviceSynchronize();
}