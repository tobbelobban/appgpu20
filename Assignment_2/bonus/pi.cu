
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>


__global__ void iteration_kernel(int* counts, curandState* states, const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        double x,y;
        x = curand_uniform(&states[tid]);
        y = curand_uniform(&states[tid]);
        if(sqrt((x*x)+(y*y)) <= 1.0) counts[tid] += 1;
    }
}

__global__ void rand_init_kernel(curandState *states, int n) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) curand_init(tid, tid, 0, &states[tid]);
}

int main(int argc, char* argv[]) {
    
    int NUM_ITERATIONS = 100000000;
    int NUM_THREADS = 10000;
    int BLOCK_SIZE = 1024;

    curandState *dev_random;
    cudaMalloc((void**)&dev_random, NUM_THREADS*sizeof(curandState));
    
    dim3 grid_dim((NUM_THREADS+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    dim3 block_dim(BLOCK_SIZE, 1, 1);

    rand_init_kernel<<<grid_dim,block_dim>>>(dev_random, NUM_THREADS);

    int* dev_counts;
    cudaMalloc((void**)&dev_counts, NUM_THREADS*sizeof(int));
    cudaMemset(dev_counts, 0, NUM_THREADS*sizeof(int));

    int NUM_ITS_PER_THREAD = NUM_ITERATIONS / NUM_THREADS;
    for(int it = 0; it < NUM_ITS_PER_THREAD; it++) {
        iteration_kernel<<<grid_dim,block_dim>>>(dev_counts, dev_random, NUM_THREADS);   
    }
    cudaDeviceSynchronize();
    int* res = (int*)malloc(sizeof(int)*NUM_THREADS);
    cudaMemcpy(res, dev_counts, sizeof(int)*NUM_THREADS, cudaMemcpyDeviceToHost);    
    cudaFree(dev_counts);
    cudaFree(dev_random);

    int count = 0;
    for(int i = 0; i < NUM_THREADS; i++) {
        count += res[i];
    }

    printf("PI = %f\n", 4*(double)count/(double)NUM_ITERATIONS);
    return 0;
}