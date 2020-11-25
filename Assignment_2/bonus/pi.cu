
/*
    KTH, APPLIED GPU PROGRAMMING DD2360
    This code is written by Tobias Edwards
    Part of course assignment 2
    compile with: nvcc -arch=sm_XX
*/

#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>
#include <getopt.h>

__global__ void iteration_kernel(int* counts, curandState* states, const int num_its, const int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        double x,y;
        for(int i = 0; i < num_its; i++) {
            x = curand_uniform(&states[tid]);
            y = curand_uniform(&states[tid]);
            if(sqrt((x*x)+(y*y)) <= 1.0) counts[tid] += 1;
        }
    }
}

__global__ void rand_init_kernel(curandState *states, int n) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) curand_init(tid, tid, 0, &states[tid]);
}

int main(int argc, char* argv[]) {
    
    int NUM_ITERATIONS = 100000000;
    int NUM_THREADS = 100000;
    int BLOCK_SIZE = 1024;

    // process program arguments
    int option, tmp;
    while((option = getopt(argc, argv, ":i:t:b:")) != -1){ 
        switch(option) { 
            case 'b':   // BLOCK_SIZE
                tmp = atoi(optarg);
                BLOCK_SIZE = tmp > 0 ? tmp : BLOCK_SIZE;
                break;
            case 'i':   // NUM_ITERATIONS
                tmp = atoi(optarg);
                NUM_ITERATIONS = tmp > 0 ? tmp : NUM_ITERATIONS;
                break;
            case 't':   // NUM_THREADS
                tmp = atoi(optarg);
                NUM_THREADS = tmp > 0 ? tmp : NUM_THREADS;
                break;
            default:
                printf("Unrecognized flag.\n");
        }
    }
    
    // handle case its < threads
    if(NUM_ITERATIONS < NUM_THREADS) {
        NUM_THREADS = NUM_ITERATIONS;
    }
    int NUM_ITS_PER_THREAD = NUM_ITERATIONS / NUM_THREADS;

    // prepare rng for each thread
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, NUM_THREADS*sizeof(curandState));
    
    // setup grid and block dimensions
    dim3 grid_dim((NUM_THREADS+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    dim3 block_dim(BLOCK_SIZE, 1, 1);

    // initialize random number streams
    rand_init_kernel<<<grid_dim,block_dim>>>(dev_random, NUM_THREADS);

    // allocate memory on device for computation
    int* dev_counts;
    cudaMalloc((void**)&dev_counts, NUM_THREADS*sizeof(int));
    cudaMemset(dev_counts, 0, NUM_THREADS*sizeof(int));
    
    // launch kernel
    iteration_kernel<<<grid_dim,block_dim>>>(dev_counts, dev_random, NUM_ITS_PER_THREAD, NUM_THREADS);   
    
    // synchronize threads
    cudaDeviceSynchronize();
    
    // collect result from device
    int* res = (int*)malloc(sizeof(int)*NUM_THREADS);
    cudaMemcpy(res, dev_counts, sizeof(int)*NUM_THREADS, cudaMemcpyDeviceToHost);    
    cudaFree(dev_counts);
    cudaFree(dev_random);

    // sum the results
    int count = 0;
    for(int i = 0; i < NUM_THREADS; i++) {
        count += res[i];
    }

    // print result
    printf("--RESULTS--\n");
    printf("Number of blocks: %i\n", grid_dim.x);
    printf("Block size: %i\n", block_dim.x);
    printf("Iterations: %i\n", NUM_ITERATIONS);
    printf("Iterations per thread: %i\n", NUM_ITS_PER_THREAD);
    printf("PI = %f\n", 4*(double)count/(double)NUM_ITERATIONS);
    return 0;
}