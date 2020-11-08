/*
    KTH, APPLIED GPU PROGRAMMING DD2360
    This code is written by Tobias Edwards
    Part of course assignment 2
    compile with: nvcc -arch=sm_XX
*/


#include <stdio.h>
#include <sys/time.h>

#define TPB 256
#define ARRAY_SIZE 100000000
#define MARGIN 1e-4

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }

__global__ void SAXPY_kernel(const float a, const float* x, float* y) {
    // compute threads id
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ARRAY_SIZE) {
        // perform thread-local computation
        y[tid] = a * x[tid] + y[tid];
    }
}

void device_SAXPY(const float a, const float* x, const float* y, float* c, const int n, double* times) {

    // allocate memory on device
    float* d_x, *d_y;
    cudaMalloc((void**)&d_x, sizeof(float)*n);
    cudaMalloc((void**)&d_y, sizeof(float)*n);

    // copy data from host to device
    cudaMemcpy(d_y, y, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(float)*n, cudaMemcpyHostToDevice);
    
    // perform computation on device
    times[0] = cpuSecond();
    SAXPY_kernel<<<(ARRAY_SIZE+TPB-1)/TPB,TPB>>>(a, d_x, d_y);
    times[1] = cpuSecond();
    
    // wait for all threads to complete
    cudaDeviceSynchronize();

    // copy result from device to host
    cudaMemcpy(c, d_y, sizeof(float)*n, cudaMemcpyDeviceToHost);

    // free allocated memory on device
    cudaFree(d_x);
    cudaFree(d_y);
}

void host_SAXPY(const float a, const float* x, const float* y, float* c, const int n, double* times) {
    times[0] = cpuSecond();
    for(int i = 0; i < n; i++) {
        c[i] = a * x[i] + y[i];
    }
    times[1] = cpuSecond();
}

bool compare(const float* res1, const float* res2, const int n) {
    for(int i = 0; i < n; i++) {
        if(abs(res1[i] - res2[i]) > MARGIN) {
            printf("Error (i = %i): %f != %f\n",i, res1[i], res2[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    double host_total_start, host_total_end, dev_total_start, dev_total_end;
    double host_compute_times[2];
    double dev_comute_times[2];

    float a = 3., max_rand = 50.;
    float* x, * y;
    int n = ARRAY_SIZE;


    // initialize vectors x and y in : c = a * x + y
    x = (float*)malloc(sizeof(float)*n);
    y = (float*)malloc(sizeof(float)*n);
    for(int i = 0; i < n; i++) {
        x[i] = (float)rand()/(float)(RAND_MAX/max_rand);
        y[i] = (float)rand()/(float)(RAND_MAX/max_rand);
    }

    // vectors for results
    float* host_c, * dev_c;
    host_c = (float*)malloc(sizeof(float)*n);
    dev_c = (float*)malloc(sizeof(float)*n);
    
    // perform computation on host
    printf("Starting SAXPY on host...");
    host_total_start = cpuSecond();
    host_SAXPY(a, x, y, host_c, n, host_compute_times);
    host_total_end = cpuSecond();
    printf(" Done!\n");

    // perform computation on device
    printf("Starting SAXPY on device...");
    dev_total_start = cpuSecond();
    device_SAXPY(a, x, y, dev_c, n, dev_comute_times);
    dev_total_end = cpuSecond();
    printf(" Done!\n");

    printf("\n");

    // check results
    printf("Comparing results... (Error margin = %f)\n", MARGIN);
    bool res = compare(host_c, dev_c, n);
    printf("The results are%s the same.\n", (res? "":" not") );
    
    printf("\n");

    //print times
    printf("--EXECUTION TIMES--\n");
    printf("Total time on device: %fs\n", dev_total_end-dev_total_start);
    printf("Total time on host: %fs\n", host_total_end-host_total_start);
    printf("Compute time on device: %fs\n", dev_comute_times[1]-dev_comute_times[0]);
    printf("Compute time on host: %fs\n", host_compute_times[1]-host_compute_times[0]);

    printf("\n");

    // free memory allocation on host
    free(x); free(y);
    free(host_c); free(dev_c);
    
    return 0;
}