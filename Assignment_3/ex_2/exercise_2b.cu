
/*
    KTH, APPLIED GPU PROGRAMMING DD2360
    This code is written by Tobias Edwards
    Part of course assignment 3
    compile with: nvcc -arch=sm_XX
*/

#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <math.h>
#include <sys/time.h>


#define err_margin 1e-4

typedef struct {
    float3 pos;
    float3 vel;
} Particle;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }

__global__ void step_kernel(Particle* p, const float v_factor, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        // update velocity
        p[tid].vel.x *= v_factor != 0.0 ? v_factor : 1;
        p[tid].vel.y *=  v_factor != 0.0 ? v_factor : 1;
        p[tid].vel.z *=  v_factor != 0.0 ? v_factor : 1;

        //update position
        p[tid].pos.x += p[tid].vel.x;
        p[tid].pos.y += p[tid].vel.y;
        p[tid].pos.z += p[tid].vel.z;
    }
}

void step_host(Particle* p, const float v_factor, const int n) {
    for(int i = 0; i < n; i++) {
            // update velocity
            p[i].vel.x *= v_factor != 0.0 ? v_factor : 1;
            p[i].vel.y *=  v_factor != 0.0 ? v_factor : 1;
            p[i].vel.z *=  v_factor != 0.0 ? v_factor : 1;
    
            //update position
            p[i].pos.x += p[i].vel.x;
            p[i].pos.y += p[i].vel.y;
            p[i].pos.z += p[i].vel.z;
    }
}

void init_particles(Particle* parts, const int n) {
    // initialize particles
    float px, py, pz, vx, vy, vz;
    for(int i = 0; i < n; i++) {
        // random values in range [-1,1]
        px = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        py = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        pz = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        vx = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        vy = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        vz = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        parts[i] = {{px, py, pz}, {vx, vy, vz}};          
    }
}

// Euclidean distance between p1 and p2
float distance_between_pts(const Particle* p1, const Particle* p2) {
    return sqrt(pow(p1->pos.x-p2->pos.x,2) + pow(p1->pos.y-p2->pos.y,2) + pow(p1->pos.z-p2->pos.z,2));
}

bool compare(const Particle* res1, const Particle* res2, const int n) {
    float dist;
    for(int i = 0; i < n; i++) {
        dist = distance_between_pts(&res1[i],&res2[i]);
        if(dist > err_margin) {
            printf("Fail (i = %i). Distance: %f\n", i, dist);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    int NUM_PARTICLES = 10000, NUM_ITERATIONS = 1000, BLOCK_SIZE = 256;
    int option, tmp;
    double host_start, host_end, dev_start, dev_end, host_time = 0, dev_time = 0;
    bool host = 1, dev = 1;

    // process program arguments
    while((option = getopt(argc, argv, ":p:i:b:hd")) != -1){ 
        switch(option) {
            case 'p':   // NUM_PARTICLES
                tmp = atoi(optarg);
                NUM_PARTICLES = tmp > 0 ? tmp : NUM_PARTICLES;
                
                break;
            case 'b':   // BLOCK_SIZE
                tmp = atoi(optarg);
                BLOCK_SIZE = tmp > 0 ? tmp : BLOCK_SIZE;
                break;
            case 'i':   // NUM_ITERATIONS
                tmp = atoi(optarg);
                NUM_ITERATIONS = tmp >= 0 ? tmp : NUM_ITERATIONS;
                break;
            case 'h':   // no HOST execution
                host = 0;
                break;
            case 'd':   // no DEV execution
                dev = 0;
                break;
            default:
                printf("Unrecognized flag.\n");
        }
     }

    // create particles
    printf("Creating particles...");
    double part_time = cpuSecond();
    Particle* p_h, *p_d;    
    p_h = (Particle*)malloc(sizeof(Particle)*NUM_PARTICLES);
    init_particles(p_h,NUM_PARTICLES);
    if(dev) {
        cudaMallocManaged(&p_d, NUM_PARTICLES*sizeof(Particle)); // allocate managed memory
        memcpy(p_d, p_h, sizeof(Particle)*NUM_PARTICLES);
    }

    part_time = cpuSecond() - part_time;
    printf(" Done! Time = %f s\n", part_time);

    // setup grid and block sizes
    dim3 dim_grid((NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
    dim3 dim_block(BLOCK_SIZE, 1, 1);
    
    // perform computations
    float v_factor;
    printf("Starting computations, please wait...");
    fflush(stdout);
    for(int it = 0; it < NUM_ITERATIONS; it++) {
        v_factor = -1.0 + 2.0 * (float)rand()/RAND_MAX; // v_factor in [-1,1]
        if(host) {
            host_start = cpuSecond();
            step_host(p_h, v_factor, NUM_PARTICLES);
            host_end = cpuSecond();
            host_time += host_end - host_start;
        }
        if(dev) {
            dev_start = cpuSecond();
            step_kernel<<<dim_grid,dim_block>>>(p_d, v_factor, NUM_PARTICLES);
            cudaDeviceSynchronize();
            dev_end = cpuSecond();
            dev_time += dev_end - dev_start;
        }
    }
    printf(" Done!\n");
    
    // check solutions
    bool res;
    if(dev && host) {
        printf("Comparing results...");
        res = compare(p_h, p_d, NUM_PARTICLES);
        printf(" Done! %s\n", res ? "CORRECT :)" : "INCORRECT :(");
    }

    // free allocated memory on device
    if(dev) cudaFree(p_d);
    if(host) free(p_h);
    
    // print timing results
    printf("\n--SETUP--\n");
    printf("NUM_ITERATIONS: %i\n", NUM_ITERATIONS);
    printf("NUM_PARTICLES: %i\n", NUM_PARTICLES);

    if(host) {
        printf("\n--HOST--\n");
        printf("Avg. execution time: %f s\n", host_time/(double)NUM_ITERATIONS);
    }
    if(dev) {
        printf("\n--DEVICE--\n");
        printf("Number of blocks: %i\n", dim_grid.x);
        printf("Block size: %i\n", dim_block.x);
        printf("Avg. execution time: %f s\n", dev_time/(double)NUM_ITERATIONS);
    }
    
    return 0;
}