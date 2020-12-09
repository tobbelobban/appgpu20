#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>

typedef struct {
    cl_float3 p;
    cl_float3 v;
} Particle;

#define MARGIN 1e-4

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) {fprintf(stderr,"Error (code=%i): %s\n",err, clGetErrorString(err));}

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

const char *particle_struct_string = "typedef struct {float3 p; float3 v;} Particle;";

const char *particle_step_kernel = 
"__kernel                                                                   \n"
"void dev_particle_step (__global Particle *p, float factor, int size)      \n"
"{ const int index = get_global_id(0);                                      \n"
"   if(index < size) {                                                      \n"
"       if(factor == 0.0) factor = 1.0;                                     \n"
"       p[index].v.x *= factor;                                             \n"
"       p[index].v.y *= factor;                                             \n"
"       p[index].v.z *= factor;                                             \n"
"       p[index].p.x += p[index].v.x;                                       \n"
"       p[index].p.y += p[index].v.y;                                       \n"
"       p[index].p.z += p[index].v.z;                                       \n"
"   }                                                                       \n"
"}";


// Euclidean distance between p1 and p2
float distance_between_pts(const Particle* p1, const Particle* p2) {
    return sqrt(pow(p1->p.x-p2->p.x,2) + pow(p1->p.y-p2->p.y,2) + pow(p1->p.z-p2->p.z,2));
}

// compare solutions between host and device
int compare_solutions(const Particle* res1, const Particle* res2, const int n) {
    float dist;
    for(int i = 0; i < n; i++) {
        dist = distance_between_pts(&res1[i],&res2[i]);
        if(dist > MARGIN) {
            printf("Fail (i = %i). Distance: %f\n", i, dist);
            return 0;
        }
    }
    return 1;
}
void host_particle_step(Particle* p, float factor, int size) {
    for(int i = 0; i < size; i++) {
            // update velocity
            if(factor == 0.0) factor = 1.0;
            p[i].v.x *= factor;
            p[i].v.y *= factor;
            p[i].v.z *= factor;
    
            //update position
            p[i].p.x += p[i].v.x;
            p[i].p.y += p[i].v.y;
            p[i].p.z += p[i].v.z;
    }
}

void init_particles(Particle *p, int size) {
    // initialize particles
    float px, py, pz, vx, vy, vz;
    for(int i = 0; i < size; ++i) {
        // random values in range [-1,1]
        px = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        py = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        pz = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        vx = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        vy = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        vz = -1.0 + 2.0 * (float)rand()/RAND_MAX;
        p[i] = (Particle){{px, py, pz}, {vx, vy, vz}};          
    }
}

int main(int argc, char *argv) {
    double time_dev, time_host;
    int NUM_PARTICLES = 10000, NUM_ITERATIONS = 1000, BLOCK_SIZE = 256;

    cl_platform_id * platforms; cl_uint     n_platform;

    // Find OpenCL Platforms
    cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

    // Find and sort devices
    cl_device_id *device_list; cl_uint n_devices;
    err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
    err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);

    // Create and initialize an OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err); 

    // initialize particles
    Particle *p;
    int BYTES = NUM_PARTICLES*sizeof(Particle);
    p = (Particle*)malloc(BYTES);
    init_particles(p, NUM_PARTICLES);
    float factor;

    // allocate buffers on device
    cl_mem p_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, BYTES, NULL, &err); CHK_ERROR(err);

    // copy data from host to device
    err = clEnqueueWriteBuffer(cmd_queue, p_dev, CL_TRUE, 0, BYTES, p, 0, NULL, NULL); CHK_ERROR(err);
    
    // compile the kernel    
    const char* strings[2];
    strings[0] = particle_struct_string;
    strings[1] = particle_step_kernel;

    cl_program program = clCreateProgramWithSource(context, 2, strings, NULL, &err); CHK_ERROR(err);
    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL); CHK_ERROR(err);
    if(err != CL_SUCCESS) 
    {
        char buffer[2048]; size_t len;
        clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n",buffer);
        exit(0);
    }

    cl_kernel kernel = clCreateKernel(program, "dev_particle_step", &err); CHK_ERROR(err);
    
    // set kernel work items/groups
    size_t workgroup_size = BLOCK_SIZE;
    size_t work_items = ((NUM_PARTICLES+workgroup_size-1)/workgroup_size)*workgroup_size;
    
    // perform iterations
    for(int i = 0; i < NUM_ITERATIONS; ++i) 
    {
        // update change factor
        factor = -1.0 + 2.0 * (float)rand()/RAND_MAX;

        // perform step on host
        host_particle_step(p, factor, NUM_PARTICLES);
        
        // set kernel arguments 
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &p_dev); CHK_ERROR(err);
        err = clSetKernelArg(kernel, 1, sizeof(float), (void*) &factor); CHK_ERROR(err);
        err = clSetKernelArg(kernel, 2, sizeof(int), (void*) &NUM_PARTICLES); CHK_ERROR(err);
        
        // perform step on device
        err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &work_items, &workgroup_size, 0, NULL, NULL); CHK_ERROR(err);    
        clFinish(cmd_queue);
    }
    
    // copy results back to device
    Particle *res_dev = (Particle*)malloc(BYTES);
    err = clEnqueueReadBuffer(cmd_queue, p_dev, CL_TRUE, 0, BYTES, res_dev, 0, NULL, NULL); CHK_ERROR(err);
    
    // ensure all cmds to device are done
    err = clFlush(cmd_queue); CHK_ERROR(err);
    err = clFinish(cmd_queue); CHK_ERROR(err);

    // Finally, release all that we have allocated.
    err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
    err = clReleaseContext(context);CHK_ERROR(err);
    err = clReleaseMemObject(p_dev); CHK_ERROR(err);
    free(platforms);
    free(device_list);

    // compare solutions
    int same = compare_solutions(res_dev, p, NUM_PARTICLES);
    printf("Same solutions? %i\n", same);

    // free allocated memory on host
    free(p); free(res_dev);

    printf("Time GPU: %f s\n", time_dev);
    printf("Time CPU: %f s\n", time_host);
    return 0;
}



// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
  switch (errorCode) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -71: return "CL_INVALID_SPEC_ID";
  case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
  case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
  case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
  case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
  case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
  case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
  case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
  case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
  case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
  case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
  case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
  case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
  case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
  case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
  case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
  case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
  case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
  case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
  default: return "CL_UNKNOWN_ERROR";
  }
}