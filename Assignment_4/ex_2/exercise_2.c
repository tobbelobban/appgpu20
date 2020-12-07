// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <sys/time.h>

#define ARRAY_SIZE 100000000
#define MARGIN 0.00001

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


const char *SAXPY_kernel = 
"__kernel                                                                         \n"
"void device_SAXPY (float a, __global float *x, __global float *y, int size)      \n"
"{ const int index = get_global_id(0);                                            \n"
"   if(index < size) y[index] = a * x[index] + y[index];}";

void SAXPY_cpu(const float a, const float *x, float *y) {
    for(int i = 0; i < ARRAY_SIZE; ++i) y[i] = a * x[i] + y[i];
}

int compare_solutions(const float *res1, const float *res2) {
    for(int i = 0; i < ARRAY_SIZE; ++i) {
        if(fabs(res1[i] - res2[i]) > MARGIN) {
            printf("Error (i == %i): %f != %f\n",i, res1[i], res2[i]);
            return 0;
        }
    }
        
    return 1;
}

int main(int argc, char *argv) {
    double time_dev, time_host;

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

    // initialize alpha, x and y
    float *x, *y, *dev_res;
    int array_bytes = ARRAY_SIZE*sizeof(float);
    x = (float*)malloc(array_bytes);
    y = (float*)malloc(array_bytes;
    for(int i = 0; i < ARRAY_SIZE; ++i) {
        x[i] = (float)rand()/(float)RAND_MAX;
        y[i] = (float)rand()/(float)RAND_MAX;
    }
    float alpha = (float)rand()/(float)RAND_MAX;
    int size = ARRAY_SIZE;

    // allocate buffers on device
    cl_mem x_dev = clCreateBuffer(context, CL_MEM_READ_ONLY, array_bytes, NULL, &err); CHK_ERROR(err);
    cl_mem y_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, array_bytes, NULL, &err); CHK_ERROR(err);

    // copy data from host to device
    err = clEnqueueWriteBuffer(cmd_queue, x_dev, CL_TRUE, 0, array_bytes, x, 0, NULL, NULL); CHK_ERROR(err);
    err = clEnqueueWriteBuffer(cmd_queue, y_dev, CL_TRUE, 0, array_bytes, y, 0, NULL, NULL); CHK_ERROR(err);
    
    // compile the kernel    
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&SAXPY_kernel, NULL, &err); CHK_ERROR(err);
    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL); CHK_ERROR(err);
    cl_kernel kernel = clCreateKernel(program, "device_SAXPY", &err); CHK_ERROR(err);

    // set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(float), (void*) &alpha); CHK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &x_dev); CHK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &y_dev); CHK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(float), (void*) &size); CHK_ERROR(err);

    // set kernel work items/groups
    size_t workgroup_size = 16;
    size_t work_items = ((ARRAY_SIZE+workgroup_size-1)/workgroup_size)*workgroup_size;
    printf("wg size = %li, wg items = %li\n", workgroup_size, work_items);

    // SAXPY
    printf("Computing SAXPY on GPU...");
    time_dev = cpuSecond();
    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &work_items, &workgroup_size, 0, NULL, NULL); CHK_ERROR(err);
    time_dev = cpuSecond() - time_dev;
    printf(" Done!\n");

    // copy results back to device
    dev_res = malloc(sizeof(float)*ARRAY_SIZE);
    err = clEnqueueReadBuffer(cmd_queue, y_dev, CL_TRUE, 0, array_bytes, dev_res, 0, NULL, NULL); CHK_ERROR(err);
    
    // ensure all cmds to device are done
    err = clFlush(cmd_queue); CHK_ERROR(err);
    err = clFinish(cmd_queue); CHK_ERROR(err);

    // Finally, release all that we have allocated.
    err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
    err = clReleaseContext(context);CHK_ERROR(err);
    err = clReleaseMemObject(x_dev); CHK_ERROR(err);
    err = clReleaseMemObject(y_dev); CHK_ERROR(err);
    free(platforms);
    free(device_list);

    // compute SAXPY on host
    printf("Computing SAXPY on CPU...");
    time_host = cpuSecond();
    SAXPY_cpu(alpha, x, y);
    time_host = cpuSecond() - time_host;
    printf(" Done!\n");

    // compare solutions
    int same = compare_solutions(dev_res, y);
    printf("Same solutions? %i\n", same);

    // free allocated memory on host
    free(x); free(y); free(dev_res);

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