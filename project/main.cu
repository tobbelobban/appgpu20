
#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#define err_margin 1e-2

typedef struct {
    int nrows, ncols, nnz;
    float ratio;
    float* vals = NULL;
    int* cols = NULL, *rows = NULL;
} crs;

typedef struct {
    int nrows, ncols, ell_ncols, nnz;
    unsigned long long size;
    float occ;
    float* vals = NULL;
    int* cols = NULL;
} ellpack;

typedef struct {
    int nrows, ncols, sell_nrows, nc, nnz, C;
    float occ;
    unsigned long long size;
    float* vals = NULL;
    int* cols = NULL, *cs = NULL, *cl = NULL;
} sell;

typedef struct {
    int nrows, ncols, nnz;
    float* vals;
    int* rows, *cols;
} coo;

void free_CRS(crs* const my_crs)
{
    free(my_crs->vals);
    free(my_crs->cols);
    free(my_crs->rows);
}

void free_ELLPACK(ellpack* const my_ellpack)
{
    free(my_ellpack->vals);
    free(my_ellpack->cols);
}

void free_SELL(sell* const my_sell) 
{
    free(my_sell->vals);
    free(my_sell->cols);
    free(my_sell->cs);
    free(my_sell->cl);
}

void free_COO(coo* const my_coo)
{
    free(my_coo->vals);
    free(my_coo->cols);
    free(my_coo->rows);
}

void print_CRS(const crs* const my_crs)
{
    printf("--CRS--\n");
    for(int r = 0; r < my_crs->nrows; ++r)
    {
        printf("row %i\n", r);
        for(int c = my_crs->rows[r]; c < my_crs->rows[r+1]; ++c)
            printf("(col: %i, val: %f)\n",my_crs->cols[c], my_crs->vals[c]);
        printf("\n");
    }
}

void print_ELLPACK(const ellpack* const my_ellpack)
{
    printf("--ELLPACK--\n");
    for(int r = 0; r < my_ellpack->nrows; ++r)
    {
        printf("row %i\n", r);
        for(int c = 0; c < my_ellpack->ell_ncols; ++c)
            printf("(col: %i, val: %f)\n",my_ellpack->cols[c*my_ellpack->nrows+r], my_ellpack->vals[c*my_ellpack->nrows+r]);
        printf("\n");
    }
}

void print_SELL(const sell* const my_sell)
{
    printf("--SELL--\n");
    for(int c = 0; c < my_sell->nc; ++c)
    {
        
        for(int r = 0; r < my_sell->C; ++r)
        {
            int row = c * my_sell->C+r;
            int offset = my_sell->cs[c]+r;
            printf("row %i\n", row);
            for(int j = 0; j < my_sell->cl[c]; ++j)
            {
                printf("(col: %i, val: %f)\n",my_sell->cols[offset], my_sell->vals[offset]);
                offset += my_sell->C;
            }                
            printf("\n");
        }
    }
}

double cpuSecond() 
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void calc_statistics(double* stats, const double* times, int size) {
    
	double temp=0.0;
	int i;
	// compute the mean
	
	for (i = 0; i < size; ++i) temp += times[i];
    stats[0] = temp;
	temp /= size;
	stats[1] = temp;
	double mean = temp;
    temp = 0;

    // compute the standard deviation
	for (i = 0; i < size; ++i) temp += (times[i] - mean) * (times[i] - mean);
	temp /= (size > 1 ? size - 1 : 1);
	stats[2] = sqrt(temp);
}

__global__
void SpMVM_CRS_kernel(float* const y, const crs my_crs, const float* const x)
{
    const int thread_row = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_row < my_crs.nrows)
    {
        float res = 0.0;
        for(int c = my_crs.rows[thread_row]; c < my_crs.rows[thread_row+1]; ++c)
        {
            res += my_crs.vals[c]*x[my_crs.cols[c]]; 
        }
        y[thread_row] = res;
    }
}

__global__
void SpMVM_ELLPACK_kernel(float* const y, const ellpack my_ell, const float* const x)
{
    const int thread_row = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_row < my_ell.nrows)
    {
        float res = 0.0;
        int offset = thread_row;
        for(int c = 0; c < my_ell.ell_ncols; ++c)
        {
            res += my_ell.vals[offset]*x[my_ell.cols[offset]];
            offset += my_ell.nrows;
        }
        y[thread_row] = res;
    }
}

__global__
void SpMVM_SELL_kernel(float* const y, const sell my_sell, const float* const x)
{
    const int thread_row = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_row < my_sell.nrows)
    {
        int local_chunk = thread_row / my_sell.C;
        int offset = my_sell.cs[local_chunk] + threadIdx.x;
        float res = 0.0;
        for(int c = 0; c < my_sell.cl[local_chunk]; ++c)
        {
            res += my_sell.vals[offset] * x[my_sell.cols[offset]];
            offset += my_sell.C;
        }
        y[thread_row] = res;
    }
}

__global__
void SpMVM_COO_kernel(float* const y, const coo my_coo, const float* const x)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < my_coo.nnz)
        atomicAdd(y+my_coo.rows[thread_id], my_coo.vals[thread_id] * x[my_coo.cols[thread_id]]);
}

float float_in(const float lo, const float hi)
{
    return ((float)rand()/(float)RAND_MAX)*(float)(hi-lo) + (float)lo;
}

void SpMVM_CRS(float* const y, const crs* const my_crs, const float* const x)
{
    float tmp;
    for(int r = 0; r < my_crs->nrows; ++r)
    {
        tmp = 0;
        for(int c = my_crs->rows[r]; c < my_crs->rows[r+1]; ++c)
            tmp += my_crs->vals[c]*x[my_crs->cols[c]]; 
        y[r] = tmp;
    }
}

void print_vector(const float* const vec, const int size)
{
    for(int i = 0; i < size; ++i)
    {
        printf("%.4f\t", vec[i]);
    }
}

int cmpints (const void * a, const void * b) {
    return ( *(int*)a - *(int*)b );
 }

void generate_sparse_matrix_CRS(const int nrows, const int ncols, const int nnz, crs* const my_crs)
{
    const unsigned long long size = (unsigned long long)nrows * (unsigned long long)ncols;
    if(nnz < 0 || nrows < 0 || ncols < 0 || size < nnz) {
        my_crs->nnz = -1;
        return;
    }

    int* cols = (int*)malloc(sizeof(int)*nnz);
    int* row_counters = (int*)malloc(sizeof(int)*nrows);
    memset(row_counters, 0, sizeof(int)*nrows);

    // distribute nnz over rows and cols randomly
    int covered = 0;
    while(covered < nnz)
    {
        int random_row = (int)float_in(0,nrows-1);
        if(row_counters[random_row] < ncols)
        {
            row_counters[random_row] += 1;
            cols[covered] = (int)float_in(0,ncols-1);
            ++covered;
        }   
    }

    float* vals = (float*)malloc(sizeof(float)*nnz);
    int* rows = (int*)malloc(sizeof(int)*nrows+1);
    
    int offset = 0;

    // try to distribute nnz over columns, skip duplicates 
    for(int r = 0; r < nrows; ++r)
    {
        rows[r] = offset;
        int rowc = row_counters[r];
        if(rowc)
        {
            // sort columns of current row
            qsort(cols+offset, rowc, sizeof(int),cmpints);

            // set first element of row
            vals[offset] = float_in(-10,10);
            int row_end = offset + rowc;
            int front = offset+1;
            for(int i = front; i < row_end; ++i)
            {
                if(cols[i] == cols[i-1])
                {
                    --rowc;
                    continue;
                }
                vals[front] = float_in(-2,2);
                cols[front] = cols[i];
                ++front;                
            }
            offset += rowc;
        }
    }
    free(row_counters);
    rows[nrows] = offset;

    // set arrays of CRS
    my_crs->vals = (float*)malloc(sizeof(float)*offset);
    my_crs->cols = (int*)malloc(sizeof(int)*offset);
    memcpy((void*)my_crs->vals, (const void*)vals, sizeof(float)*offset);
    memcpy((void*)my_crs->cols, (const void*)cols, sizeof(int)*offset);
    my_crs->rows = rows;

    // set other members of CRS
    my_crs->nrows = nrows;
    my_crs->ncols = ncols;
    my_crs->nnz = offset;
    my_crs->ratio = (float)offset/(float)size;

    free(vals);
    free(cols);
}

void generate_COO_from_CRS(const crs* const my_crs, coo* const my_coo)
{
    float* vals = (float*)malloc(sizeof(float)*my_crs->nnz); 
    int* cols = (int*)malloc(sizeof(int)*my_crs->nnz);
    int* rows = (int*)malloc(sizeof(int)*my_crs->nnz);
    int offset = 0;
    for(int r = 0; r < my_crs->nrows; ++r)
    {
        for(int c = my_crs->rows[r]; c < my_crs->rows[r+1]; ++c)
        {
            vals[offset] = my_crs->vals[offset];
            cols[offset] = my_crs->cols[offset];
            rows[offset] = r;
            ++offset;
        }
    }
    my_coo->nrows = my_crs->nrows;
    my_coo->ncols = my_crs->ncols;
    my_coo->nnz = my_crs->nnz;
    my_coo->vals = vals;
    my_coo->cols = cols;
    my_coo->rows = rows;
}

void generate_ELLPACK_from_CRS(const crs* const my_crs, ellpack* const my_ellpack)
{
    // find length of longest row
    int maxrow_len= 0;
    int* row_len_counters = (int*)malloc(sizeof(int)*my_crs->nrows);
    for(int r = 0; r < my_crs->nrows; ++r)
    {
        int tmp_len = my_crs->rows[r+1] - my_crs->rows[r];
        row_len_counters[r] = tmp_len;
        if(tmp_len > maxrow_len)
            maxrow_len = tmp_len;
    }

    // allocate ELLPACK arrays
    const unsigned long long ellpack_size = (unsigned long long)maxrow_len * (unsigned long long)my_crs->nrows;
    float* vals = (float*)malloc(sizeof(float)*ellpack_size);
    int* cols = (int*)malloc(sizeof(int)*ellpack_size);
    if(vals == NULL || cols == NULL)
    {
        my_ellpack->nnz = -1;
        return;
    }
    // insert data in ellpack arrays
    int offset = 0;
    for(int c = 0; c < maxrow_len; ++c)
    {
        for(int r = 0; r < my_crs->nrows; ++r)
        {
            if(row_len_counters[r])
            {
                vals[offset] = my_crs->vals[my_crs->rows[r+1] - row_len_counters[r]];
                cols[offset] = my_crs->cols[my_crs->rows[r+1] - row_len_counters[r]];
                row_len_counters[r] -= 1;
            } 
            else 
            {
                vals[offset] = 0.0;
                cols[offset] = 0;
            }
            ++offset;   
        }
    }
    free(row_len_counters);

    // set members of ellpack
    my_ellpack->nrows = my_crs->nrows;
    my_ellpack->ncols = my_crs->ncols;
    my_ellpack->ell_ncols = maxrow_len;
    my_ellpack->nnz = my_crs->nnz;
    my_ellpack->occ = (float)my_ellpack->nnz/(float)ellpack_size;
    my_ellpack->vals = vals;
    my_ellpack->cols = cols;
    my_ellpack->size = ellpack_size;
}

void generate_SELL_from_CRS(const crs* const my_crs, sell* const my_sell, const int C)
{
    
    // initialize values and arrays
    const int num_chunks = (my_crs->nrows + C - 1)/ C;
    const int num_rows = num_chunks * C;
    int* chunk_maxlens = (int*)malloc(sizeof(int)*num_chunks);
    int* chunk_starts = (int*)malloc(sizeof(int)*num_chunks);
    int* row_len_counters = (int*)malloc(sizeof(int)*num_rows);
    if(chunk_maxlens == NULL || chunk_starts == NULL || row_len_counters == NULL)
    {
        my_sell->nnz = -1;
        return;
    }
    
    // find length of longest row per chunk
    int curr_row = 0;
    int offset = 0;
    for(int c = 0; c < num_chunks; ++c)
    {
        chunk_maxlens[c] = 0;
        chunk_starts[c] = offset;
        for(int r = 0; r < C; ++r)
        {
            int row_len = 0;
            if(curr_row < my_crs->nrows)
                row_len = my_crs->rows[curr_row+1] - my_crs->rows[curr_row];
            row_len_counters[curr_row]= row_len;
            if(row_len > chunk_maxlens[c]) 
                chunk_maxlens[c] = row_len;    
            ++curr_row;
        }
        offset += C * chunk_maxlens[c];
    }
    
    // allocate SELL arrays
    float* vals = (float*)malloc(sizeof(float)*offset);
    int* cols = (int*)malloc(sizeof(int)*offset);
    if(vals == NULL || cols == NULL)
    {
        my_sell->nnz = -1;
        return;
    }
    
    offset = 0;
    // insert data into SELL arrays
    for(int c = 0; c < num_chunks; ++c)
    {
        for(int k = 0; k < chunk_maxlens[c]; ++k)
        {
            int row_offset = c * C;
            for(int r = row_offset; r < row_offset+C; ++r)
            {
                if(row_len_counters[r])
                {
                    vals[offset] = my_crs->vals[my_crs->rows[r+1]-row_len_counters[r]];
                    cols[offset] = my_crs->cols[my_crs->rows[r+1]-row_len_counters[r]];
                    row_len_counters[r] -= 1;
                }
                else
                {
                    vals[offset] = 0.0;
                    cols[offset] = 0;
                }
                ++offset;
            }
        }
    }
    free(row_len_counters);

    // set members of SELL
    my_sell->nrows = my_crs->nrows;
    my_sell->ncols = my_crs->ncols;
    my_sell->sell_nrows = num_rows;
    my_sell->nnz = my_crs->nnz;
    my_sell->nc = num_chunks;
    my_sell->C = C;
    my_sell->size = offset;
    my_sell->occ = (float)my_sell->nnz/(float)offset;
    my_sell->cs = chunk_starts;
    my_sell->cl = chunk_maxlens;
    my_sell->vals = vals;
    my_sell->cols = cols;
}

void init_vector(float* const vec, const int size)
{
    for(int i = 0; i < size; ++i)
        vec[i] = float_in(-10,10);
}

bool compare_vectors(const float* const vec1, const float* const vec2, int size)
{
    for(int i = 0; i < size; ++i)
    {
        if(fabs(vec1[i]-vec2[i]) > err_margin)
        {
            printf("Error (index = %i): vec1[%i] = %f != %f = vec2[%i]\n", i, i, vec1[i], vec2[i], i);
            return false;
        }
    }
    return true;
}

void checkCudaOp(const cudaError_t cudaRes)
{
    if(cudaRes != cudaSuccess)
    {
        printf("CUDA error = %i. Exiting!\n", cudaRes);
        printf("%s\n",cudaGetErrorString(cudaRes));
        exit(0);
    }
}

int main(int argc, char* argv[])
{
    int seed = 100, nrows = 10000, ncols = 10000, nnz = 100, iterations = 10;
    int block_size = 32;
    bool same, verify = false;

    // process command arguments
    int option;
    while((option = getopt(argc, argv, ":i:b:r:c:n:s:v")) != -1)
    { 
        switch(option)
        {
            case 'r':   // rows
                nrows = atoi(optarg);
                break;
            case 'c':   // columns
                ncols = atoi(optarg);
                break;
            case 'n':   // number of non-zero elements
                nnz = atoi(optarg);
                break;
            case 'b':
                block_size = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            case 'v':
                verify = true;
                break;
            default:
                printf("Unrecognized flag.\n");
        }
    }

    srand(seed);

    crs host_crs;
    ellpack host_ellpack;
    sell host_sell;
    coo host_coo;

    printf("\n-- Generating sparse m x n matrix --\n"); fflush(stdout);
    
    generate_sparse_matrix_CRS(nrows, ncols, nnz, &host_crs);
    if(host_crs.nnz < 0) 
    {
        printf("Error in sparse matrix generator. Exiting.\n");
        free_CRS(&host_crs);
        exit(0);
    }

    printf("m\t= %i\nn\t= %i\nNNZ\t= %i\nNNZ/(m x n) = %f\n\n",host_crs.nrows, host_crs.ncols, host_crs.nnz, host_crs.ratio);
    printf("block size\t= %i\n\n", block_size);
    fflush(stdout);

    printf("############################################\n\n");
    
    unsigned long long total_size =  sizeof(float)*host_crs.nnz + sizeof(int)*host_crs.nnz + sizeof(int)*(host_crs.nrows+1);
    printf("-- CRS --\nSIZE (bytes)\nvals CRS\t= %li \ncols CRS\t= %li\nrows CRS\t= %li\ntotal size CRS\t= %lli\n\n", sizeof(float)*host_crs.nnz, sizeof(int)*host_crs.nnz, sizeof(int)*(host_crs.nrows+1), total_size);

    // setup grid and block sizes
    dim3 dim_grid((nrows+block_size-1)/block_size, 1, 1);
    dim3 dim_block(block_size, 1, 1);
    cudaError_t cudaRes;

    // timers
    double* SpMVM_times = (double*)malloc(sizeof(double)*iterations);
    double* mem_times = (double*)malloc(sizeof(double)*iterations);;

    // for statistics stats[0] = total time, stats[1] = mean, stats[2] = std. deviation
    double stats[3];
    int same_counter;

    // declare variables for SpMVM
    float* x, *d_x, *y, *d_y, *d_res, *d_vals;
    int* d_cols, *d_rows;

    x = (float*)malloc(sizeof(float)*host_crs.ncols);
    y = (float*)malloc(sizeof(float)*host_crs.nrows);
    d_res = (float*)malloc(sizeof(float)*host_crs.nrows);

    init_vector(x,ncols);
    // SpMVM-CRS on host
    
    if(verify)
        SpMVM_CRS(y, &host_crs, x);
    
    //allocate x and y vectors on device
    cudaRes = cudaMalloc((void**)&d_x, sizeof(float)*host_crs.ncols); checkCudaOp(cudaRes);
    cudaRes = cudaMemcpy(d_x, x, sizeof(float)*host_crs.ncols, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
    cudaRes = cudaMalloc((void**)&d_y, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);
    
    // ###########################################################################################
    // ###########################################################################################
    
    // SpMVM-CRS on device

    crs d_crs;
    d_crs.nrows = host_crs.nrows;
    d_crs.ncols = host_crs.ncols;
    d_crs.nnz = host_crs.nnz;
    d_crs.ratio = host_crs.ratio;
    
    same_counter = 0;
    for(int i = 0; i < iterations; ++i)
    {
        mem_times[i] = cpuSecond();

        cudaRes = cudaMalloc((void**)&d_vals, sizeof(float)*d_crs.nnz); checkCudaOp(cudaRes);
        cudaRes = cudaMemcpy(d_vals, host_crs.vals, sizeof(float)*d_crs.nnz, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
        d_crs.vals = d_vals;

        cudaRes = cudaMalloc((void**)&d_cols, sizeof(int)*d_crs.nnz); checkCudaOp(cudaRes);
        cudaRes = cudaMemcpy(d_cols, host_crs.cols, sizeof(int)*d_crs.nnz, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
        d_crs.cols = d_cols;

        cudaRes = cudaMalloc((void**)&d_rows, sizeof(int)*(d_crs.nrows+1)); checkCudaOp(cudaRes);
        cudaRes = cudaMemcpy(d_rows, host_crs.rows, sizeof(int)*(d_crs.nrows+1), cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
        d_crs.rows = d_rows;
        
        mem_times[i] = cpuSecond() - mem_times[i];

        // set y to 0
        cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);

        //warm up run
        SpMVM_CRS_kernel<<<dim_grid,dim_block>>>(d_y,  d_crs, d_x);
        cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);

        // set y to 0
        cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);

        // perform SpMVM with CRS on GPU
        SpMVM_times[i] = cpuSecond();
        SpMVM_CRS_kernel<<<dim_grid,dim_block>>>(d_y,  d_crs, d_x);
        cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);
        SpMVM_times[i] = cpuSecond() - SpMVM_times[i];
        if(verify)
        {
            cudaRes = cudaMemcpy(d_res, d_y, sizeof(float)*d_crs.nrows, cudaMemcpyDeviceToHost); checkCudaOp(cudaRes);
            same = compare_vectors(y, d_res, host_crs.nrows);
            if(same) same_counter += 1;
        }
        // free allocations for SpMVM-CRS on device
        cudaRes = cudaFree(d_vals); checkCudaOp(cudaRes);
        cudaRes = cudaFree(d_cols); checkCudaOp(cudaRes);
        cudaRes = cudaFree(d_rows); checkCudaOp(cudaRes);
    }
    
    // get statistics for memory allocations and transfer
    calc_statistics(stats, mem_times, iterations);

    printf("iterations = %i\n", iterations);
    // print CRS statistics
    if(verify) printf("number of correct iterations: %i\n", same_counter);
    printf("\nMEMORY\nalloc & copy CRS\t= %f s\nmean mem CRS\t\t= %f s\nstddev mem CRS\t\t= %f s\n", stats[0], stats[1], stats[2]);

    // get statistics for SpMVM-CRS
    calc_statistics(stats, SpMVM_times, iterations);
    printf("\nSpMVM\ntotal time CRS\t\t= %f s\nmean time CRS\t\t= %f s\nstddev time CRS\t\t= %f s\n\n", stats[0], stats[1], stats[2]);
    
    //clear previous result
    cudaRes = cudaMemset((void*)d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);
    memset((void*)d_res, 0, sizeof(float)*host_crs.nrows); 

    // ###########################################################################################
    // ###########################################################################################
    
    // SpMVM-ELLPACK on device
    
    generate_ELLPACK_from_CRS(&host_crs, &host_ellpack);

    if(host_ellpack.nnz > 0)
    {
        total_size = sizeof(float)*host_ellpack.size + sizeof(int)*host_ellpack.size;
        printf("############################################\n\n");
        printf("-- ELLPACK --\nSIZE (bytes)\nvals ELLPACK\t\t= %lli\ncols ELLPACK\t\t= %lli\ntotal size ELLPACK\t= %lli\noccupancy ELLPACK\t= %f\n\n", sizeof(float)*host_ellpack.size, sizeof(int)*host_ellpack.size, total_size, host_ellpack.occ);

        ellpack d_ellpack;
        d_ellpack.nrows = host_ellpack.nrows;
        d_ellpack.ncols = host_ellpack.ncols;
        d_ellpack.ell_ncols = host_ellpack.ell_ncols;
        d_ellpack.nnz = host_ellpack.nnz;
        d_ellpack.occ = host_ellpack.occ;
        d_ellpack.size = host_ellpack.size;
    
        same_counter = 0;
        for(int i = 0; i < iterations; ++i)
        {
            mem_times[i] = cpuSecond();

            cudaRes = cudaMalloc((void**)&d_vals, sizeof(float)*d_ellpack.size); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_vals, host_ellpack.vals, sizeof(float)*d_ellpack.size, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_ellpack.vals = d_vals;
        
            cudaRes = cudaMalloc((void**)&d_cols, sizeof(int)*d_ellpack.size); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_cols, host_ellpack.cols, sizeof(int)*d_ellpack.size, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_ellpack.cols = d_cols; 
            
            mem_times[i] = cpuSecond() - mem_times[i];
            
            // set y to 0
            cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);

            // warm up
            SpMVM_ELLPACK_kernel<<<dim_grid,dim_block>>>(d_y, d_ellpack, d_x);
            cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);
            
            // set y to 0
            cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);
            
            SpMVM_times[i] = cpuSecond();
            SpMVM_ELLPACK_kernel<<<dim_grid,dim_block>>>(d_y, d_ellpack, d_x);
            cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);
            SpMVM_times[i] = cpuSecond() - SpMVM_times[i];
            if(verify)
            {
                cudaRes = cudaMemcpy(d_res, d_y, sizeof(float)*host_crs.nrows, cudaMemcpyDeviceToHost); checkCudaOp(cudaRes);
                same = compare_vectors(y, d_res, host_crs.nrows);
                if(same) same_counter += 1;
            }
    
            // free allocations on device
            cudaRes = cudaFree(d_vals); checkCudaOp(cudaRes);
            cudaRes = cudaFree(d_cols); checkCudaOp(cudaRes);
        }
        
        // free ELLPACK on host
        free_ELLPACK(&host_ellpack);
        
        // get statistics for memory allocations and transfer
        calc_statistics(stats, mem_times, iterations);

        printf("iterations = %i\n", iterations);
        
        // print ELLPACK statistics
        if(verify) printf("number of correct iterations: %i\n", same_counter);
        printf("\nMEMORY\nalloc & copy ELLPACK\t\t= %f s\nmean mem ELLPACK\t\t= %f s\nstddev mem ELLPACK\t\t= %f s\n", stats[0], stats[1], stats[2]);

        // get statistics for SpMVM-ELLPACK
        calc_statistics(stats, SpMVM_times, iterations);
        printf("\nSpMVM\ntotal time ELLPACK\t\t= %f s\nmean time ELLPACK\t\t= %f s\nstddev time ELLPACK\t\t= %f s\n\n", stats[0], stats[1], stats[2]);

        //clear previous result
        cudaRes = cudaMemset((void*)d_y, 0, sizeof(float)*nrows); checkCudaOp(cudaRes);
        memset((void*)d_res, 0, sizeof(float)*nrows);
    }
    
    // ###########################################################################################
    // ###########################################################################################

    // SpMVM-SELL on device

    generate_SELL_from_CRS(&host_crs, &host_sell, block_size);

    if(host_sell.nnz > 0)
    {
        total_size = sizeof(float)*host_sell.size + sizeof(int)*host_sell.size + sizeof(int)*host_sell.nc + sizeof(int)*host_sell.nc;
        printf("############################################\n\n");
        printf("-- SELL --\nSIZE (bytes)\nvals SELL\t\t= %lli\ncols SELL\t\t= %lli\ncs\t\t= %li\ncl\t\t= %li\ntotal size SELL\t= %lli\noccupancy SELL\t= %f\n\n", sizeof(float)*host_sell.size, sizeof(int)*host_sell.size, sizeof(int)*host_sell.nc, sizeof(int)*host_sell.nc, total_size, host_sell.occ);

        sell d_sell;
        d_sell.nrows = host_sell.nrows;
        d_sell.ncols = host_sell.ncols;
        d_sell.sell_nrows = host_sell.sell_nrows;
        d_sell.nc = host_sell.nc;
        d_sell.nnz = host_sell.nnz;
        d_sell.size = host_sell.size;
        d_sell.C = host_sell.C;
        d_sell.occ = host_sell.occ;
        int* d_cs, *d_cl;
        
        same_counter = 0;
        for(int i = 0; i < iterations; ++i)
        {
            mem_times[i] = cpuSecond();

            cudaRes = cudaMalloc((void**)&d_vals, sizeof(float)*host_sell.size); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_vals, host_sell.vals, sizeof(float)*host_sell.size, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_sell.vals = d_vals;
        
            cudaRes = cudaMalloc((void**)&d_cols, sizeof(int)*host_sell.size); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_cols, host_sell.cols, sizeof(int)*host_sell.size, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_sell.cols = d_cols;
            
            cudaRes = cudaMalloc((void**)&d_cs, sizeof(int)*host_sell.nc); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_cs, host_sell.cs, sizeof(int)*host_sell.nc, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_sell.cs = d_cs;
        
            cudaRes = cudaMalloc((void**)&d_cl, sizeof(int)*host_sell.nc); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_cl, host_sell.cl, sizeof(int)*host_sell.nc, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_sell.cl = d_cl;
            
            mem_times[i] = cpuSecond() - mem_times[i];

            // set y to 0
            cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);
            
            //warm up
            SpMVM_SELL_kernel<<<dim_grid,dim_block>>>(d_y, d_sell, d_x);
            cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);
            
            // set y to 0
            cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);

            SpMVM_times[i] = cpuSecond();
            SpMVM_SELL_kernel<<<dim_grid,dim_block>>>(d_y, d_sell, d_x);
            cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);
            SpMVM_times[i] = cpuSecond() - SpMVM_times[i];
            if(verify)
            {
                cudaRes = cudaMemcpy(d_res, d_y, sizeof(float)*host_crs.nrows, cudaMemcpyDeviceToHost); checkCudaOp(cudaRes);
                same = compare_vectors(y, d_res, host_crs.nrows);
                if(same) same_counter += 1;
            }
            
            // free allocations on device
            cudaRes = cudaFree(d_vals); checkCudaOp(cudaRes);
            cudaRes = cudaFree(d_cols); checkCudaOp(cudaRes);
            cudaRes = cudaFree(d_cs); checkCudaOp(cudaRes);
            cudaRes = cudaFree(d_cl); checkCudaOp(cudaRes);
        }
        
        // free SELL on host
        free_SELL(&host_sell);
    
        // get statistics for memory allocations and transfer
        calc_statistics(stats, mem_times, iterations);

        printf("iterations = %i\n", iterations);
        
        // print SELL statistics
        if(verify) printf("number of correct iterations: %i\n", same_counter);
        printf("\nMEMORY\nalloc & copy SELL\t= %f s\nmean mem SELL\t\t= %f s\nstddev mem SELL\t\t= %f s\n", stats[0], stats[1], stats[2]);

        // get statistics for SpMVM-SELL
        calc_statistics(stats, SpMVM_times, iterations);
        printf("\nSpMVM\ntotal time SELL\t\t= %f s\nmean time SELL\t\t= %f s\nstddev time SELL\t= %f s\n\n", stats[0], stats[1], stats[2]);

        //clear previous result
        cudaRes = cudaMemset((void*)d_y, 0, sizeof(float)*nrows); checkCudaOp(cudaRes);
        memset((void*)d_res, 0, sizeof(float)*nrows);
    }
    
    // ###########################################################################################
    // ###########################################################################################

    // SpMVM-COO on device

    generate_COO_from_CRS(&host_crs, &host_coo);

    if(host_coo.nnz > 0)
    {
        printf("############################################\n\n");
        total_size = sizeof(float)*host_coo.nnz + sizeof(int)*host_coo.nnz + sizeof(int)*host_coo.nnz;
        printf("-- COO --\nSIZE (bytes)\nvals COO\t= %li\ncols COO\t= %li\nrows COO\t= %li\ntotal size COO\t= %lli\n\n", sizeof(float)*host_coo.nnz, sizeof(int)*host_coo.nnz, sizeof(int)*host_coo.nnz, total_size);

        // setup grid size for COO 
        dim3 COO_dim_grid((host_coo.nnz+block_size-1)/block_size, 1, 1);
        coo d_coo;

        d_coo.nrows = host_coo.nrows;
        d_coo.ncols = host_coo.ncols;
        d_coo.nnz = host_coo.nnz;
        
        same_counter = 0;
        for(int i = 0; i < iterations; ++i)
        {
            mem_times[i] = cpuSecond();

            cudaRes = cudaMalloc((void**)&d_vals, sizeof(float)*host_coo.nnz); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_vals, host_coo.vals, sizeof(float)*host_coo.nnz, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_coo.vals = d_vals;
        
            cudaRes = cudaMalloc((void**)&d_cols, sizeof(int)*host_coo.nnz); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_cols, host_coo.cols, sizeof(int)*host_coo.nnz, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_coo.cols = d_cols;

            cudaRes = cudaMalloc((void**)&d_rows, sizeof(int)*host_coo.nnz); checkCudaOp(cudaRes);
            cudaRes = cudaMemcpy(d_rows, host_coo.rows, sizeof(int)*host_coo.nnz, cudaMemcpyHostToDevice); checkCudaOp(cudaRes);
            d_coo.rows = d_rows;

            mem_times[i] = cpuSecond() - mem_times[i];

            // set y to 0
            cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);

            //warm up
            SpMVM_COO_kernel<<<COO_dim_grid,dim_block>>>(d_y, d_coo, d_x);
            cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);
            
            // set y to 0
            cudaRes = cudaMemset(d_y, 0, sizeof(float)*host_crs.nrows); checkCudaOp(cudaRes);

            SpMVM_times[i] = cpuSecond();
            SpMVM_COO_kernel<<<COO_dim_grid,dim_block>>>(d_y, d_coo, d_x);
            cudaRes = cudaDeviceSynchronize(); checkCudaOp(cudaRes);
            SpMVM_times[i] = cpuSecond() - SpMVM_times[i];
            if(verify)
            {
                cudaRes = cudaMemcpy(d_res, d_y, sizeof(float)*host_crs.nrows, cudaMemcpyDeviceToHost); checkCudaOp(cudaRes);
                same = compare_vectors(y, d_res, host_crs.nrows);
                if(same) same_counter += 1;
            }
            
            // free allocations on device
            cudaRes = cudaFree(d_vals); checkCudaOp(cudaRes);
            cudaRes = cudaFree(d_cols); checkCudaOp(cudaRes);
            cudaRes = cudaFree(d_rows); checkCudaOp(cudaRes);
        }
        
        // free SELL on host
        free_COO(&host_coo);
    
        // get statistics for memory allocations and transfer
        calc_statistics(stats, mem_times, iterations);

        printf("iterations = %i\n", iterations);
        
        // print SELL statistics
        if(verify) printf("number of correct iterations: %i\n", same_counter);
        printf("\nMEMORY\nalloc & copy COO\t= %f s\nmean mem COO\t\t= %f s\nstddev mem COO\t\t= %f s\n", stats[0], stats[1], stats[2]);

        // get statistics for SpMVM-SELL
        calc_statistics(stats, SpMVM_times, iterations);
        printf("\nSpMVM\ntotal time COO\t\t= %f s\nmean time COO\t\t= %f s\nstddev time COO\t\t= %f s\n", stats[0], stats[1], stats[2]);
    }

    // free allocations on host and device
    cudaRes = cudaFree(d_y); checkCudaOp(cudaRes);
    cudaRes = cudaFree(d_x); checkCudaOp(cudaRes);
    
    free_CRS(&host_crs);
    
    free(x);
    free(y);
    free(d_res);
    free(SpMVM_times);
    free(mem_times);

    return 0;
}