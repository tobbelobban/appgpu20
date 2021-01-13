
#include <stdio.h>
#include <stdint.h>
#include <getopt.h>
#include <math.h>
#include <sys/time.h>

#define err_margin 1e-3

typedef struct {
    int nrows, ncols, nnz;
    float ratio;
    float* vals = NULL;
    int* cols = NULL, *rows = NULL;
} crs;

typedef struct {
    int nrows, ncols, ell_ncols, nnz;
    float occ;
    float* vals = NULL;
    int* cols = NULL;
} ellpack;

typedef struct {
    int nrows, ncols, nc, nnz, C;
    float occ;
    float* vals = NULL;
    int* cols = NULL, *cs = NULL, *cl = NULL;
} sell;

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

void generate_sparse_matrix_CRS(const int nrows, const int ncols, const int nnz, crs* const my_crs)
{
    const unsigned long long len = (unsigned long long)nrows * (unsigned long long)ncols;
    if(nnz < 0 || nrows < 0 || ncols < 0 || len < nnz) {
        my_crs->nnz = -1;
        return;
    }

    // allocate vectors for CRS
    float* vals = (float*)malloc(sizeof(float)*nnz);
    int* cols = (int*)malloc(sizeof(int)*nnz);
    int* rows = (int*)malloc(sizeof(int)*(nrows+1));
    memset(rows,0,sizeof(int)*nrows);
    
    // distribute nnz over rows randomly
    int remaining = nnz;
    while(remaining)
    {
        int random_row = nrows*((double)rand()/(RAND_MAX+1.0));
        if(rows[random_row] < ncols)
        {
            rows[random_row] += 1;
            --remaining;
        }
        
    }
    printf("bob\n");
    // insert random values and cols 
    float* tmp_row_vals = (float*)malloc(sizeof(float)*ncols);
    int offset = 0;
    for(int r = 0; r < nrows; ++r)
    {
        remaining = rows[r];
        if(!remaining) continue;
        rows[r] = offset;
        memset(tmp_row_vals,0,sizeof(float)*ncols);
        while(remaining)
        {
            int random_col = ncols*((double)rand()/(RAND_MAX+1.0));
            if(tmp_row_vals[random_col]) continue;
            tmp_row_vals[random_col] = 1.0;
            --remaining;
        }
        for(int c = 0; c < ncols; ++c)
        {
            if(tmp_row_vals[c])
            {
                vals[offset] = float_in(-10,10);
                cols[offset] = c;
                ++offset;
            }
        }
    }
    free(tmp_row_vals);

    rows[nrows] = offset;

    // set members of crs
    my_crs->nrows = nrows;
    my_crs->ncols = ncols;
    my_crs->nnz = offset;
    my_crs->vals = vals;
    my_crs->cols = cols;
    my_crs->rows = rows;
    my_crs->ratio = (float)nnz/(float)len;
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
    const int ellpack_size = maxrow_len * my_crs->nrows;
    float* vals = (float*)malloc(sizeof(float)*ellpack_size);
    int* cols = (int*)malloc(sizeof(int)*ellpack_size);

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
}

void generate_SELL_from_CRS(const crs* const my_crs, sell* const my_sell, const int C)
{
    // we assume nrows % C == 0
    if(my_crs->nrows % C)
    {
        my_sell->nnz = -1;
        return;
    }

    // initialize values and arrays
    const int num_chunks = my_crs->nrows / C;
    int* chunk_maxlens = (int*)malloc(sizeof(int)*num_chunks);
    int* chunk_starts = (int*)malloc(sizeof(int)*num_chunks);
    int* row_len_counters = (int*)malloc(sizeof(int)*my_crs->nrows);
    
    // find length of longest row per chunk
    int curr_row = 0, sell_size = 0;
    for(int c = 0; c < num_chunks; ++c)
    {
        chunk_maxlens[c] = 0;
        chunk_starts[c] = sell_size;
        for(int r = 0; r < C; ++r)
        {
            int row_len = my_crs->rows[curr_row+1] - my_crs->rows[curr_row];
            row_len_counters[curr_row] = row_len;
            if(row_len > chunk_maxlens[c]) 
                chunk_maxlens[c] = row_len;
            ++curr_row;
        }
        sell_size += C * chunk_maxlens[c];
    }
    
    // allocate SELL arrays
    float* vals = (float*)malloc(sizeof(float)*sell_size);
    int* cols = (int*)malloc(sizeof(int)*sell_size);
    int offset = 0;

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
    my_sell->nnz = my_crs->nnz;
    my_sell->nc = num_chunks;
    my_sell->C = C;
    my_sell->occ = (float)my_sell->nnz/(float)sell_size;
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

int main(int argc, char* argv[])
{
    int seed = 100, nrows = 10000, ncols = 10000, nnz = 100;
    int block_size = 32;
    bool same;

    // process command arguments
    int option;
    while((option = getopt(argc, argv, ":b:r:c:n:s:")) != -1)
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
            default:
                printf("Unrecognized flag.\n");
        }
    }

    srand(seed);

    crs host_crs;
    ellpack host_ellpack;
    sell host_sell;

    generate_sparse_matrix_CRS(nrows, ncols, nnz, &host_crs);
    
    if(host_crs.nnz != nnz) 
    {
        printf("Error in sparse matrix generator. Exiting.\n");
        free_CRS(&host_crs);
        exit(0);
    }

    generate_ELLPACK_from_CRS(&host_crs, &host_ellpack);

    generate_SELL_from_CRS(&host_crs, &host_sell, block_size);

    // setup grid and block sizes
    dim3 dim_grid((nrows+block_size-1)/block_size, 1, 1);
    dim3 dim_block(block_size, 1, 1);

    // declare variables for SpMVM
    float* x, *d_x, *y, *d_y, *d_res, *d_vals;
    int* d_cols, *d_rows;

    x = (float*)malloc(sizeof(float)*ncols);
    y = (float*)malloc(sizeof(float)*nrows);

    init_vector(x,ncols);

    // SpMVM-CRS on host
    SpMVM_CRS(y, &host_crs, x);
    
    // SpMVM-CRS on device
    crs d_crs;
    d_crs.nrows = nrows;
    d_crs.ncols = ncols;
    d_crs.nnz = nnz;
    d_crs.ratio = host_crs.ratio;
    
    cudaMalloc((void**)&d_vals, sizeof(float)*nnz);
    cudaMemcpy(d_vals, host_crs.vals, sizeof(float)*nnz, cudaMemcpyHostToDevice);
    d_crs.vals = d_vals;

    cudaMalloc((void**)&d_cols, sizeof(int)*nnz);
    cudaMemcpy(d_cols, host_crs.cols, sizeof(int)*nnz, cudaMemcpyHostToDevice);
    d_crs.cols = d_cols;

    cudaMalloc((void**)&d_rows, sizeof(int)*(nrows+1));
    cudaMemcpy(d_rows, host_crs.rows, sizeof(int)*(nrows+1), cudaMemcpyHostToDevice);
    d_crs.rows = d_rows;
    
    cudaMalloc((void**)&d_x, sizeof(float)*ncols);
    cudaMemcpy(d_x, x, sizeof(float)*ncols, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_y, sizeof(float)*nrows);
    cudaMemcpy(d_y, y, sizeof(float)*nrows, cudaMemcpyHostToDevice);

    // call kernel 
    SpMVM_CRS_kernel<<<dim_grid,dim_block>>>(d_y,  d_crs, d_x);
    cudaDeviceSynchronize();
    
    // store result
    d_res = (float*)malloc(sizeof(float)*nrows);
    cudaMemcpy(d_res, d_y, sizeof(float)*nrows, cudaMemcpyDeviceToHost);
    
    same = compare_vectors(d_res, y, host_crs.nrows);
    printf("CRS same? %s\n", (same ? "yes" : "no")); 
    
    // free allocations for SpMVM-CRS on device
    cudaFree(d_vals);
    cudaFree(d_cols);
    cudaFree(d_rows);
    
    //clear previous result
    cudaMemset((void*)d_y, 0, sizeof(float)*nrows);
    
    // SpMVM-ELLPACK on device
    ellpack d_ellpack;
    const int ellpack_size = host_ellpack.nrows * host_ellpack.ell_ncols;
    d_ellpack.nrows = nrows;
    d_ellpack.ncols = ncols;
    d_ellpack.ell_ncols = host_ellpack.ell_ncols;
    d_ellpack.nnz = nnz;
    d_ellpack.occ = host_ellpack.occ;

    cudaMalloc((void**)&d_vals, sizeof(float)*ellpack_size);
    cudaMemcpy(d_vals, host_ellpack.vals, sizeof(float)*ellpack_size, cudaMemcpyHostToDevice);
    d_ellpack.vals = d_vals;

    cudaMalloc((void**)&d_cols, sizeof(int)*ellpack_size);
    cudaMemcpy(d_cols, host_ellpack.cols, sizeof(int)*ellpack_size, cudaMemcpyHostToDevice);
    d_crs.cols = d_cols;
    
    // call kernel 
    SpMVM_ELLPACK_kernel<<<dim_grid,dim_block>>>(d_y,  d_ellpack, d_x);
    cudaDeviceSynchronize();

    // store result
    cudaMemcpy(d_res, d_y, sizeof(float)*nrows, cudaMemcpyDeviceToHost);
    
    same = compare_vectors(d_res, y, host_crs.nrows);
    printf("ELLPACK same? %s\n", (same ? "yes" : "no"));

    // free allocations on device
    cudaFree(d_vals);
    cudaFree(d_cols);
    
    //clear previous result
    cudaMemset((void*)d_y, 0, sizeof(float)*nrows);
    
    // SpMVM-SELL on device
    sell d_sell;
    const int sell_size = round((float)host_sell.nnz/(float)host_sell.occ);
    d_sell.nrows = nrows;
    d_sell.ncols = ncols;
    d_sell.nc = host_sell.nc;
    d_sell.nnz = nnz;
    d_sell.C = host_sell.C;
    d_sell.occ = d_sell.occ;
    int* d_cs, *d_cl;

    cudaMalloc((void**)&d_vals, sizeof(float)*sell_size);
    cudaMemcpy(d_vals, host_sell.vals, sizeof(float)*sell_size, cudaMemcpyHostToDevice);
    d_sell.vals = d_vals;

    cudaMalloc((void**)&d_cols, sizeof(int)*sell_size);
    cudaMemcpy(d_cols, host_sell.cols, sizeof(int)*sell_size, cudaMemcpyHostToDevice);
    d_sell.cols = d_cols;
    
    cudaMalloc((void**)&d_cs, sizeof(int)*host_sell.nc);
    cudaMemcpy(d_cs, host_sell.cs, sizeof(int)*host_sell.nc, cudaMemcpyHostToDevice);
    d_sell.cs = d_cs;

    cudaMalloc((void**)&d_cl, sizeof(int)*host_sell.nc);
    cudaMemcpy(d_cl, host_sell.cl, sizeof(int)*host_sell.nc, cudaMemcpyHostToDevice);
    d_sell.cl = d_cl;

    // call kernel 
    SpMVM_SELL_kernel<<<dim_grid,dim_block>>>(d_y,  d_sell, d_x);
    cudaDeviceSynchronize();

    // store result
    cudaMemcpy(d_res, d_y, sizeof(float)*nrows, cudaMemcpyDeviceToHost);
    
    same = compare_vectors(d_res, y, host_crs.nrows);
    printf("SELL same? %s\n", (same ? "yes" : "no"));

    // free allocations on device
    cudaFree(d_vals);
    cudaFree(d_cols);
    cudaFree(d_cs);
    cudaFree(d_cl);

    // free allocations in host and device
    cudaFree(d_res);
    cudaFree(d_y);
    cudaFree(d_x);

    free_CRS(&host_crs);
    free_ELLPACK(&host_ellpack);
    free_SELL(&host_sell);
    
    free(x);
    free(y);
    free(d_res);

    return 0;
}