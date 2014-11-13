//
// Compile: nvcc matrixMul.cu matrixMul_gold.cpp -o mMul
// Use: mMul
//
#include <stdio.h>
#include "matrixMul.h"

// includes, kernels
#include "mMul.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

extern "C"
void computeGold( float*, const float*, const float*, unsigned int w);

void printMatrix(float *M, int width) {
    int i, j;
    for (i=0;i<width;i++){
        for (j=0;j<width;j++){
            printf("%.3f ",M[i*width+j]);
        }
        printf("\n");
    }
    printf("\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // set seed for rand()
    srand(2006);

    if (argc != 2) {
        printf("usage: ./mMul [size of matrix]\n");
        exit(1);
    }

    int width = atoi(argv[1]); 

    // allocate host memory for matrices M and N
    unsigned int size_M = width * width;
    unsigned int mem_size_M = sizeof(float) * size_M;
    float* h_M = (float*)malloc(mem_size_M);
    unsigned int size_N = width * width;
    unsigned int mem_size_N = sizeof(float) * size_N;
    float* h_N = (float*)malloc(mem_size_N);

    // initialize host memory
    randomInit(h_M, size_M);
    randomInit(h_N, size_N);

    // allocate device memory
    float* d_M;
    cudaMalloc((void**) &d_M, mem_size_M);
    float* d_N;
    cudaMalloc((void**) &d_N, mem_size_N);

    cudaEventRecord(start, 0);

    // copy host memory to device
    cudaMemcpy(d_M, h_M, mem_size_M,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, mem_size_N,
                              cudaMemcpyHostToDevice);

    // allocate device memory for result
    unsigned int size_P = width * width;
    unsigned int mem_size_P = sizeof(float) * size_P;
    float* d_P;
    cudaMalloc((void**) &d_P, mem_size_P);

    // allocate host memory for the result
    float* h_P = (float*) malloc(mem_size_P);

    // printMatrix(h_N,width);
    // printMatrix(h_M,width);
    

    // setup execution parameters
    dim3 blocks(ceil(width/(double)ROW_SIZE), ceil(width/(double)COLUMN_SIZE), 1);
    dim3 threads(THREAD_BLOCK_0, THREAD_BLOCK_1, 1);

    // kernel warmup
    // matrixMulKernelGlobal<<< blocks, threads >>>(d_M, d_N, d_P, width);
    matrixMulKernelShared<<< blocks, threads >>>(d_M, d_N, d_P, width);
    cudaThreadSynchronize();
    
    // copy result from device to host
    cudaMemcpy(h_P, d_P, mem_size_P,
                              cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Elapsed time = %f ms\n", time);

    // compute reference solution
    float* reference = (float*)malloc(mem_size_P);
    computeGold(reference, h_M, h_N, width);

    // check result
    printDiff(reference, h_P, width, width, 100, 1.0e-5f);

    // clean up memory
    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) 
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]) / data1[k];
            if (fDiff > fListTol) 
            {                
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    printf(" \n  Total Errors = %d\n\n", error_count);
}

