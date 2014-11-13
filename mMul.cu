#include <stdio.h>
#include "matrixMul.h"

///////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: P = M * N
///////////////////////////////////////////////////////////////////////////////

__global__ void
matrixMulKernelGlobal( float* Md, float* Nd, float* Pd, int width)
{

    // Thread index
    int k;
    float Psub = 0.0;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < width && Col < width) {
    	for (k = 0; k < width; ++k) {
	        Psub += Md[Row * width + k] * Nd[k * width + Col];
	    }

	    Pd[Row * width + Col] = Psub;
    }
}

__global__ void
matrixMulKernelShared( float* Md, float* Nd, float* Pd, int width)
{
    __shared__ float Rmem[ROW_SIZE][K_SIZE];
    __shared__ float Cmem[COLUMN_SIZE][K_SIZE];

    // Thread index
    int k, r, c, K, R, C, K_Block;
    // int tid = threadIdx.x * blockDim.y + threadIdx.y;
    float Psub;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // initialize submatrix to 0
    for (c=0; c < COLUMN_SIZE; c += THREAD_BLOCK_1) {
        C = c + Col;
        for (r=0; r < ROW_SIZE; r += THREAD_BLOCK_0) {
            R = r + Row;
            if (R < width && C < width) {

                Pd[R * width + C] = 0.0f;
            }
        }
    }


    for (K_Block=0; K_Block < width; K_Block += K_SIZE) {

        // copy in C submatrix
        for (k=threadIdx.y; k < K_SIZE; k += THREAD_BLOCK_0) {
            K = k + K_Block;
            for (c=0; c < COLUMN_SIZE; c += THREAD_BLOCK_1) {
                C = c + Col;
                if (K < width && C < width) {
                    Cmem[c + threadIdx.x][k] = Nd[K * width + C];
                    // printf("C[%i,%i]=%.3f\n", c+threadIdx.x, k, Cmem[c+threadIdx.x][k]);
                }
            }
        }
        // if (Row + Col == 0)printf("++++++++++++++++++++++++++\n");

        // copy in R submatrix
        for (r=0; r < ROW_SIZE; r += THREAD_BLOCK_0) {
            R = r + Row;
            for (k=threadIdx.x; k < K_SIZE; k += THREAD_BLOCK_1) {
                K = k + K_Block;
                if (K < width && R < width) {
                    Rmem[r + threadIdx.y][k] = Md[R * width + K];
                    // printf("R[%i,%i]=%.3f\n", r+threadIdx.y, k, Rmem[r+threadIdx.y][k]);
                }
            }
        }
        // if (Row + Col == 0)printf("--------------------------\n");

        // ensure data read in before use
        __syncthreads();

        for (c=0; c < COLUMN_SIZE; c += THREAD_BLOCK_1) {
            C = c + Col;
            for (r=0; r < ROW_SIZE; r += THREAD_BLOCK_0) {
                R = r + Row;
                if (R < width && C < width) {

                    Psub = 0.0f;
                    for (k=0; k < K_SIZE && k < width - K_Block; k++) {
                        
                        Psub += Rmem[r + threadIdx.y][k] * Cmem[c + threadIdx.x][k];
                        // printf("C[%i,%i]=%.3f\n", c + threadIdx.x, k, Cmem[c + threadIdx.x][k]);
                        // printf("R[%i,%i]=%.3f\n", r + threadIdx.y, k, Rmem[r + threadIdx.y][k]);
                    }

                    // printf("O[%i,%i]=%.3f\n", R, C, Psub);

                    Pd[R * width + C] += Psub;
                }
            }
        }
        // ensure data used before overwritten
        __syncthreads();
    }
}