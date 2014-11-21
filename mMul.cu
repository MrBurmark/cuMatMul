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
    float Psub = 0.0f;
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
    __shared__ float Cmem[K_SIZE][COLUMN_SIZE];

    // Thread index
    int k, r, c, K, R, C, Block_K;
    // int tid = threadIdx.x * blockDim.y + threadIdx.y;
    float Psub;
    int Block_Row = blockIdx.y * ROW_SIZE;
    int Block_Col = blockIdx.x * COLUMN_SIZE;
    int Row = Block_Row + threadIdx.y;
    int Col = Block_Col + threadIdx.x;
    int Row_Bound = min(Block_Row + ROW_SIZE, width);
    int Col_Bound = min(Block_Col + COLUMN_SIZE, width);
    int K_Bound;

    // initialize submatrix to 0
    for (R = Row; R < Row_Bound; R += THREAD_BLOCK_0)
    {
        r = R - Block_Row;
        for (C = Col; C < Col_Bound; C += THREAD_BLOCK_1) 
        {
            c = C - Block_Col;
            Pd[R * width + C] = 0.0f;
        }
    }


    for (Block_K = 0; Block_K < width; Block_K += K_SIZE) {

        K_Bound = min(Block_K + K_SIZE, width);

        // copy in C submatrix
        for (K = Block_K + threadIdx.y; K < K_Bound; K += THREAD_BLOCK_0) 
        {
            k = K - Block_K;
            for (C = Col; C < Col_Bound; C += THREAD_BLOCK_1) 
            {
                c = C - Block_Col;
                Cmem[k][c] = Nd[K * width + C];
                // printf("C[%i,%i]=%.3f\n", c+threadIdx.x, k + threadIdx.y, Cmem[c+threadIdx.x][k + threadIdx.y]);
            }
        }
        // if (Row + Col == 0)printf("++++++++++++++++++++++++++\n");

        // copy in R submatrix
        for (R = Row; R < Row_Bound; R += THREAD_BLOCK_0)
        {
            r = R - Block_Row;
            for (K = Block_K + threadIdx.x; K < K_Bound; K += THREAD_BLOCK_1) 
            {
                k = K - Block_K;
                Rmem[r][k] = Md[R * width + K];
                // printf("R[%i,%i]=%.3f\n", r+threadIdx.y, k + threadIdx.x, Rmem[r+threadIdx.y][k + threadIdx.x]);
            }
        }
        // if (Row + Col == 0)printf("--------------------------\n");

        // ensure data read in before use
        __syncthreads();

        for (C = Col; C < Col_Bound; C += THREAD_BLOCK_1) 
        {
            c = C - Block_Col;
            for (R = Row; R < Row_Bound; R += THREAD_BLOCK_0)
            {
                r = R - Block_Row;

                Psub = 0.0f;
                for (k=0; k < K_Bound - Block_K; k++) 
                {
                    
                    Psub += Rmem[r][k] * Cmem[k][c];
                    // printf("C[%i,%i]=%.3f\n", c + threadIdx.x, k, Cmem[c + threadIdx.x][k]);
                    // printf("R[%i,%i]=%.3f\n", r + threadIdx.y, k, Rmem[r + threadIdx.y][k]);
                }

                // printf("O[%i,%i]=%.3f\n", R, C, Psub);

                Pd[R * width + C] += Psub;
            }
        }
        // ensure data used before overwritten
        __syncthreads();
    }
}
