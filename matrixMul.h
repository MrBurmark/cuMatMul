#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

//#define WIDTH 32 // Matrix width

#define GLOBAL 0
#define ROW_SIZE 64 // divides matrix width
#define COLUMN_SIZE 64 // divides matrix width
#define K_SIZE 32 // divides matrix width
#define THREAD_BLOCK_0 8 // divides width, ROW_SIZE, K_SIZE evenly
#define THREAD_BLOCK_1 32 // divides width, COLUMN_SIZE, K_SIZE evenly

#endif // _MATRIXMUL_H_

