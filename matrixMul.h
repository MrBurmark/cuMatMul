#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#define WIDTH 32 // Matrix width
#define ROW_SIZE 16
#define COLUMN_SIZE 16
#define K_SIZE 16
#define THREAD_BLOCK_0 16 // divides ROW_SIZE, K_SIZE evenly
#define THREAD_BLOCK_1 16 // divides COLUMN_SIZE, K_SIZE evenly

#endif // _MATRIXMUL_H_

