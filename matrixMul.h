#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

//#define WIDTH 32 // Matrix width
#ifndef CHECK
#define CHECK 1 
#endif

#ifndef GLOBAL
#define GLOBAL 0
#endif
#ifndef ROW_SIZE
#define ROW_SIZE 64 // divides matrix width
#endif
#ifndef COLUMN_SIZE
#define COLUMN_SIZE 64 // divides matrix width
#endif
#ifndef K_SIZE
#define K_SIZE 32 // divides matrix width
#endif
#ifndef THREAD_BLOCK_0
#define THREAD_BLOCK_0 16 // divdes width, ROW_SIZE, K_SIZE evenly
#endif
#ifndef THREAD_BLOCK_1
#define THREAD_BLOCK_1 32 // divides width, COLUMN_SIZE, K_SIZE evenly
#endif
#endif // _MATRIXMUL_H_
