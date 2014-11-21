#!/bin/bash

for rs in 32 64; do
	for cs in 32 64 128; do
		for ks in 16 32 64; do
			for tb0 in 8 16; do
				for tb1 in 16 32; do
					nvcc -D CHECK=0 -D GLOBAL=0 -D ROW_SIZE=$rs -D COLUMN_SIZE=$cs -D K_SIZE=$ks -D THREAD_BLOCK_0=$tb0 -D THREAD_BLOCK_1=$tb1 matrixMul.cu matrixMul_gold.cpp -o mMul -O3 -arch=compute_20 -code=sm_20,sm_30,sm_35
					./mMul 1024 >> test.log
				done
			done
		done
	done
done
