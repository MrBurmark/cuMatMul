
#define LOG "test.log"

#include <stdio.h>

int main(int argc, char **argv) {

	int w, w1;
	int rs, rs1;
	int cs, cs1;
	int ks, ks1;
	int tb0, tb01;
	int tb1, tb11;
	int ok = 1;
	double t=1e99, t1;
	FILE *fp;

	fp = fopen(LOG, "r");

	while (ok != EOF) {
		ok = fscanf(fp, "width %i, ROW_SIZE %i, COLUMN_SIZE %i, K_SIZE %i, THREAD_BLOCK_0 %i, THREAD_BLOCK_1 %i\nElapsed time = %lf ms\n", &w1, &rs1, &cs1, &ks1, &tb01, &tb11, &t1);
		// printf("width %i, ROW_SIZE %i, COLUMN_SIZE %i, K_SIZE %i, THREAD_BLOCK_0 %i, THREAD_BLOCK_1 %i\nElapsed time = %lf ms\n", w1, rs1, cs1, ks1, tb01, tb11, t1);
		if (t1 < t) {
			t = t1;
			w = w1;
			rs = rs1;
			cs = cs1;
			ks = ks1;
			tb0 = tb01;
			tb1 = tb11;
		}
	}

	printf("width %i, ROW_SIZE %i, COLUMN_SIZE %i, K_SIZE %i, THREAD_BLOCK_0 %i, THREAD_BLOCK_1 %i\nElapsed time = %lf ms\n", w, rs, cs, ks, tb0, tb1, t);

}