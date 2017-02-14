
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"
#include <stdio.h>
#include "matlabfiles.h"
#include "Header.h"
#include <time.h>

#define TX 32
#define ATOMIC 1 // 0 for non-atomic addition

void vInnerProduct(float* result, float* vec1, float* vec2, int size)
{
	for (int i = 0; i < size; i++){
		*result += vec1[i] * vec2[i];
	}
}


int main()
{
	int *N = (int*)calloc(6, sizeof(int));
	N[0] = 10;
	for (int i = 0; i < 6; i++){
		if (i != 0){
			N[i] = N[i - 1] * 10;
		}
	float *u = (float*)calloc(N[i], sizeof(float));
	float *v = (float*)calloc(N[i], sizeof(float));
	float *w = (float*)calloc(1, sizeof(float));
	float *wShared = (float*)calloc(1, sizeof(float));
	float *wCPU = (float*)calloc(1, sizeof(float));
	for (int j = 0; j < N[i]; j++){
		u[j] = 0.25;
		v[j] = 0.75;
	}

	addWithCuda(w, u, v, N[i]);
	addWithCudaShared(wShared, u, v, N[i]); 
	clock_t dotProductStart = clock();
	vInnerProduct(wCPU, u, v, N[i]);
	clock_t dotProductEnd = clock();
	float dotProductTime = (dotProductStart - dotProductEnd) / CLOCKS_PER_SEC;
	printf("Kernel time in CPU (ms): %f\n", dotProductTime*1000);
	printf("N=%d CPU=>%f \n", N[i], *wCPU);
	printf("N=%d GPU=>%f \n", N[i], *w);
	printf("N=%d SharedGPU=>%f \n\n", N[i], *wShared);
	MATFILE *mf;
	int err;;
	//mf = openmatfile("Lab.mat", &err);
	//if (!mf) printf("Can¡¯t open mat file %d\n", err);/*
	//matfile_addmatrix(mf, "Velocity", spdbuffer, IMAX, 1, 0);
	//matfile_addmatrix(mf, "Torque", toqbuffer, IMAX, 1, 0);*/
	//matfile_close(mf);
	free(u);
	free(v);
	free(w);
	free(wShared);
	free(wCPU);
	}
	free(N);
	return 0;
}
