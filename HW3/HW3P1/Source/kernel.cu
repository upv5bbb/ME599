
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"
#include <stdio.h>
#include "Header.h"
#define TX 32
#define ATOMIC 1 // 0 for non-atomic addition

__global__ void addKernel(float *dot, float *a, float *b)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const int s_idx = threadIdx.x;
	__shared__ float s_prod[TX];
	s_prod[s_idx] = a[i] * b[i];
	__syncthreads();

	if (s_idx == 0) {
		float blockSum = 0.0;
		for (int j = 0; j < blockDim.x; ++j){
			blockSum += s_prod[j];
		}
		printf("Block_%d, blockSum = %f\n", blockIdx.x, blockSum);
		if (ATOMIC) {
			atomicAdd(dot, blockSum);
		}
		else {
			*dot += blockSum;
		}
	}
}
__global__ void addKernelShared(float *dot, float *a, float *b)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (ATOMIC) {
		atomicAdd(dot, a[i] * b[i]);
	}
	else {
		*dot += a[i]*b[i];
	}
}


// Helper function for using CUDA to add vectors in parallel.
void addWithCuda(float *dot, const float *a, const float *b, unsigned int size)
{
    float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c ;
	cudaEvent_t startKernel, stopKernel;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);

    
    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_a, size * sizeof(float));
	cudaMalloc((void**)&dev_b, size * sizeof(float));
	cudaMalloc((void**)&dev_c, sizeof(float));
	cudaMemset(dev_c, 0, sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(startKernel);
	addKernel <<< (size + TX - 1) / TX, TX >>>(dev_c, dev_a, dev_b);
	cudaEventRecord(stopKernel);
    
	cudaEventSynchronize(stopKernel);

	float kernelTimeinMs = 0;
	cudaEventElapsedTime(&kernelTimeinMs, startKernel, stopKernel);
	printf("Kernel time (ms): %f\n", kernelTimeinMs);

    cudaMemcpy(dot, dev_c,  1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}
void addWithCudaShared(float *dot, const float *a, const float *b, unsigned int size)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c;
	cudaEvent_t startKernel, stopKernel;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);


	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc((void**)&dev_a, size * sizeof(float));
	cudaMalloc((void**)&dev_b, size * sizeof(float));
	cudaMalloc((void**)&dev_c, sizeof(float));
	cudaMemset(dev_c, 0, sizeof(float));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(startKernel);
	addKernelShared << < (size + TX - 1) / TX, TX >> >(dev_c, dev_a, dev_b);
	cudaEventRecord(stopKernel);

	cudaEventSynchronize(stopKernel);

	float kernelTimeinMs = 0;
	cudaEventElapsedTime(&kernelTimeinMs, startKernel, stopKernel);
	printf("Kernel time with Shared Memory (ms): %f\n", kernelTimeinMs);

	cudaMemcpy(dot, dev_c, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
}
