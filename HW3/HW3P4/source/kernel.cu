
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#define TX 32
#define TY 32
#define N 200
#define LEN  20

// scale coordinates onto [-LEN, LEN]
__device__
float scale(int i, int w) { return 2 * LEN*(((1.f*i) / w) - 0.5f); }

__global__ void newtonKernel(float *c, const float *a)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= N) return;
	const float b = a[i] - (a[i] * a[i] * a[i] - a[i]) / (3 * (a[i] * a[i]) - 1);
	c[i] = b - (b * b * b - b) / (3 * (b * b) - 1);
}


// Helper function for using CUDA to add vectors in parallel.
void newton(float *c, const float *a, unsigned int size)
{
	float *dev_a = 0;
	float *dev_c = 0;

	cudaMalloc((void**)&dev_c, size * sizeof(float));
	cudaMalloc((void**)&dev_a, size * sizeof(float));
	cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(dev_c, 0, size * sizeof(float));

	newtonKernel << <(size + TX - 1) / TX, TX >> >(dev_c, dev_a);

	cudaDeviceSynchronize();
	cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
}

__device__
int2 newton2DColor(float x, float y){

	float u = x*x*x - 3 * x*y*y - 1;
	float v = 3 * x*x*y - y*y*y;
	float ux = 3 * x*x - 3 * y*y;
	float uy = -6 * x*y;
	float vx = 6 * x*y;
	float vy = 3 * x*x - 3 * y*y;

	float2 r1, r2, r3;
	r1.x = 1.f; r1.y = 0.f;
	r2.x = -0.5f; r2.y = 0.866f;
	r3.x = -0.5f; r3.y = -0.866f;
	int i = 0;
	float err = 0.01f;
	while ((abs(u) > err || abs(v) > err) && i<50){
		x = x - (u*ux + v*vx) / (ux*ux + vx*vx);
		y = y - (u*uy + v*vy) / (uy*uy + vy*vy);
		u = x*x*x - 3 * x*y*y - 1;
		ux = 3 * x*x - 3 * y*y;
		uy = -6 * x*y;
		v = 3 * x*x*y - y*y*y;
		vx = 6 * x*y;
		vy = 3 * x*x - 3 * y*y;
		i++;
	}
	float dist1 = sqrtf((x - r1.x)*(x - r1.x) + (y - r1.y)*(y - r1.y));
	float dist2 = sqrtf((x - r2.x)*(x - r2.x) + (y - r2.y)*(y - r2.y));
	float dist3 = sqrtf((x - r3.x)*(x - r3.x) + (y - r3.y)*(y - r3.y));
	int color = 0;
	if (dist1 < err)
	{
		color = 1;
	}
	if (dist2 < err)
	{
		color = 2;
	}
	if (dist3 < err)
	{
		color = 3;
	}
	int2 coloridx;
	coloridx.x = color;
	coloridx.y = i;
	return coloridx;


}

__global__
void complexImageKernel(uchar4 *d_out, int w, int h) {
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int r = blockIdx.y*blockDim.y + threadIdx.y;
	if ((c >= w) || (r >= h)) return; // Check if within image bounds
	const int i = c + r*w; // 1D indexing
	const float x0 = scale(c, w);
	const float y0 = scale(r, h);

	const int2 coloridx = newton2DColor(x0, y0);
	int color = coloridx.x;
	int itr = coloridx.y;
	float intensity = 1.f - (1.f*itr) / 50;
	if (color == 1){
		d_out[i].x = intensity * 255;
		d_out[i].y = 0;
		d_out[i].z = 0;
		d_out[i].w = 255;
	}
	if (color == 2){
		d_out[i].x = 0;
		d_out[i].y = intensity * 255;
		d_out[i].z = 0;
		d_out[i].w = 255;
	}
	if (color == 3){
		d_out[i].x = 0;
		d_out[i].y = 0;
		d_out[i].z = intensity * 255;
		d_out[i].w = 255;
	}
}


void kernelLauncher(uchar4 *d_out, int w, int h) {
	const dim3 blockSize(TX, TY);
	const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
	cudaEvent_t startKernel, stopKernel;
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);

	cudaEventRecord(startKernel);
	complexImageKernel << <gridSize, blockSize >> >(d_out, w, h);
	cudaEventRecord(stopKernel);
	cudaEventSynchronize(stopKernel);

	float kernelTimeinMs = 0;
	cudaEventElapsedTime(&kernelTimeinMs, startKernel, stopKernel);
	printf("Kernel time (ms): %f\n", kernelTimeinMs);
}