/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc add.cu 
 *   ./a.out 
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

__global__ void add_kernel(int* a, int* b, int*c){
	*c = *a + *b;
}

int main(void)
{
	printf("My First CUDA Application\n");
	int a, b, c;
	int *d_a, *d_b, *d_c;
	a = 10; b=20; c=0;
	int size = sizeof(int);

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	float time;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	// Launch add() kernel on GPU
	add_kernel<<<1, 1>>>(d_a, d_b, d_c);
	
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	
	printf("Time to generate:  %3.1f ms \n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy result back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	printf("Result is: %d\n", c);

	// Cleanup
	if(d_a) cudaFree(d_a);
	if(d_b) cudaFree(d_b);
	if(d_c) cudaFree(d_c);

	return 0;
}
