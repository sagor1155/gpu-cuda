/*
* How to compile (assume cuda is installed at /usr/local/cuda/)
*   nvcc add.cu 
*   ./a.out 
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>
 
#define N 2048

__global__ void add_kernel(int* a, int* b, int*c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
 
void random_ints(int* a, int num) {
    for(int i=0; i<num; i++){
        a[i] = rand() % 1000;
    }
}

void add_in_cpu(int* a, int* b, int*c){
    for(int i=0; i<N; i++){
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    printf("Vector addition in CPU\n");
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size); 

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    clock_t cpu_startTime, cpu_endTime;
    double cpu_ElapseTime=0;
    cpu_startTime = clock();

    float gpu_time;
    cudaEvent_t start, stop;    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Launch add() kernel on GPU
    // add_kernel<<<N, 1>>>(d_a, d_b, d_c);
    add_in_cpu(a, b, c);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    cpu_endTime = clock();
    cpu_ElapseTime = (double)((cpu_endTime - cpu_startTime)/(CLOCKS_PER_SEC/1000.0));
    printf("Time to generate:  %3.3f ms (in cpu)\n", cpu_ElapseTime);

    printf("Time to generate:  %3.3f ms (in gpu)\n", gpu_time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host
    // cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Result: \n");
    for(int i=0; i<15; i++){
        printf("%d ", c[i]);
    }
    printf(".... %d \n", c[N-1]); 

    // Cleanup
    if(a) free(a);    if(b) free(b);    if(c) free(c);
    if(d_a) cudaFree(d_a);    if(d_b) cudaFree(d_b);    if(d_c) cudaFree(d_c);

    return 0;
}
 
