/*
* How to compile (assume cuda is installed at /usr/local/cuda/)
*   nvcc add.cu 
*   ./a.out 
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
 
#define N 512

__global__ void add_kernel(int* a, int* b, int*c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}
 
void random_ints(int* a, int num) {
    for(int i=0; i<num; i++){
        a[i] = rand() % 1000;
    }
}

int main(void)
{
    printf("Vector addition using GPU block\n");
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

    clock_t t; 
    t = clock(); 

    // float time;
    // cudaEvent_t start, stop;    
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    
    // Launch add() kernel on GPU
    add_kernel<<<N, 1>>>(d_a, d_b, d_c);
    
    cudaDeviceSynchronize();
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    
    t = clock() - t; 
    double time_taken = (((double)t)/CLOCKS_PER_SEC)/1000.0; // in milli-seconds 
    printf("Took %4.3f milli-seconds to execute \n", time_taken); 

    // printf("Time to generate:  %4.3f ms \n", time);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Result: \n");

    for(int i=0; i<20; i++){
        printf("%d ", c[i]);
    }
    printf("\n"); 

    // Cleanup
    if(a) free(a);    if(b) free(b);    if(c) free(c);
    if(d_a) cudaFree(d_a);    if(d_b) cudaFree(d_b);    if(d_c) cudaFree(d_c);

    return 0;
}
 
