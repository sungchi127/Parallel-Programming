#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int thread_X = 1;
const int thread_Y = 1;
const int resX = 1600;
const int resY = 1200;
const int blockDim_X = 8;
const int blockDim_Y = 8;

__device__ int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re, z_im = c_im;
    int i;
    const float eps = 1e-6;
    if(sqrt(c_re * c_re + c_im * c_im) + eps < 0.25) return maxIterations;
    for (i = 0; i < maxIterations; ++i) {
        if(z_re * z_re + z_im * z_im > 4.f) 
            break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel(int *img, int maxIterations, float stepX, float stepY, float lowerX, float lowerY, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    
    int _X = (blockIdx.x * blockDim.x + threadIdx.x) * thread_X;
    int _Y = (blockIdx.y * blockDim.y + threadIdx.y) * thread_Y;
    
    for(int i = 0; i < thread_X; ++i) {
        for(int j = 0; j < thread_Y; ++j) {
            int thisX = _X + i;
            int thisY = _Y + j;
            if(thisX >= resX || thisY >= resY) continue;
            float x = lowerX + thisX * stepX;
            float y = lowerY + thisY * stepY;  
            img[thisY * pitch + thisX] = mandel(x, y, maxIterations);
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // allocate cuda memory
    // int *hostMem;
    // cudaMallocHost(&hostMem, sizeof(int) * resX * resY);

    int *cudaMem;
    size_t pitch;
    cudaMallocPitch((void**)&cudaMem, &pitch, resX * sizeof(int), resY);


    dim3 numBlocks((resX - 1) / (blockDim_X * thread_X) + 1, (resY - 1) / (blockDim_Y * thread_Y) + 1);
    dim3 numThreads(blockDim_X, blockDim_Y);

    mandelKernel<<<numBlocks, numThreads>>>(cudaMem, maxIterations, stepX, stepY, lowerX, lowerY, pitch / sizeof(int));

    cudaMemcpy2D(img, resX * sizeof(int), cudaMem, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    // memcpy(img, hostMem, resX * resY * sizeof(int));

    // allocate host memory
    // int *hostMem;
    // cudaHostAlloc((void**)&hostMem, pitch * resY * sizeof(int), cudaHostAllocDefault);

    // copy cuda memory to host
    // cudaMemcpy(hostMem, cudaMem, pitch * resY * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaFree(cudaMem);

    // copy host memory to image
    // for(int i = 0; i < resY; ++i) 
        // memcpy(img + i * resX, hostMem + i * pitch, resX * sizeof(int));
}
 