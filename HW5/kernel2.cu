#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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


__global__ void mandelKernel(int *img, int maxIterations, float stepX, float stepY, float lowerX, float lowerY, int resX, int resY, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(thisX < resX && thisY < resY) {
        float x = lowerX + thisX * stepX;
        float y = lowerY + thisY * stepY;    
        img[thisY * pitch + thisX] = mandel(x, y, maxIterations);
    }
    
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // allocate cuda memory
    int *cudaMem;
    size_t pitch;
    cudaMallocPitch((void**)&cudaMem, &pitch, resX * sizeof(int), resY);
    pitch /= sizeof(int);

    const int blockDim_X = 16;
    const int blockDim_Y = 16;

    dim3 numBlocks((resX - 1) / blockDim_X + 1, (resY - 1) / blockDim_Y + 1);
    dim3 numThreads(blockDim_X, blockDim_Y);

    mandelKernel<<<numBlocks, numThreads>>>(cudaMem, maxIterations, stepX, stepY, lowerX, lowerY, resX, resY, pitch);

    // allocate host memory
    int *hostMem;
    cudaHostAlloc((void**)&hostMem, pitch * resY * sizeof(int), cudaHostAllocDefault);

    // copy cuda memory to host
    cudaMemcpy(hostMem, cudaMem, pitch * resY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(cudaMem);

    // copy host memory to image
    for(int i = 0; i < resY; ++i) 
        memcpy(img + i * resX, hostMem + i * pitch, resX * sizeof(int));
}
 