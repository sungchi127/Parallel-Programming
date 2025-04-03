#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hostFE.h"
#include "helper.h"

void hostFE_(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageWidth * imageHeight;

    int args[3] = {imageWidth, imageHeight, filterWidth};
    
    cl_mem image_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), (void*)inputImage, NULL);
    cl_mem filter_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filterSize * sizeof(float), (void*)filter, NULL);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, NULL);
    cl_mem arg_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 3 * sizeof(int), (void*)args, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&filter_buffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image_buffer);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_buffer);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&arg_buffer);

    size_t global_work_size[1] = {imageSize};
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, NULL);
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    status = clEnqueueReadBuffer(commandQueue, output_buffer, CL_TRUE, 0, imageSize * sizeof(float), outputImage, NULL, NULL, NULL);

}

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageWidth * imageHeight;

    int args[3] = {imageWidth, imageHeight, filterWidth};
    
    cl_mem image_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), (void*)inputImage, NULL);
    cl_mem filter_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, filterSize * sizeof(float), (void*)filter, NULL);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, NULL);
    cl_mem arg_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 3 * sizeof(int), (void*)args, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution2D", NULL);

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&filter_buffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image_buffer);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output_buffer);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&arg_buffer);

    size_t global_work_size[2] = {imageWidth, imageHeight};
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, NULL);
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    status = clEnqueueReadBuffer(commandQueue, output_buffer, CL_TRUE, 0, imageSize * sizeof(float), outputImage, NULL, NULL, NULL);

}