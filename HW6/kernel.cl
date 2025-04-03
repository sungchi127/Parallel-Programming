__kernel void convolution(__global float *filter, __global float *image, __global float *output, __global int *args) {
    int imageWidth = args[0];
    int imageHeight = args[1];
    int filterWidth = args[2];

    int id = get_global_id(0);
    
    int i = id / imageWidth;
    int j = id % imageWidth;

    int halffilterSize = filterWidth / 2;
    
    float sum = 0;
    for(int k = -halffilterSize; k <= halffilterSize; ++k) {
        for(int l = -halffilterSize; l <= halffilterSize; ++l) {
            if(filter[(k + halffilterSize) * filterWidth + l + halffilterSize] == 0.0) continue;
            if(i + k >= 0 && i + k < imageHeight && 
               j + l >= 0 && j + l < imageWidth) {

                sum += image[(i + k) * imageWidth + j + l] * filter[(k + halffilterSize) * filterWidth + l + halffilterSize];

            }
        }
    }
    output[i * imageWidth + j] = sum;
}

__kernel void convolution2D(__global float *filter, __global float *image, __global float *output, __global int *args) {
    int imageWidth = args[0];
    int imageHeight = args[1];
    int filterWidth = args[2];
    
    int i = get_global_id(1);
    int j = get_global_id(0);

    int halffilterSize = filterWidth / 2;
    
    float sum = 0;
    for(int k = -halffilterSize; k <= halffilterSize; ++k) {
        for(int l = -halffilterSize; l <= halffilterSize; ++l) {
            if(filter[(k + halffilterSize) * filterWidth + l + halffilterSize] == 0.0) continue;
            if(i + k >= 0 && i + k < imageHeight && 
               j + l >= 0 && j + l < imageWidth) {

                sum += image[(i + k) * imageWidth + j + l] * filter[(k + halffilterSize) * filterWidth + l + halffilterSize];

            }
        }
    }
    output[i * imageWidth + j] = sum;
}
