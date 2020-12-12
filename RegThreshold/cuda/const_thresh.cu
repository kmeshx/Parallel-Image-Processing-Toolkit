/*
        Different Image Sizes
        ./const_thresh <img_path> <num_threads> <threshold_val>
*/
//TODO parallelize over intensities??
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#define STB_IMAGE_IMPLEMENTATION
#include "../../utils/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../utils/stb_image_write.h"
#include "../../utils/cycletimer.h"
#define MAX_INTENSITY 256
#define CHANNEL_NUM 1



//assumes k_width, k_height = 3
__global__ void thresh_kernel(uint8_t* old_img, uint8_t* new_img, int img_width, int img_height, int threshold){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int img_idx = j*img_width + i;
    if(old_img[img_idx]> threshold)
        new_img[img_idx] = 255;
    else
        new_img[img_idx] = 0;
}

void calculate_at_threshold(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img, uint8_t* new_img_device, uint8_t* old_img_device, int threshold) {
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(block_side, block_side);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    thresh_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, width, height, threshold);    
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  

}

// eg: ./edge /afs/andrew.cmu.edu/usr12/sbali/private/proj/images/building.jpg 32 <type>
// NOTE shared doesn't support arg block size it is just a place holder here
int main(int argc, char **argv){
    const char *img_file = argv[1];
    int block_side = atoi(argv[2]);
    int width, height, bpp;
    uint8_t* old_img_device;
    uint8_t* new_img_device;
    uint8_t* old_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  

    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);
    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&old_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );

    //cudaMalloc(&kernel_device, sizeof(float) * 9);
    //cudaMemcpy(kernel_device, HSOBEL, sizeof(float) * 9, cudaMemcpyHostToDevice);
    int threshold = atoi(argv[3]);
    double start_time_exc = currentSeconds();
    for(int i=0; i<200; i++){
        calculate_at_threshold(old_img, width, height, block_side, new_img, new_img_device, old_img_device, threshold);
    }
    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time Without Startup: %f\n", duration_exc);
    return 1;
    
}
