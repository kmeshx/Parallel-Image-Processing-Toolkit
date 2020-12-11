#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 1
#define BLOCK_SIDE 5

float HSOBEL_H[3] = {-1.0/3, 0, 1.0/3};
float HSOBEL_V[3] = {1.0/3, 2.0/3, 1.0/3};
float HSOBEL[9] = {-1.0/9, -2.0/9, -1.0/9, 0, 0, 0, 1.0/9, 2.0/9, 1.0/9};


__global__ void one_pass_kernel(uint8_t* &old_img, uint8_t* &new_img, float kernel[9], int k_width, int k_height, int img_width, int img_height){
    int jj, ii;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double tmp;

    for(jj = j; jj<jj_last; jj++){
        for(ii = i; ii< ii_last; ii++){
            tmp += kernel[(jj-j) * k_width + (ii-i)] * (old_img[jj * img_width + ii]);
        }
    }
    new_img[i*img_height + j] = (uint8_t)tmp;
}

void edge_detect(uint8_t* old_img, int width, int height) {
    dim3 threadsPerBlock(BLOCK_SIDE, BLOCK_SIDE, 1);
    const int NUM_BLOCKS_X = (width+threadsPerBlock.x-1)/threadsPerBlock.x;
    const int NUM_BLOCKS_Y = (height+threadsPerBlock.y-1)/threadsPerBlock.y;
    dim3 gridDim(NUM_BLOCKS_X, NUM_BLOCKS_Y, 1);
    uint8_t* old_img_device;
    uint8_t* new_img_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);
    printf("ENTERED");
    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&old_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    printf("COPIED");
    k_means_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, HSOBEL, 3, 3, width, height);
    printf("DONE");
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);

    //cudaMemcpy(new_img, new_img_device, sizeof(Point) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
    printf("Finished k-means\n");
}


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = "cs_test1.jpg";
    int width, height, bpp;
    printf("READING");
    uint8_t* old_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    printf("READ");
    edge_detect(old_img, width, height);
    return 1;
    
}