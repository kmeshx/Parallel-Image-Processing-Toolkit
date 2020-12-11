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

#define CHANNEL_NUM 1
#define BLOCK_SIDE 32

float HSOBEL_H[3] = {-1.0/3, 0, 1.0/3};
float HSOBEL_V[3] = {1.0/3, 2.0/3, 1.0/3};
float HSOBEL[9] = {-1.0/9, -2.0/9, -1.0/9, 0, 0, 0, 1.0/9, 2.0/9, 1.0/9};


__global__ void one_pass_kernel(uint8_t* old_img, uint8_t* new_img, float* kernel, int k_width, int k_height, int img_width, int img_height){
    int i, j, jj, ii;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i>=img_width || j>=img_height){return;}

    double tmp=0;
    int jj_last = min(j + k_height, img_height);
    int ii_last = min(i + k_width, img_width);
    for(jj = j; jj<jj_last; jj++){
        for(ii = i; ii< ii_last; ii++){
            tmp += kernel[(jj-j) * k_width + (ii-i)] * (old_img[jj * img_width + ii]);
        }
    }
    new_img[j*img_width + i] = (uint8_t)tmp;
}

//TODO why isn't 2d indexing working??
void edge_detect(uint8_t* old_img, int width, int height) {

    uint8_t* old_img_device;
    uint8_t* new_img_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);
    float* kernel_device;

    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&old_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&kernel_device, sizeof(float) * 9);

    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_device, HSOBEL, sizeof(float) * 9, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(BLOCK_SIDE, BLOCK_SIDE);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    printf("HERE %d, %d; %d, %d\n", gridDim.x, gridDim.y, threadsPerBlock.x, threadsPerBlock.y);
    one_pass_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, kernel_device, 3, 3, width, height);
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    //cudaMemcpy(new_img, new_img_device, sizeof(Point) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
}


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = argv[1];
    printf("Starting off ... %s \n", img_file);

    int width, height, bpp;
    uint8_t* old_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    edge_detect(old_img, width, height);
    return 1;
    
}