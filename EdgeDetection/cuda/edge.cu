/*
    opts: ./edge <img_path> 
*/
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

float HSOBEL_H[3] = {-1.0/3, 0, 1.0/3};
float HSOBEL_V[3] = {1.0/3, 2.0/3, 1.0/3};
float HSOBEL[9] = {1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3, 1.0/3};
//{-1.0/9, -2.0/9, -1.0/9, 0, 0, 0, 1.0/9, 2.0/9, 1.0/9};


__global__ void single_kernel(uint8_t* old_img, uint8_t* new_img, float* kernel, int k_width, int k_height, int img_width, int img_height){
    int i, j, jj, ii;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<img_width && j<img_height){


    float tmp=0.f;
    //int jj_last = min(j + k_height, img_height);
    //int ii_last = min(i + k_width, img_width);
    for(jj = j; jj<j + k_height; jj++){
        for(ii = i; ii< i + k_width; ii++){
            float og_tmp  = tmp;
            if(jj<img_height && ii<img_width){
            tmp += kernel[(jj-j) * k_width + (ii-i)] * (float)(old_img[jj * img_width + ii]);
            }
        }
    }
    new_img[j*img_width + i] = (uint8_t)(tmp);
    }
}



void edge_detect(uint8_t* old_img, int width, int height, int block_side) {
    uint8_t* old_img_device;
    uint8_t* new_img_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);
    float* kernel_device;
    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&old_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&kernel_device, sizeof(float) * 9);
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_device, HSOBEL, sizeof(float) * 9, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(block_side, block_side);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    printf("HERE %d, %d; %d, %d\n", width, height, gridDim.x, gridDim.y);
    single_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, kernel_device, 3, 3, width, height);
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
}


// eg: ./edge /afs/andrew.cmu.edu/usr12/sbali/private/proj/images/building.jpg 32
int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = argv[1];
    int block_side = atoi(argv[2]);
    printf("Starting off ... %s \n", img_file);
    int width, height, bpp;
    uint8_t* old_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    edge_detect(old_img, width, height, block_side);
    return 1;
    
}




