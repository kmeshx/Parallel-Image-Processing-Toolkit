/*
        Different Image Sizes
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
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define MAX_INTENSITY 256
#define CHANNEL_NUM 3
#define BLOCK_SIDE 5
#define SCAN_BLOCK_DIM BLOCK_SIDE
#include "exclusiveScan.cu_inl"
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)


//assumes k_width, k_height = 3
__global__ void otsu_kernel(uint8_t* old_img, uint8_t* new_img, int* histograms[MAX_INTENSITY], int num_blocks_x, int img_width, int img_height){
    int i, j, jj, ii, support_id, img_id, n_img_id;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    int linearThreadIndex =  threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ uint input[MAX_INTENSITY][BLOCK_SIDE];
    __shared__ uint output[MAX_INTENSITY][BLOCK_SIDE];
    __shared__ uint scratch[MAX_INTENSITY][2 * BLOCK_SIDE];
    cur_elem = old_img[img_width * j + i];
    input[cur_elem][linearThreadIndex] += 1;

    __syncthreads();
    sharedMemExclusiveScan(linearThreadIndex, input, output, scratch, BLOCKSIZE);
    __syncthreads();

    for(int j=0; j< MAX_INTENSITY; j++){
        histograms[cur_elem][num_blocks_x * blockIdx.y + blockIdx.x] += 1;
    }
}
void edge_detect_shared(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img, uint8_t* new_img_device, uint8_t* old_img_device) {
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(THREADS_PER_BLK, THREADS_PER_BLK);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    shared_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, width, height);    
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
    char type = argv[3][0];
    double start_time_exc = currentSeconds();
    for(int i=0; i<200; i++){
        if(type=='n')
            edge_detect_single(old_img, width, height, block_side, new_img, new_img_device, old_img_device);
        else if(type=='s')
            edge_detect_shared(old_img, width, height, block_side, new_img, new_img_device, old_img_device);

    }
    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time Without Startup: %f\n", duration_exc);
    return 1;
    
}
