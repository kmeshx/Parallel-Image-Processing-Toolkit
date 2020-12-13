/*
        Different Image Sizes
        1. atomic add 
        2. use 1 thread to combine all
        3. scan
        4. making blocksize = MAX_INTENSITY
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
#define BLOCK_SIDE 5
#define SCAN_BLOCK_DIM 256
#include "inclusiveScan.cu_inl"
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)
/*
__global__ void set_kernel(uint8_t* old_img, uint8_t* new_img, int** histograms, int num_blocks_x, int img_width, int img_height){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    int linearThreadIndex =  threadIdx.y * blockDim.x + threadIdx.x;
    if(i>=img_width || j>=img_height) return;
    histograms[old_img[idx]][linearThreadIndex] = 1;
    if(new_img[i][j] ==current_intensity)

}
    //assumes k_width, k_height = 3
__global__ void otsu_kernel(uint8_t* old_img, uint8_t* new_img, int** histograms, int num_blocks_x, int img_width, int img_height){
    int i, j, jj, ii, support_id, img_id, n_img_id;


    __shared__ uint input[SCAN_BLOCK_DIM];
    __shared__ uint output[SCAN_BLOCK_DIM];
    __shared__ uint scratch[2 * SCAN_BLOCK_DIM];
    cur_elem = old_img[img_width * j + i];
    input[cur_elem] += 1;
    __syncthreads();
    sharedMemExclusiveScan(linearThreadIndex, input, output, scratch, BLOCKSIZE);
    
    __syncthreads();
    for(int j=0; j< MAX_INTENSITY; j++){
        
      
        
        histograms[j][num_blocks_x * blockIdx.y + blockIdx.x] += 1;
       
    }
}
*/


// modified from https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void set_histograms(uint8_t* old_img, uint8_t* new_img, int* histograms, int* sum_vals, int img_width, int img_height){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i>=img_width || j>=img_height) return;
    int cur_color = old_img[img_width * j + i];
    atomicAdd(&histograms[cur_color], 1);
    atomicAdd(&sum_vals[cur_color], cur_color);

}

//TODO convert this to the variable
__global__ void otsu_single(int* global_threshold, uint8_t* old_img, uint8_t* new_img, int* histograms,  int* sum_vals, int img_width, int img_height){

    __shared__ uint input_histogram[SCAN_BLOCK_DIM];
    __shared__ uint output_histogram[SCAN_BLOCK_DIM];
    __shared__ uint scratch_histogram[2 * SCAN_BLOCK_DIM];

    __shared__ uint input_sum[SCAN_BLOCK_DIM];
    __shared__ uint output_sum[SCAN_BLOCK_DIM];
    __shared__ uint scratch_sum[2 * SCAN_BLOCK_DIM];
    __shared__ float max[1];
    __shared__ int threshold;
    
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id==0){
        max[0] = 0;
    }
    __syncthreads();
    input_histogram[id] = histograms[id];
    input_sum[id] = sum_vals[id];
    //printf("HERE %d", max[0]);

    __syncthreads();
    sharedMemInclusiveScan(id, input_histogram, output_histogram, scratch_histogram, SCAN_BLOCK_DIM);
    __syncthreads();

    sharedMemInclusiveScan(id, input_sum, output_sum, scratch_sum, SCAN_BLOCK_DIM);
    __syncthreads();
    float p1_num = output_histogram[id] + 0.001;
    float p2_num = output_histogram[SCAN_BLOCK_DIM-1] - p1_num + 0.002;

    int total_sum = output_sum[SCAN_BLOCK_DIM-1];
    int p1_sum = output_sum[id];
    int p2_sum = total_sum - p1_sum;
    float p1_mu = ((float)(input_sum[id]))/p1_num;
    float p2_mu = ((float)p2_sum)/p2_num;
    float mu_diff = (p1_mu - p2_mu)/256;
    float var = p1_num * p2_num * mu_diff * mu_diff;
    
    __syncthreads();
    atomicMax(max, var);
    __syncthreads();
    if(var==max[0]){
        threshold = id;
    }
    __syncthreads();

    *global_threshold = threshold;


}

__global__ void set_val(int* threshold, uint8_t* old_img, uint8_t* new_img, int img_width, int img_height){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int id = j*img_width + i;
    if(i>=img_width && j>=img_width) return;
    //printf("HEREH %d", *threshold);
    if(old_img[id] > *threshold){
        new_img[id] = 255;
    }
    else{
        new_img[id] = 0;
    }
}

void set_otsu(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img, uint8_t* new_img_device, uint8_t* old_img_device, int* histograms, int* sum_vals) {
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    int* global_threshold_device;
    cudaMalloc(&global_threshold_device, sizeof(int));
    int* threshold = (int*)malloc(sizeof(int));

    dim3 threadsPerBlock(block_side, block_side);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    //(uint8_t* old_img, uint8_t* new_img, int* histograms, int* sum_vals, int img_width, int img_height)
    set_histograms<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, histograms, sum_vals, width, height); 
    
    //dim3 threadsPerBlock(MAX_INTENSITY);
    //dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    otsu_single<<<1, MAX_INTENSITY>>>(global_threshold_device, old_img_device, new_img_device, histograms, sum_vals, width, height);    
    

    set_val<<<gridDim, threadsPerBlock>>>(global_threshold_device, old_img_device, new_img_device, width, height); 

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
    int* histograms, *sum_vals;
    uint8_t* old_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);

    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&old_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );

    cudaMalloc(&histograms, sizeof(int) * MAX_INTENSITY);
    cudaMalloc(&sum_vals, sizeof(int) * MAX_INTENSITY );

    //cudaMalloc(&kernel_device, sizeof(float) * 9);
    //cudaMemcpy(kernel_device, HSOBEL, sizeof(float) * 9, cudaMemcpyHostToDevice);
    char type = argv[3][0];
    float start_time_exc = currentSeconds();
    for(int i=0; i<1; i++){
        set_otsu(old_img, width, height, block_side, new_img, new_img_device, old_img_device, histograms, sum_vals);

    }
    float end_time = currentSeconds();
    float duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time Without Startup: %f\n", duration_exc);
    return 1;
    
}
