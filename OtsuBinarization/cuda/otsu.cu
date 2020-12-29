/*
        Different Image Sizes
        1. atomic add 
        2. use 1 thread to combine all
        3. scan
        4. making blocksize = MAX_INTENSITY
        for some reason has different writing times ??, otherwise faster
        had faster despite the input array being of the same size
*/

/*
OTSU CUDA

Building

Not stream
1. 0.633789 : 1
2. 0.323242: 2
3. 0.245117: 4
4. 0.221680 : 8
5. 0.217773: 16
6. 0.220703: 32

Stream:
1.  0.545898: 1
2. 0.269531 : 2
3. 0.182617: 4
4. 0.153320: 8
5. 0.153320: 16
6. 0.154297: 32


Other:

Not Stream
0.147461
 0.071289
 0.051758
 0.044922
 0.044922
0.044922


Stream:
0.123047
0.049805
0.030273
0.029297
0.029297
0.029297

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
#define NCHUNK 400

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

    __shared__ int input_histogram[SCAN_BLOCK_DIM];
    __shared__ int output_histogram[SCAN_BLOCK_DIM];
    __shared__ int scratch_histogram[2 * SCAN_BLOCK_DIM];

    __shared__ int input_sum[SCAN_BLOCK_DIM];
    __shared__ int output_sum[SCAN_BLOCK_DIM];
    __shared__ int scratch_sum[2 * SCAN_BLOCK_DIM];
    __shared__ float max[1];
    __shared__ int threshold;
    
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id==0){
        max[0] = 0;
    }
    //__syncthreads();
    input_histogram[id] = histograms[id];
    input_sum[id] = sum_vals[id];
    //printf("HERE %d", max[0]);

    __syncthreads();
    sharedMemInclusiveScan(id, input_histogram, output_histogram, scratch_histogram, input_sum, output_sum, scratch_sum, SCAN_BLOCK_DIM);

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
    
    atomicMax(max, var);
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

__global__ void set_histograms_zero(int* histograms,int* sum_vals){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    histograms[i] = 0;
    sum_vals[i]=0;


}

void set_otsu(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img_device, uint8_t* old_img_device, int* histograms, int* sum_vals) {
    int* global_threshold_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);
    cudaMalloc(&histograms, sizeof(int) * MAX_INTENSITY);
    cudaMalloc(&sum_vals, sizeof(int) * MAX_INTENSITY );
    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&old_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&global_threshold_device, sizeof(int));
    float start_time_exc = currentSeconds();
    dim3 threadsPerBlock(block_side, block_side);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);

    for(int i=0; i< NCHUNK; i++){
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
   
    int* threshold = (int*)malloc(sizeof(int));

    //(uint8_t* old_img, uint8_t* new_img, int* histograms, int* sum_vals, int img_width, int img_height)
    set_histograms<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, histograms, sum_vals, width, height); 
    //dim3 threadsPerBlock(MAX_INTENSITY);
    //dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    otsu_single<<<1, MAX_INTENSITY>>>(global_threshold_device, old_img_device, new_img_device, histograms, sum_vals, width, height);    
    set_val<<<gridDim, threadsPerBlock>>>(global_threshold_device, old_img_device, new_img_device, width, height); 
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    printf("OK");
    }
    //stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
    float end_time = currentSeconds();
    cudaFree(histograms);
    cudaFree(sum_vals);
    cudaFree(new_img_device);
    cudaFree(old_img_device);
    
    float duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time Without Startup: %f\n", duration_exc);
}

void set_otsu_streamed(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img_device, uint8_t* old_img_device, int* histograms, int* sum_vals) {
    cudaStream_t stream[NCHUNK];
    int i;
    int* global_threshold_device;
    int* threshold = (int*)malloc(sizeof(int));
    dim3 threadsPerBlock(block_side, block_side);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    int col_di = MAX_INTENSITY*sizeof(int);
    int img_di = height * width*CHANNEL_NUM;
    uint8_t* new_img = (uint8_t*)malloc(NCHUNK* sizeof(uint8_t) * height * width * CHANNEL_NUM);
    /*for(i=0; i<NCHUNK; i++){
        new_img[i] = (uint8_t*)malloc();
    }
    */
    cudaMalloc(&global_threshold_device, NCHUNK * sizeof(int));
    int col_shift, img_shift;
    cudaMalloc(&histograms, NCHUNK * sizeof(int) * MAX_INTENSITY);
    cudaMalloc(&sum_vals, NCHUNK * sizeof(int) * MAX_INTENSITY );
    cudaMalloc(&new_img_device,  NCHUNK * sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&old_img_device, NCHUNK * sizeof(uint8_t) * height * width*CHANNEL_NUM );
    
    for(i = 0; i < NCHUNK;i++){
        cudaStreamCreate(&stream[i]);
    }

    float start_time_exc = currentSeconds();
    col_shift = 0;
    img_shift = 0;

    for(i=0; i<NCHUNK; i++){
        img_shift += img_di;
        cudaMemcpyAsync(old_img_device + img_shift , old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice, stream[i]);
    }

    col_shift = 0;
    img_shift = 0;

    for(i=0;i<NCHUNK;i++) {
        col_shift += col_di;
        img_shift += img_di;
        set_histograms_zero<<<1, MAX_INTENSITY, 0, stream[i]>>>(histograms + col_shift, sum_vals+col_shift);
        set_histograms<<<gridDim, threadsPerBlock, 0, stream[i]>>>(old_img_device + img_shift, new_img_device+img_shift, histograms+col_shift, sum_vals+col_shift, width, height); 
        otsu_single<<<1, MAX_INTENSITY, 0, stream[i]>>>(global_threshold_device + i*sizeof(int), old_img_device+ img_shift, new_img_device+img_shift, histograms+col_shift, sum_vals+col_shift, width, height);    
        set_val<<<gridDim, threadsPerBlock, 0, stream[i]>>>(global_threshold_device+ i*sizeof(int), old_img_device+ img_shift, new_img_device+img_shift, width, height); 
    }

    
    img_shift = 0;
    for(i=0;i<NCHUNK;i++) {
        img_shift += img_di;
        cudaMemcpyAsync(new_img+img_shift, new_img_device + img_shift, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost, stream[i]);
    }    
    
    for(i=0; i<NCHUNK; i++)
    {
        cudaStreamSynchronize(stream[i]);

    }
    
    
    for(i=0;i<NCHUNK;i++) {
        printf("OK");
    }   
    float end_time = currentSeconds();
    
    for(i=0;i<NCHUNK;i++) {
        stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img +img_di , width*CHANNEL_NUM);  
    }   
    
    float duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time Without Startup: %f\n", duration_exc);
    

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

    //cudaMalloc(&kernel_device, sizeof(float) * 9);
    //cudaMemcpy(kernel_device, HSOBEL, sizeof(float) * 9, cudaMemcpyHostToDevice);
    char type = argv[3][0];
    //set_otsu_streamed(old_img, width, height, block_side, new_img_device, old_img_device, histograms, sum_vals);
    
    if(type=='s'){
        set_otsu_streamed(old_img, width, height, block_side, new_img_device, old_img_device, histograms, sum_vals);
    }
    else{
        set_otsu(old_img, width, height, block_side, new_img_device, old_img_device, histograms, sum_vals);

    }
    
    cudaFree(histograms);
    cudaFree(sum_vals);
    cudaFree(new_img_device);
    cudaFree(old_img_device);
    
    return 1;
    
}
