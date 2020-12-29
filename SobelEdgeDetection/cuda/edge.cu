/*
    opts: ./edge /afs/andrew.cmu.edu/usr12/sbali/private/proj/images/building.jpg 32 <s|n>
*/
/* 
    do 5/10 images of increasing sizes for all
    analyze on both shared, not shared
    add timing without memory copying 
    on different block sizes
    try using streams?
*/
/*
SOBEL (NOT SHARED)

SMALL
0.000424
0.000220
0.000171
0.000154
0.000162
 0.000153
0.000154

MEDIUM 
0.001142
0.000533
0.000367
0.000315
0.000314
0.000329
0.000329

LARGE0.004808
0.002161
0.001374
0.001186
0.001193
0.001236
0.001200

SOBEL SHARED 

SMALL
1- 0.000448
2- 0.000229
4- 0.000170
8-  0.000155
16- 0.000158
24- 0.000153
32- 0.000165


MEDIUM 
1- 0.001256
2- 0.000550
4- 0.000358
8- 0.000317
16-  0.000323
24- 0.000333
32- 0.000328

LARGE
1- 0.005209
2- 0.001975
4- 0.001312
8-  0.002101
16- 0.001190
24-  0.001268
32- 0.001192


SOBEL SHARED DIFF

SMALL
 1- 0.000578
2-  0.000276
4- 0.000211
8- 0.000207
16- 0.000200
24- 0.000229
32- 0.000202


MEDIUM
1-  0.001604
2-0.000636
4-  0.000395
8- 0.000367
16-  0.000353
24- 0.000361
32- 0.000349

LARGE
1-  0.006609
2- 0.002341
4- 0.001429
8-  0.001219
16-0.001221
24- 0.001327
32-  0.001236
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
#define THREADS_PER_BLK 32

__device__ __constant__  float HSOBEL_H[3] = {-1.0, 0, 1.0};
__device__ __constant__ float HSOBEL_V[3] = {1.0, 2.0, 1.0};
__device__ __constant__ float HSOBEL[9] = {-1.0, -2.0, -1.0, 0, 0, 0, 1.0, 2.0, 1.0};

__global__ void single_kernel(uint8_t* old_img, uint8_t* new_img, float kernel[9], int k_width, int k_height, int img_width, int img_height){
    int i, j, jj, ii;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i>=img_width || j>= img_height) return;
    float tmp=0.f;
    int jj_last = min(j+k_height, img_height);
    int ii_last = min(i+k_width, img_width);

    for(jj = j; jj< jj_last; jj++){
        for(ii = i; ii< ii_last; ii++){            
            tmp += HSOBEL[(jj-j) * k_width + (ii-i)] * (old_img[jj * img_width + ii]);
        }
    }
    new_img[j*img_width + i] = (uint8_t)sqrt(tmp*tmp);

}

//shared requires constant
//assumes k_width, k_height = 3
__global__ void shared_kernel(uint8_t* old_img, uint8_t* new_img, int img_width, int img_height){
    int i, j, jj, ii, support_id, img_id, n_img_id;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ float support[(THREADS_PER_BLK+2)*(THREADS_PER_BLK+2)]; 
    if(i<img_width && j< img_height) {
        support_id = (THREADS_PER_BLK+2)*threadIdx.y + threadIdx.x;
        img_id = img_width * j + i;
        support[support_id] =  old_img[img_id];

        if(threadIdx.x < 2){
            n_img_id = j*img_width + (THREADS_PER_BLK + i);
            if(n_img_id<img_width * img_height)
            support[(THREADS_PER_BLK+2)*threadIdx.y + (THREADS_PER_BLK + threadIdx.x)] = old_img[n_img_id];        
        }

        if(threadIdx.y < 2){
            n_img_id = (THREADS_PER_BLK + j)*img_width + i;
            if(n_img_id<img_width * img_height)
            support[(THREADS_PER_BLK+2)*(THREADS_PER_BLK + threadIdx.y) + (threadIdx.x)] = old_img[n_img_id];
        }
    
        if(threadIdx.x < 2 && threadIdx.y<2){
            n_img_id = (THREADS_PER_BLK + j)*img_width + (THREADS_PER_BLK + i);
            if(n_img_id<img_width * img_height)
            support[(THREADS_PER_BLK+2)*(THREADS_PER_BLK + threadIdx.y) + (THREADS_PER_BLK + threadIdx.x)] = old_img[n_img_id];
        }
    }   
    
    __syncthreads();
    
    if(i<img_width && j< img_height) {
        float tmp=0.f;
        int jj_last = min(j+3, img_height) - j;
        int ii_last = min(i+3, img_width) - i;
        for(jj = 0; jj< jj_last; jj++){
            for(ii = 0; ii< ii_last; ii++){            
                tmp += HSOBEL[3*jj + ii] * support[(threadIdx.y+jj)*(THREADS_PER_BLK+2) + (threadIdx.x+ii)];
            }   
        }
        new_img[img_id] = (uint8_t)sqrt(tmp*tmp);
    }
}

//shared requires constant
//assumes k_width, k_height = 3
__global__ void shared_sep_kernel(uint8_t* old_img, uint8_t* new_img, int img_width, int img_height){
    int i, j, jj, ii, support_id, img_id, n_img_id;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    //
    __shared__ float support[(THREADS_PER_BLK+2)*(THREADS_PER_BLK+2)]; 
    __shared__ float tmp_buf[(THREADS_PER_BLK+2)*(THREADS_PER_BLK+2)]; 
    if(i==0 && j==0){
        printf("ENTERED");
    }
    

    if(i<img_width && j< img_height) {
        support_id = (THREADS_PER_BLK+2)*threadIdx.y + threadIdx.x;
        img_id = img_width * j + i;
        support[support_id] =  old_img[img_id];

        if(threadIdx.x < 2){
            n_img_id = j*img_width + (THREADS_PER_BLK + i);
            if(n_img_id<img_width * img_height)
            support[(THREADS_PER_BLK+2)*threadIdx.y + (THREADS_PER_BLK + threadIdx.x)] = old_img[n_img_id];        
        }

        if(threadIdx.y < 2){
            n_img_id = (THREADS_PER_BLK + j)*img_width + i;
            if(n_img_id<img_width * img_height)
            support[(THREADS_PER_BLK+2)*(THREADS_PER_BLK + threadIdx.y) + (threadIdx.x)] = old_img[n_img_id];
        }
    
        if(threadIdx.x < 2 && threadIdx.y<2){
            n_img_id = (THREADS_PER_BLK + j)*img_width + (THREADS_PER_BLK + i);
            if(n_img_id<img_width * img_height)
            support[(THREADS_PER_BLK+2)*(THREADS_PER_BLK + threadIdx.y) + (THREADS_PER_BLK + threadIdx.x)] = old_img[n_img_id];
        }
    }   

    
    __syncthreads();

    
    
    if(i<img_width && j< img_height) {
        float tmp=0.f;
        int ii_last = min(i+3, img_width) - i;
        for(ii = 0; ii< ii_last; ii++){            
            tmp += HSOBEL_V[ii] * support[(threadIdx.y)*(THREADS_PER_BLK+2) + (threadIdx.x+ii)];
        }   
        tmp_buf[(threadIdx.y)*(THREADS_PER_BLK+2) + threadIdx.x] = tmp;
    }
    
    if(threadIdx.y < 2){
        float tmp=0.f;
        int ii_last = min(i+3, img_width) - i;
        for(ii = 0; ii< ii_last; ii++){         
            tmp += HSOBEL_V[ii] * support[(threadIdx.y+THREADS_PER_BLK)*(THREADS_PER_BLK+2) + (threadIdx.x+ii)];
        } 
        tmp_buf[(threadIdx.y+THREADS_PER_BLK )*(THREADS_PER_BLK+2) + (threadIdx.x)] = tmp;   
    }
    if(threadIdx.x < 2){
        float tmp=0.f;
        int ii_last = min(i+3, img_width) - i;
        for(ii = 0; ii< ii_last; ii++){         
            tmp += HSOBEL_V[ii] * support[(threadIdx.y)*(THREADS_PER_BLK+2) + (THREADS_PER_BLK+threadIdx.x+ii)];
        } 
        tmp_buf[(threadIdx.y )*(THREADS_PER_BLK+2) + (THREADS_PER_BLK+threadIdx.x)] = tmp;  
    }
    if(threadIdx.x < 2 && threadIdx.y < 2){
        float tmp=0.f;
        int ii_last = min(i+3, img_width) - i;
        for(ii = 0; ii< ii_last; ii++){         
            tmp += HSOBEL_V[ii] * support[(THREADS_PER_BLK+threadIdx.y)*(THREADS_PER_BLK+2) + (THREADS_PER_BLK+threadIdx.x+ii)];
        } 
        tmp_buf[(THREADS_PER_BLK+threadIdx.y )*(THREADS_PER_BLK+2) + (THREADS_PER_BLK+threadIdx.x)] = (uint8_t)sqrt(tmp*tmp);  
    }

    __syncthreads();
    

    if(i<img_width && j< img_height) {
        float tmp=0.f;
        int jj_last = min(j+3, img_height) - j;
        for(jj = 0; jj< jj_last; jj++){            
            tmp += HSOBEL_H[jj] * tmp_buf[(threadIdx.y+jj)*(THREADS_PER_BLK+2) + (threadIdx.x)];
        }   
        //tmp = tmp_buf[(threadIdx.y)*(THREADS_PER_BLK+2) + (threadIdx.x)];
        new_img[j*img_width + i] = (uint8_t)sqrt(tmp*tmp);
    }
    return;


}

void edge_detect_single(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img, uint8_t* new_img_device, uint8_t* old_img_device) {
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(block_side, block_side);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    single_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, HSOBEL, 3, 3, width, height);
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("edge_single.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  

}

void edge_detect_shared(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img, uint8_t* new_img_device, uint8_t* old_img_device) {
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(THREADS_PER_BLK, THREADS_PER_BLK);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    shared_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, width, height);    
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    //stbi_write_png("edge_shared.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  

}

void edge_detect_shared_sep(uint8_t* &old_img, int width, int height, int block_side, uint8_t* new_img, uint8_t* new_img_device, uint8_t* old_img_device) {
    cudaMemcpy(old_img_device, old_img, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(THREADS_PER_BLK, THREADS_PER_BLK);
    dim3 gridDim((width+threadsPerBlock.x-1)/threadsPerBlock.x, (height+threadsPerBlock.y-1)/threadsPerBlock.y);
    shared_sep_kernel<<<gridDim, threadsPerBlock>>>(old_img_device, new_img_device, width, height);    
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    //stbi_write_png("edge_shared_sep.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  

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
    for(int i=0; i<1; i++){
        if(type=='n')
            edge_detect_single(old_img, width, height, block_side, new_img, new_img_device, old_img_device);
        else if(type=='s')
            edge_detect_shared(old_img, width, height, block_side, new_img, new_img_device, old_img_device);
        else if(type=='t')
            edge_detect_shared_sep(old_img, width, height, block_side, new_img, new_img_device, old_img_device);

    }
    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time Without Startup: %f\n", duration_exc);
    return 1;
    
}




