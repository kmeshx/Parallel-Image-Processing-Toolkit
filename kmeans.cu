//#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
//#include <stdio.h>
#include "exclusiveScan.cu_inl"
#include "cycletimer.h"
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 3
#define BLOCK_SIDE 1024
#define BLOCK_SIZE 1024
//#define SCAN_BLOCK_DIM BLOCK_SIZE 
#define IMG_FILE "cs_test1.jpg"
#define NUM_CLUSTERS 3
#define NUM_ITERS 2048

/*b = # of blocks = total_sum_points/BLOCK_SIZE*/
//initialize means before doing kernel stuff
//Main k-means kernel per block
/*
1. K: Fill points block wise in global memory points array
2. loop over iterations
        K: set assignments for each pixel in assignments array
        for each cluster:
            K: ---
            no<> fill in output of mask into SHARED mem
            syncthreads
            no<> run shared mem scan
            syncthreads (after summing for current cluster 
            in block)
            no<> only in one thread:
            *update current block means by block array
            *array with this sum/count info
            ---
            In seq:
            *sum over all different blocks
            *then set final means (in means' cluster index)
3. fill in image based on final assignment and means array
*/
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void fill_points(Point* points, int height, int width, uint8_t* rgb_image){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    int total_num_points = width * height;
    unsigned int x, y;
    uint8_t r, g, b;
    int point_channel = point*CHANNEL_NUM;
    if(point<total_num_points){
        int factor = (width*CHANNEL_NUM);
        y = (unsigned int)(point_channel/factor);
        x = (unsigned int)(point_channel%factor);
        r = rgb_image[point_channel]; 
        g = rgb_image[point_channel+1];
        b = rgb_image[point_channel+2];
        points[point] = (Point(x, y, r, g, b));
    }   
}

__global__ void set_new_img(Point* means, size_t* assignments, int total_num_points, uint8_t* new_img){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    Point p;
    int c;
    if(point<total_num_points)
    {
        c = assignments[point];
        p = means[c]; //use means by cluster
        new_img[CHANNEL_NUM*point] = p.r;
        new_img[CHANNEL_NUM*point+1] = p.g;
        new_img[CHANNEL_NUM*point+2] = p.b;
    }
}

__global__ void set_assignments(Point* data, size_t*  assignments, Point* means, int k, int total_num_points){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    if(point < total_num_points){
        int assignment = 0;
        Point p,m;
        double best_distance = CHANNEL_NUM*256*256;
        size_t best_cluster = 0;
        for (size_t cluster = 0; cluster < k; ++cluster) {
            p = data[point];
            m = means[cluster];
            double distance = p.color_distance(m);
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster;
                assignment = best_cluster;
            }
        }
        assignments[point] = assignment; 
    }
}

__global__ void cluster_from_blocks(Point *means_cluster_device, Point *means_block_device, 
    int cluster, int b){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    if(point == 0){
    Point p = Point(0);
    //printf("CB b: %d\n", b);
    for(int block_id = 0; block_id < b; block_id++){
        //printf("CB Block ID: %d\n", block_id);
        //printf("CB Block Val: %d\n", means_block_device[block_id]);
        p.sum(means_block_device[block_id]);
    }
    int c = p.count;
    p.r = p.r/c;
    p.g = p.g/c;
    p.b = p.b/c;
    p.x = p.x/c;
    p.y = p.y/c;
    p.count = 0;
    means_cluster_device[cluster] = p;
}
}

__device__ void mask_cluster(Point* data, size_t* assignments, 
    int total_num_points, int k, int cluster, int id, 
    uint *block_mask_cluster, int btid){
    if(id < total_num_points){
        int l = 0;
        uint *addr = block_mask_cluster+NUM_FIELDS*btid;
        if(assignments[id]==cluster) {
            point_to_cells((uint*)((void*)(data+id)), addr);
        }
        else{
            init_point_cells(addr, 0);
        }
        //printf("BMC id: %d, val: %d\n", id, block_mask_cluster[btid]);
    }
}

__inline__ __device__ void set_block_means(uint* prefix_sums, Point *means_block,
    int first_block_id, int id, int block_id){
    if(id == first_block_id){
        //Point p = prefix_sums[SCAN_BLOCK_DIM-1];
        means_block[block_id].x = 0;//p.x;
        means_block[block_id].y = 0;//p.y;
        means_block[block_id].r = 0;//p.r;
        means_block[block_id].g = 0;//p.g;
        means_block[block_id].b = 0;//p.b;
        means_block[block_id].count = 0;//p.count;
    }
}

__global__ void update_mean(Point* means_cluster, Point *means_block, 
    Point* data, size_t* assignments, 
    int total_num_points, int k, int cluster){
    //printf("UM\n");
    __shared__ uint block_mask_cluster[SCAN_BLOCK_DIM*NUM_FIELDS];
    __shared__ uint sum_output[SCAN_BLOCK_DIM*NUM_FIELDS];
    __shared__ uint scratch[SCAN_BLOCK_DIM*NUM_FIELDS*2];
    int first_block_id = blockIdx.x * blockDim.x;
    int id = first_block_id + threadIdx.x;
    if(id < total_num_points){
    mask_cluster(data, assignments, total_num_points, k, cluster, 
        id, block_mask_cluster, threadIdx.x);
    __syncthreads();
    sharedMemExclusiveScan(threadIdx.x, 
        block_mask_cluster, sum_output, 
    scratch, SCAN_BLOCK_DIM);
    __syncthreads();
    /*set_block_means(sum_output, means_block, first_block_id, id, blockIdx.x);
    */
    }
}

void k_means_main(dim3 gridDim, dim3 threadsPerBlock, Point* points, 
    Point *means_cluster_device, Point *means_block_device, 
    size_t* assignments, int number_of_iterations, int k, int b,
    int height, int width, 
    uint8_t* rgb_image, uint8_t* new_img, 
    Point *data_scratch){ 

    int total_num_points = width*height;
    fill_points<<<gridDim, threadsPerBlock>>>(points, height, width, rgb_image);

    /*Step 2 from comments*/
    for(int i = 0; i< number_of_iterations; i++){
        set_assignments<<<gridDim, threadsPerBlock>>>(points, assignments, means_cluster_device, 
        k, total_num_points);
        for(int cluster = 0; cluster < k; cluster++){
            //printf("UM1\n");
            update_mean<<<gridDim, threadsPerBlock>>>(means_cluster_device, means_block_device, points, 
            assignments, total_num_points, k, cluster);
            //printf("UM2\n");
            cluster_from_blocks<<<gridDim, threadsPerBlock>>>(means_cluster_device, means_block_device, cluster, b);
            //printf("Cluster: %d, Iter: %d\n", cluster, i);
        }
    }

    //at the very end, after accumulating blockwise sums and all, set new image 
    set_new_img<<<gridDim, threadsPerBlock>>>(means_cluster_device, assignments, total_num_points, new_img);
}


/*Init means for all blocks. & for all clusters.
Since we will update means for clusters in sequence
we only need num_blocks number of slots. */
void set_init_means(uint8_t *rgb_image, Point *means_cluster_host, 
    Point *means_block_host, 
    int k, int b,
    int width, int height){
    unsigned int x, y;
    uint8_t r, g, bl;
    int factor, init_ind;
    for(int cluster_index = 0; cluster_index < k; cluster_index++){
        factor = (width*CHANNEL_NUM);
        init_ind = CHANNEL_NUM*cluster_index*((height*width)/k);
        y = (unsigned int)(init_ind/factor);
        x = (unsigned int)(init_ind%factor);
        r = rgb_image[init_ind];
        g = rgb_image[init_ind+1];
        bl = rgb_image[init_ind+2];
        means_cluster_host[cluster_index] = Point(x,y,r,g,bl);
    }
    for(int block_index = 0; block_index < b; block_index++){
        means_block_host[block_index] = Point();
    }
}

void k_means(uint8_t* rgb_image, int width, int height, 
    size_t k, size_t number_of_iterations) {

    dim3 threadsPerBlock(BLOCK_SIDE, 1, 1);
    const int NUM_BLOCKS_X = (width*height+threadsPerBlock.x-1)/threadsPerBlock.x;
    const int NUM_BLOCKS_Y = 1;
    dim3 gridDim(NUM_BLOCKS_X , NUM_BLOCKS_Y, 1);
    int b = NUM_BLOCKS_X;
    int total_points = width*height*CHANNEL_NUM;

    //initialize means in before launching kernels since k will typically
    //be much smaller compared to image sizes
    Point *means_cluster_host = (Point*) malloc(sizeof(Point) * k);
    Point *means_block_host = (Point*) malloc(sizeof(Point) * b);
    set_init_means(rgb_image, means_cluster_host, means_block_host, k, b, width, height);

    //GPU mallocs
    Point* means_cluster_device;
    Point* means_block_device;
    Point* points_device;
    Point *data_scratch;
    int total_num_points = height*width;
    int scratch_len = nextPow2(total_num_points);
    size_t* assignments_device;
    uint8_t* new_img_device;
    uint8_t* rgb_img_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * total_points);

    cudaMalloc(&means_cluster_device, sizeof(Point) * k);
    cudaMalloc(&means_block_device, sizeof(Point) * b);
    cudaMalloc(&points_device, sizeof(Point) * height * width);
    cudaMalloc(&data_scratch, sizeof(Point) * scratch_len);

    cudaMalloc(&assignments_device, sizeof(size_t) * height * width);
    cudaMalloc(&new_img_device, sizeof(uint8_t) * total_points);
    cudaMalloc(&rgb_img_device, sizeof(uint8_t) * total_points);

    //copy from host to GPU
    cudaMemcpy(rgb_img_device, rgb_image, sizeof(uint8_t) * total_points, cudaMemcpyHostToDevice);
    cudaMemcpy(means_cluster_device, means_cluster_host, sizeof(Point) * k, cudaMemcpyHostToDevice);
    cudaMemcpy(means_block_device, means_block_host, sizeof(Point) * b, cudaMemcpyHostToDevice);

    //time main computational functions
    double start_time_exc = currentSeconds();

    k_means_main(gridDim, threadsPerBlock, points_device, 
    means_cluster_device, means_block_device, 
    assignments_device, number_of_iterations, k, b,  
    height, width, 
    rgb_img_device, new_img_device, 
    data_scratch);

    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    printf("Time: %f\n", duration_exc);

    //copy image back into host from device
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
}

int main(int argc, char **argv){
    const char *img_file = "cs_test1.jpg";
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    k_means(rgb_image, width, height, NUM_CLUSTERS, NUM_ITERS);    
    return 1;
}