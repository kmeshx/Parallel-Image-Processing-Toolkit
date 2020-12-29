#include <algorithm>
#include <cstdlib>
#include <limits>
#include "cycletimer.h"
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define BLOCK_SIDE 1024
#define CHANNEL_NUM 3
#include <unistd.h>
#include <stdio.h>
#include <cstdlib>

typedef unsigned int uint;

struct Point;
struct Point {
    volatile unsigned int x, y;     // coordinates
    volatile unsigned int r, g, b;
    volatile int count;     // no default cluster

    __device__ __host__ Point() : 
        x(0), 
        y(0),
        r(0),
        g(0),
        b(0),
        count(1) {}
    
    __device__ __host__ Point(int count) : 
        x(0), 
        y(0),
        r(0),
        g(0),
        b(0),
        count(count) {}

    __device__ __host__ Point(unsigned int x, unsigned int y, uint8_t r, uint8_t g, uint8_t b) : 
        x(x), 
        y(y),
        r(r),
        g(g),
        b(b),
        count(1) {}

    __device__ __host__ Point(unsigned int x, unsigned int y, uint8_t r, uint8_t g, uint8_t b, int c) : 
    x(x), 
    y(y),
    r(r),
    g(g),
    b(b),
    count(c) {}

    __device__ double color_distance(Point p){
        double v = ((double)((p.r - r) * (p.r - r))) + 
        ((double)((p.g - g) * (p.g - g))) +
        ((double)((p.b - b) * (p.b - b)));
        return v;
    }
    __inline__ __device__ __host__ void sum(Point p){
        x+=p.x;
        y+=p.y;
        r+=p.r;
        b+=p.b;
        g+=p.g;
        count+=p.count;
    }

    __inline__ __device__ __host__ void diff(Point p){
        x-=p.x;
        y-=p.y;
        r-=p.r;
        b-=p.b;
        g-=p.g;
        count-=p.count;
    }
    __device__ __host__ void print_point(){
        printf("X: %d, Y: %d\nR: %d, G: %d, B: %d\nCount: %d\n",
        x,y,r,g,b,count);
    }
};


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
        //printf("Cluster %d\n", cluster);
        Point p = Point(0);
        Point m;
        //printf("CB b: %d\n", b);
        for(int block_id = 0; block_id < b; block_id++){
            //printf("CB Block ID: %d\n", block_id);
            //printf("CB Block Val: %d\n", means_block_device[block_id]);
            m = means_block_device[block_id];
            //m.print_point();
            p.sum(m);
        }
        int c = p.count;
        p.r = p.r/c;
        p.g = p.g/c;
        p.b = p.b/c;
        p.x = p.x/c;
        p.y = p.y/c;
        means_cluster_device[cluster] = p;
    }
}

__global__ void mask_cluster(Point* data, size_t* assignments, int total_num_points, 
    int k, int cluster, Point *data_scratch){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < total_num_points){
        if(assignments[id]==cluster){data_scratch[id] = data[id];}
        else{data_scratch[id] = Point(0);}
        //if(data_scratch[id].r > 1)printf("ID: %d, p.r: %d\n", id, data_scratch[id].r);
    }
}

__global__ void sum_for_blocks(Point *data_scratch, Point *means_block, 
    int total_num_points, int NUM_CHUNKS){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //this id between 0 and b = NUM_CHUNKS
    if(id < NUM_CHUNKS){
        int points_per_chunk = (total_num_points/NUM_CHUNKS) + 1;
        int start_point_index = id*points_per_chunk;
        int end_point_index = (id+1)*points_per_chunk;
        Point p = Point(0);
        Point d;
        for(int i = start_point_index; (i < end_point_index) && (i < total_num_points); i++){
            d = data_scratch[i];
            if(d.count == 1) p.sum(d);
        }
        means_block[id] = p;
        //p.print_point();
        //printf("SFB Id: %d\n", id);

    }
}

void update_mean(dim3 gridDim, dim3 threadsPerBlock, dim3 gridDimC, dim3 threadsPerBlockC, 
    Point *means_block, Point* data, size_t* assignments, 
    int total_num_points, int k, int cluster, Point *data_scratch, int b){
    //mask & keep points only within cluster instead of all k clusters
    mask_cluster<<<gridDim, threadsPerBlock>>>(data, assignments, total_num_points, 
    k, cluster, data_scratch);
    //printf("wtf");
    sum_for_blocks<<<gridDim, threadsPerBlock>>>(data_scratch, means_block, total_num_points, b);
    //printf("xxx");
}

void k_means_main(dim3 gridDim, dim3 threadsPerBlock, dim3 gridDimC, dim3 threadsPerBlockC,
    Point* points, Point *means_cluster_device, Point *means_block_device, 
    size_t* assignments, int number_of_iterations, int k, int b,
    int height, int width, uint8_t* rgb_image, uint8_t* new_img, Point *data_scratch){ 

    int total_num_points = width*height;
    fill_points<<<gridDim, threadsPerBlock>>>(points, height, width, rgb_image);

    /*Step 2 from comments*/
    for(int i = 0; i< number_of_iterations; i++){
        set_assignments<<<gridDim, threadsPerBlock>>>(points, assignments, 
        means_cluster_device, k, total_num_points);
        for(int cluster = 0; cluster < k; cluster++){
            //printf("UM1\n");
            update_mean(gridDim, threadsPerBlock, gridDimC, threadsPerBlockC, 
            means_block_device, points, assignments, 
            total_num_points, k, cluster, data_scratch, b);
            //printf("UM2\n");
            cluster_from_blocks<<<gridDim, threadsPerBlock>>>(means_cluster_device, 
            means_block_device, cluster, b);
            //printf("Cluster: %d, Iter: %d\n", cluster, i);
        }
    }

    //at the very end, after accumulating blockwise sums and all, set new image 
    set_new_img<<<gridDim, threadsPerBlock>>>(means_cluster_device, assignments, total_num_points, new_img);
}


/*Init means for all blocks. & for all clusters.
Since we will update means for clusters in sequence
we only need num_blocks number of slots. */
void set_init_means(uint8_t *rgb_image, Point *means_cluster_host, Point *means_block_host, 
    int k, int b, int width, int height){
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
        means_cluster_host[cluster_index] = Point(x,y,r,g,bl,0);
        //means_cluster_host[cluster_index].print_point();
    }
    for(int block_index = 0; block_index < b; block_index++){
        means_block_host[block_index] = Point();
    }
}

void k_means(uint8_t* rgb_image, int width, int height, 
    size_t k, size_t number_of_iterations, int b) {

    int total_points = width*height;
    int total_cpoints = total_points*CHANNEL_NUM;

    //parallelize over pixels kernel dims
    dim3 threadsPerBlock(BLOCK_SIDE, 1, 1);
    const int NUM_BLOCKS_X = (total_points+threadsPerBlock.x-1)/threadsPerBlock.x;
    const int NUM_BLOCKS_Y = 1;
    dim3 gridDim(NUM_BLOCKS_X , NUM_BLOCKS_Y, 1);
    //create chunk grid dims
    //total_points/NUM_CHUNKS * NUM_CHUNKS + total_points%NUM_CHUNKS = total_points
    const int NUM_THREADS_XC = (total_points/b) + 1;
    dim3 threadsPerBlockC(NUM_THREADS_XC, 1, 1);
    dim3 gridDimC(b , 1, 1);
    //printf("TPB: %d, NC: %d\n", NUM_THREADS_XC, gridDimC.x);

    //initialize means in before launching kernels since k will typically
    //be much smaller compared to image sizes
    Point *means_cluster_host = (Point*) malloc(sizeof(Point) * k);
    Point *means_block_host = (Point*) malloc(sizeof(Point) * b);
    set_init_means(rgb_image, means_cluster_host, means_block_host, k, b, width, height);

    //GPU mallocs
    Point* means_cluster_device;
    Point* means_block_device;
    Point* points_device;
    Point* data_scratch;
    size_t* assignments_device;
    uint8_t* new_img_device;
    uint8_t* rgb_img_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * total_cpoints);

    cudaMalloc(&means_cluster_device, sizeof(Point) * k);
    cudaMalloc(&means_block_device, sizeof(Point) * b);
    cudaMalloc(&points_device, sizeof(Point) * total_points);
    cudaMalloc(&data_scratch, sizeof(Point) * total_points);

    cudaMalloc(&assignments_device, sizeof(size_t) * total_points);
    cudaMalloc(&new_img_device, sizeof(uint8_t) * total_cpoints);
    cudaMalloc(&rgb_img_device, sizeof(uint8_t) * total_cpoints);

    //copy from host to GPU
    cudaMemcpy(rgb_img_device, rgb_image, sizeof(uint8_t) * total_cpoints, cudaMemcpyHostToDevice);
    cudaMemcpy(means_cluster_device, means_cluster_host, sizeof(Point) * k, cudaMemcpyHostToDevice);
    cudaMemcpy(means_block_device, means_block_host, sizeof(Point) * b, cudaMemcpyHostToDevice);

    //time main computational functions
    double start_time_exc = currentSeconds();

    k_means_main(gridDim, threadsPerBlock, gridDimC, threadsPerBlockC, 
    points_device, means_cluster_device, means_block_device, 
    assignments_device, number_of_iterations, k, b,  
    height, width, rgb_img_device, new_img_device, data_scratch);

    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    printf("%f, ", duration_exc);

    //copy image back into host from device
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * total_cpoints, cudaMemcpyDeviceToHost);
    stbi_write_png("out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
}

int main_single(int argc, char **argv){
    const char *img_file = argv[1];//"images/cs_test1.jpg";
    int NUM_CLUSTERS = atoi(argv[2]);
    int NUM_ITERS = atoi(argv[3]);
    int NUM_CHUNKS = atoi(argv[4]);
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(img_file, &width, &height, 
    &bpp, CHANNEL_NUM);  
    k_means(rgb_image, width, height, NUM_CLUSTERS, NUM_ITERS, NUM_CHUNKS);    
    return 1;
}

int main(int argc, char **argv){
    int NUM_IMGS_all = 1;
    int NUM_CLUSTERS_all = 3;//atoi(argv[2]);
    int NUM_ITERS_all = 2048; //atoi(argv[3]);
    int NUM_CHUNKS_all = 8; //atoi(argv[4]);

    static const char* imgs[] = {"images/medium.jpg",
    "images/small.jpg", "images/large.jpg"};
    static const int chunks[] = {32, 64, 128, 156, 192, 256, 384, 512};
    for(int i = 0; i < NUM_IMGS_all; i++){
        printf("Image %d\n", i);
        for(int j = 0; j < NUM_CHUNKS_all; j++){
            int width, height, bpp;
            uint8_t* rgb_image = stbi_load(imgs[i], &width, &height, 
            &bpp, CHANNEL_NUM);  
            //printf("read");
            k_means(rgb_image, width, height, NUM_CLUSTERS_all, 
                NUM_ITERS_all, chunks[j]);    
            
        }
        printf("\n");
    }
    return 1;
}
