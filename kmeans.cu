#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
//#include <random>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 3
#define BLOCK_SIDE 5

struct Point;
//using DataFrame = std::vector<Point>;

struct Point {
    double x, y;     // coordinates
    int cluster;     // no default cluster
    double min_dist;  // default infinite dist to nearest cluster
    double r, g, b;

    __device__ Point() : 
        x(0.0), 
        y(0.0),
        r(0.0),
        g(0.0),
        b(0.0),
        cluster(-1),
        min_dist(__DBL_MAX__) {}
        
    __device__ Point(double x, double y, double r, double g, double b) : 
        x(x), 
        y(y),
        r(r),
        g(g),
        b(b),
        cluster(-1),
        min_dist(__DBL_MAX__) {}

    __device__ double euclid_distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }

    __device__ double color_distance(Point p){
        return (p.r - r) * (p.r - r) + (p.b - b) * (p.b - b) + (p.g - g) * (p.g - g);
    }
};


void raw_print(uint8_t *rgb_image, int width, int height){
    int l = width*height*CHANNEL_NUM;
    float x, y;
    int factor = (width*CHANNEL_NUM);
    for(int i = 0; i < l; i+=CHANNEL_NUM){
        y = (float) (i/factor);
        x = (float) (i%factor);
        printf("X: %f, Y: %f, R: %d, G: %d, B: %d\n", 
        x, y, rgb_image[i], rgb_image[i+1], rgb_image[i+2]);
    }
}

__device__ void update_mean(Point* means, Point* data, size_t* assignments, int one_d_id, int total_num_points){
    
    int counts = 0;
    Point p;
    for (size_t point = 0; point < total_num_points; ++point) {
        if(assignments[point] == one_d_id){
            p.x += data[point].x;
            p.y += data[point].y;
            p.r += data[point].r;
            p.g += data[point].g;
            p.b += data[point].b;
            counts += 1;
        }
    }
    means[one_d_id].x = p.x / counts;
    means[one_d_id].y = p.y / counts;
    means[one_d_id].r = p.r / counts;
    means[one_d_id].g = p.g / counts;
    means[one_d_id].b = p.b / counts;
}

//assignments, means, height, width, index_x, index_y
__device__ void set_assignments(Point* data, size_t*  assignments, Point* means, int point, int k, int width, int height ){
    int assignment = -1;
    double best_distance = (width * height + 1)*(width * height + 1);
    //std::numeric_limits<double>::max();
    size_t best_cluster = 0;
    for (size_t cluster = 0; cluster < k; ++cluster) {
        double distance = data[point].color_distance(means[cluster]);
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
            assignment = best_cluster;
        }
    }
    assignments[point] = assignment; 
}

/*
TODO check data.size()
*/
__global__ void k_means_kernel(Point* &points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, uint8_t* rgb_image, uint8_t* new_img, int* init_mean_nums){
    
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%d", point);
    int total_num_points = width * height;
    if(point<total_num_points){
        double x, y;
        double r, g, b;
        int factor = (width*CHANNEL_NUM);

        int point_channel = point *CHANNEL_NUM;
        y = (float) (point_channel/factor);
        x = (float) (point_channel%factor);
        r = rgb_image[point_channel]; 
        g = rgb_image[point_channel+1];
        b = rgb_image[point_channel+2];
        points[point] = (Point(x, y, r, g, b));
    }
    
    __syncthreads();
    if(point < k){
        means[point] = Point();
        //init_mean_nums[point];
    }
    __syncthreads();
    
    for(int i = 0; i< number_of_iterations; i++){
        if(point < total_num_points){
            set_assignments(points, assignments, means, point, k, width, height);
        }
        __syncthreads();
        // now parallelize over clusters
        // TODO USE SCAN 
        int id = point;
        if(id < k){
            update_mean(means, points, assignments, id, total_num_points);
        }
        __syncthreads();
    }

    Point p;
    int c;
    //TODO check len
    if(point<total_num_points)
    {
        c = assignments[point];
        p = points[c];
        new_img[CHANNEL_NUM*point] = p.r;
        new_img[CHANNEL_NUM*point+1] = p.g;
        new_img[CHANNEL_NUM*point+2] = p.b;

    }
 
}

/*
void print_df(Points* &points, int width, int height){
    int l = width*height;
    Point p;
    printf("Size: %d\n", points.size());
    for(int i = 0; i < l; i++){
        p = points[i];
        printf("X: %f, Y: %f, R: %f, G: %f, B: %f\n",
                p.x, p.y, p.r, p.g, p.b);
    }
    printf("Printed DF\n");
}
*/


void k_means(uint8_t* rgb_image, int width, int height,
                  size_t k,
                  size_t number_of_iterations) {
    dim3 threadsPerBlock(BLOCK_SIDE, 1, 1);
    const int NUM_BLOCKS_X = (width*height+threadsPerBlock.x-1)/threadsPerBlock.x;
    const int NUM_BLOCKS_Y = 1;
    //(height+threadsPerBlock.y-1)/threadsPerBlock.y;
    dim3 gridDim(NUM_BLOCKS_X , NUM_BLOCKS_Y, 1);
    
    //TODO CUDA RANDOM MEANS and assignments 
    //static std::random_device seed;
    //static std::mt19937 random_number_generator(seed());
    /*
    std::uniform_int_distribution<size_t> indices(0, data.size() - 1);
    // Pick centroids as random points from the dataset.
    int index;
    for (int i=0; i<k; i++) {
      index = (int)(indices(random_number_generator)/CHANNEL_NUM);
      means.at(i) = data[index];
    }
    */

    Point* means_device;
    Point* points_device;
    size_t* assignments_device;
    uint8_t* new_img_device;
    uint8_t* rgb_img_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);
    int* init_mean_nums;
    printf("ENTERED");
    cudaMalloc(&means_device, sizeof(Point) * k);
    cudaMalloc(&points_device, sizeof(Point) * height * width );
    cudaMalloc(&init_mean_nums, sizeof(int) * k );
    cudaMalloc(&assignments_device, sizeof(size_t) * height * width);
    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&rgb_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    
    cudaMemcpy(rgb_img_device, rgb_image, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    printf("COPIED");
    //cudaMemcpy(&new_img_device, new_img, sizeof(uint8_t) * height * width, cudaMemcpyDeviceToHost);
    /* Set seed */
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate n floats on device */
    //curandGenerator_t gen;
    //CURAND_CALL(curandGenerateUniform(gen, init_mean_nums, height * width));
    //(Point* &points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, uint8_t* new_img, int* init_mean_nums)
    k_means_kernel<<<gridDim, threadsPerBlock>>>(points_device, means_device, assignments_device, 
                                                number_of_iterations, k,  height, width, rgb_img_device, new_img_device, init_mean_nums);
    printf("DONE");
    cudaMemcpy(new_img, rgb_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);

    //cudaMemcpy(new_img, new_img_device, sizeof(Point) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
    printf("Finished k-means\n");
}


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = "cs_test1.jpg";
    int width, height, bpp;
    printf("READING");
    uint8_t* rgb_image = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    printf("READ");
    //Point* df = get_df(rgb_image, width, height);
    k_means(rgb_image, width, height, 3, 2);
    return 1;
    
}
