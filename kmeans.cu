#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 3
#define BLOCK_SIDE 5

struct Point;
using DataFrame = std::vector<Point>;

struct Point {
    double x, y;     // coordinates
    int cluster;     // no default cluster
    double min_dist;  // default infinite dist to nearest cluster
    double r, g, b;

    Point() : 
        x(0.0), 
        y(0.0),
        r(0.0),
        g(0.0),
        b(0.0),
        cluster(-1),
        min_dist(__DBL_MAX__) {}
        
    Point(double x, double y, double r, double g, double b) : 
        x(x), 
        y(y),
        r(r),
        g(g),
        b(b),
        cluster(-1),
        min_dist(__DBL_MAX__) {}

    double euclid_distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }

    double color_distance(Point p){
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

Points* get_df(uint8_t *rgb_image, int width, int height){
    
    int l = width*height*CHANNEL_NUM;
    double x, y;
    double r, g, b;
    int factor = (width*CHANNEL_NUM);
    for(int i = 0; i < l; i+=CHANNEL_NUM){
        y = (float) (i/factor);
        x = (float) (i%factor);
        r = rgb_image[i]; 
        g = rgb_image[i+1];
        b = rgb_image[i+2];
        points[i] = (Point(x, y, r, g, b));
    }
}


__device__ void update_mean(Point* means, int one_d_id, int total_num_points){
    
    counts = 0;
    for (size_t point = 0; point < total_num_points; ++point) {
        if(assignment[point] == one_d_id){
            p.x += data[point].x;
            p.y += data[point].y;
            p.r += data[point].r;
            p.g += data[point].g;
            p.b += data[point].b;
            counts += 1;
        }
    }
    means[one_d_id].x = p.x / count;
    means[one_d_id].y = p.y / count;
    means[one_d_id].r = p.r / count;
    means[one_d_id].g = p.g / count;
    means[one_d_id].b = p.b / count;
}

//assignments, means, height, width, index_x, index_y
__device__ void set_assignments(int* assignments, Point* means, int point){
    int assignment = -1
    double best_distance = std::numeric_limits<double>::max();
    size_t best_cluster = 0;
    for (size_t cluster = 0; cluster < k; ++cluster) {
        double distance = data[point].color_distance(means[cluster]);
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
            assignment = best_cluster;
        }
    }
    assignments[index_x * width + index_y] = assignment; 
}

__device__ void set_init_assignments(int* assignments, Point* means, int point){


}
/*
TODO check data.size()
*/
__global__ void k_means_kernel(Point* &points, Point* means, int* assignments, int number_of_iterations, int k, int height, int width, uint8_t* new_img, int* init_menas, int* init_mean_nums){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    int total_num_points = width * height;
    if(point < k){
        means[point] = init_mean_nums[point];
    }
    __sync_threads();
    
    for(int i = 0; i< iters; i++){
        if(point < total_num_points){
            set_assignments(assignments, means, point);
        }
        __sync_threads();
        // now parallelize over clusters
        // TODO USE SCAN 
        int id = point;
        if(id < k){
            update_mean(means, one_d_id, total_num_points);
        }
        __sync_threads();
    }

    Point p;
    int c;
    //TODO check len
    if(point<total_num_points)
    {
        c = assignments[i];
        p = points[c];
        rgb_image[CHANNEL_NUM*idx] = p.r;
        rgb_image[CHANNEL_NUM*idx+1] = p.g;
        rgb_image[CHANNEL_NUM*idx+2] = p.b;

    }
 
}

DataFrame print_df(DataFrame& points, int width, int height){
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



void k_means(Points* &df, int width, int height,
                  size_t k,
                  size_t number_of_iterations) {
    dim3 threadsPerBlock(BLOCK_SIDE, BLOCK_SIDE, 1);
    const int NUM_BLOCKS_X = (width+threadsPerBlock.x-1)/threadsPerBlock.x;
    const int NUM_BLOCKS_Y = (height+threadsPerBlock.y-1)/threadsPerBlock.y;
    dim3 gridDim(NUM_BLOCKS_X , NUM_BLOCKS_Y, 1);
    //TODO CUDA RANDOM MEANS and assignments 

    static std::random_device seed;
    static std::mt19937 random_number_generator(seed());


    /*
    std::uniform_int_distribution<size_t> indices(0, data.size() - 1);
  
    // Pick centroids as random points from the dataset.
    
    int index;
    for (int i=0; i<k; i++) {
      index = (int)(indices(random_number_generator)/CHANNEL_NUM);
      means.at(i) = data[index];
    }
    */

    Point* means_device, points_device;
    size_t* assignments_device;
    uint8_t* new_img_device, new_img;
    int* init_mean_nums;
    // = malloc(sizeof(Point)  );
    cudaMalloc(means_device, sizeof(Point) * k);
    cudaMalloc(points_device, sizeof(Point) * height * width );
    cudaMalloc(init_mean_nums, sizeof(int) * height * width );

    cudaMalloc(assignments_device, sizeof(size_t) * height * width);
    cudaMalloc(new_img_device, sizeof(uint8_t) * height * width );
    cudaMemcpy(new_img_device, new_img, sizeof(uint8_t) * height * width, cudaMemcpyDeviceToHost);
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    /* Generate n floats on device */
    curandGenerator_t gen;
    CURAND_CALL(curandGenerateUniform(gen, init_mean_nums, height * width));
    k_means_kernel<<<gridDim, threadsPerBlock>>>(points_device, means_device, assignments_device, 
                                                number_of_iterations, k,  height, width, new_img_device, init_mean_nums);
    cudaMemcpy(new_img, points, sizeof(Point) * height * width * CHANNEL_NUM, cudaMemcpyHostToDevice);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
    printf("Finished k-means\n");
}


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = "cs_test1.jpg";
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    Points* df = get_df(rgb_image, width, height);
    k_means(df, width, height, 3, 2);
    return 1;
    
}
