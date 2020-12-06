#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdio.h>
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
#define SCAN_BLOCK_DIM BLOCK_SIZE 

struct Point;
struct Point {
    unsigned int x, y;     // coordinates
    unsigned int r, g, b;
    int count;     // no default cluster

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

    __device__ double euclid_distance(Point p) {
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    }
    __device__ double color_distance(Point p){
        double v = ((double)((p.r - r) * (p.r - r))) + 
        ((double)((p.g - g) * (p.g - g))) +
        ((double)((p.b - b) * (p.b - b)));
        return v;
    }
    __inline__ __device__ void sum(Point p){
        x+=p.x;
        y+=p.y;
        r+=p.r;
        b+=p.b;
        g+=p.g;
        count+=p.count;
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
/*
Point* get_df(uint8_t *rgb_image, int width, int height){
    
    int l = width*height*CHANNEL_NUM;
    Point* points = (Point*)malloc(sizeof(Point) * width * height);
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
    return points;
}
*/

//--------------------------SCAN----------------------------//
extern float toBW(int bytes, float sec);
/* Helper function to round up to a power of 2. 
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
/*
upsweep phase of exclusive scan algorithm in parallel. 
*/
__global__ void upsweep_kernel(Point *device_data, int length, int twod, int twod1){
    //compute overall index in 1D array device_data
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int two_step_index = index*twod1;
    if(two_step_index+twod1-1 < length){
        device_data[two_step_index+twod1-1].sum(device_data[two_step_index+twod-1]);
    }
} 

__global__ void downsweep_kernel(Point *device_data, int length, int twod, 
    int twod1, int og_length){
    //compute overall index in 1D array device_data
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int two_step_index = index*twod1;
    Point t;// = Point();
    Point twod1_val;// = Point();
    if(two_step_index+twod1-1 < length){
        t = device_data[two_step_index+twod-1]; 
        twod1_val = device_data[two_step_index+twod1-1];
        device_data[two_step_index+twod-1] = twod1_val;
        device_data[two_step_index+twod1-1].sum(t); 
    }
} 

__global__ void set_extras_zeroes_kernel(Point *device_data, int length, int og_length){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= og_length && index < length){
        device_data[index] = Point();
    }
}

void exclusive_scan(Point* device_data, int length)
{
    const int threadsPerBlock = 512;
    int og_length = length;
    length = nextPow2(length);
    int blocks = (length+threadsPerBlock - 1) / threadsPerBlock;
    set_extras_zeroes_kernel<<<blocks, threadsPerBlock>>>(device_data, length, og_length);
    //cudaThreadSynchronize();
    //upsweep phase
    for (int twod = 1; twod < length; twod*=2){
        int twod1 = twod*2;
        blocks = (length/twod + threadsPerBlock) / threadsPerBlock;       
        upsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, length, twod, twod1);
        //cudaThreadSynchronize();
    }     
    const Point zero_const = Point();
    cudaMemcpy(&device_data[length-1], &zero_const, sizeof(Point),cudaMemcpyHostToDevice); 
    // downsweep phase
    for (int twod = length/2; twod >= 1; twod /= 2)
    {
        int twod1 = twod*2;
        blocks = (length/twod + threadsPerBlock) / threadsPerBlock;
        downsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, length, twod, twod1, og_length);
        //cudaThreadSynchronize();
    }
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
//----------------------------SCAN DONE---------------------------//

__global__ void mask_cluster(Point* means, Point* data, size_t* assignments, 
int total_num_points, int k, int cluster, Point *data_scratch){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < total_num_points){
        if(assignments[id]==cluster){
            data_scratch[id] = data[id];
        }
        else{
            data_scratch[id] = Point(0);
        }
        //printf("ID: %d, p.r: %f\n", id, data_scratch[id].r);
    }
}

__global__ void set_final_means(Point* means, Point* data, size_t* assignments, 
    int total_num_points, int cluster, int k, Point *p){

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id == cluster){
        if(assignments[total_num_points-1]==cluster){
            Point pp = *p;
            pp.sum(data[total_num_points-1]);
            *p = pp;
        }
        means[id].x = p->x / p->count;
        means[id].y = p->y / p->count;
        means[id].r = p->r / p->count;
        means[id].g = p->g / p->count;
        means[id].b = p->b / p->count;
    }

}

void update_mean(Point* means, Point* data, size_t* assignments, 
    int total_num_points, int k, Point *data_scratch,
    dim3 gridDim, dim3 threadsPerBlock){
    //printf("Update Mean\n");
    Point p;
    //data_scratch has the power of 2 length already
    for (size_t cluster = 0; cluster < k; ++cluster) {
        //----per cluster sum and count using scan----//
        //printf("Mask Cluster %d\n", cluster);
        mask_cluster<<<gridDim, threadsPerBlock>>>(means, data, assignments,
        total_num_points, k, cluster, data_scratch);
        //data scratch now has 0 val Points in all cluster points but
        //the current one's. 
        //printf("Scan\n");
        exclusive_scan(data_scratch, total_num_points);
        //total_num_points-1 index has sum excluding last element
        //printf("Set final means\n");
        set_final_means<<<gridDim, threadsPerBlock>>>(means, data, assignments, 
        total_num_points, cluster, k, &data_scratch[total_num_points-1]);
        //printf("Means Set\n");
    }
}

__global__ void update_mean_kernel(Point* means, Point* data, size_t* assignments, 
    int total_num_points, int k){
    int one_d_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(one_d_id < k){
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
    //: %d, %f, %f, %f\n", one_d_id, 
    //means[one_d_id].r,means[one_d_id].g,means[one_d_id].b);}

    //printf("P: x - %d, y - %d, r - %d, g - %d, b - %d, count - %d\n", p.x, p.y,
//p.r, p.g, p.b, counts);
    }
}

//assignments, means, height, width, index_x, index_y
__global__ void set_assignments(Point* data, size_t*  assignments, Point* means, int k, int width, int height ){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    int total_num_points = width * height;
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

__global__ void fill_points(Point* points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, 
    uint8_t* rgb_image, uint8_t* new_img, int* init_mean_nums){
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
    //if(points[t].b != 0){//printf("PCV: %d, %f\n", point, points[t].b);}
    
}

__global__ void set_new_img(Point* points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, 
    uint8_t* rgb_image, uint8_t* new_img, int* init_mean_nums){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    int total_num_points = width * height;
    Point p;
    int c;
    //TODO check len
    if(point<total_num_points)
    {
        c = assignments[point];
        p = means[c];
        ////printf("%d\n", point);
        new_img[CHANNEL_NUM*point] = p.r;
        new_img[CHANNEL_NUM*point+1] = p.g;
        new_img[CHANNEL_NUM*point+2] = p.b;
    }
}

__global__ void set_means_init(Point* points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, 
    uint8_t* rgb_image, uint8_t* new_img, int* init_mean_nums){
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    int t = (int) (point/CHANNEL_NUM);
    Point m;
    if(point < k){
        int init_ind = point*((height*width)/k);
        means[point] = points[init_ind];
        /*if(point==0) means[point] = Point(0,0,89,116,58);
        if(point==1) means[point] = Point(0,0,215,192,123);
        if(point==2) means[point] = Point(0,0,50,50,29);
        if(point==3) means[point] = Point(0,0,127,80,42);
        */
        m = means[point];
        //printf("M: x - %f, y - %f, r - %f, g - %f, b - %f\n", m.x, m.y,
//m.r, m.g, m.b);
    }
    
}
void k_means_main(dim3 kgridDim, dim3 kthreadsPerBlock, dim3 gridDim, dim3 threadsPerBlock, Point* points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, 
    uint8_t* rgb_image, uint8_t* new_img, int* init_mean_nums, Point *data_scratch){ 
    int total_num_points = width*height;
    fill_points<<<gridDim, threadsPerBlock>>>(points,means,assignments,number_of_iterations,
    k,height,width,rgb_image,new_img,init_mean_nums);
    //cudaThreadSynchronize();
    set_means_init<<<kgridDim, kthreadsPerBlock>>>(points,means,assignments,number_of_iterations,
    k,height,width,rgb_image,new_img,init_mean_nums);
    //cudaThreadSynchronize();
    for(int i = 0; i< number_of_iterations; i++){
        set_assignments<<<gridDim, threadsPerBlock>>>(points, assignments, means, 
        k, width, height);
        //cudaThreadSynchronize();
        //update_mean<<<kgridDim, kthreadsPerBlock>>>(means, points, assignments, 
        //total_num_points, k);
        update_mean(means, points, assignments, total_num_points, 
        k, data_scratch, gridDim, threadsPerBlock);
        //cudaThreadSynchronize();
    }
    set_new_img<<<gridDim, threadsPerBlock>>>(points,means,assignments,number_of_iterations,
    k,height,width,rgb_image,new_img,init_mean_nums);
    //cudaThreadSynchronize();
}

/*
__global__ void k_means_kernel(Point* points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, 
    uint8_t* rgb_image, uint8_t* new_img, int* init_mean_nums){ 
    int point = blockIdx.x * blockDim.x + threadIdx.x;
    ////printf("%d", point);
    int total_num_points = width * height;
    double x, y;
    uint8_t r, g, b;
    int point_channel = point*CHANNEL_NUM;
    if(point<total_num_points){
        int factor = (width*CHANNEL_NUM);
        y = (double)(point_channel/factor);
        x = (double)(point_channel%factor);
        r = rgb_image[point_channel]; 
        g = rgb_image[point_channel+1];
        b = rgb_image[point_channel+2];
        points[point] = (Point(x, y, r, g, b));
    }
    
    cudaThreadSynchronize();
    if(point < k){
        means[point] = Point(0,0,255,255,255);
    }
    cudaThreadSynchronize();
    
    for(int i = 0; i< number_of_iterations; i++){
        if(point < total_num_points){
            set_assignments(points, assignments, means, point, k, width, height);
        }
        cudaThreadSynchronize();
        // now parallelize over clusters
        // TODO USE SCAN 
        int id = point;
        if(id < k){
            update_mean(means, points, assignments, id, total_num_points);
        }
        cudaThreadSynchronize();
    }

    Point p;
    int c;
    //TODO check len
    if(point<total_num_points)
    {
        c = assignments[point];
        p = points[c];
        ////printf("%d\n", point);
        new_img[CHANNEL_NUM*point] = p.r;
        new_img[CHANNEL_NUM*point+1] = p.g;
        new_img[CHANNEL_NUM*point+2] = p.b;

    }
    cudaThreadSynchronize();
}
*/
/*
void print_df(Points* &points, int width, int height){
    int l = width*height;
    Point p;
    //printf("Size: %d\n", points.size());
    for(int i = 0; i < l; i++){
        p = points[i];
        //printf("X: %f, Y: %f, R: %f, G: %f, B: %f\n",
                p.x, p.y, p.r, p.g, p.b);
    }
    //printf("Printed DF\n");
}
*/


void k_means(uint8_t* rgb_image, int width, int height,
                  size_t k,
                  size_t number_of_iterations) {
    dim3 threadsPerBlock(BLOCK_SIDE, 1, 1);
    const int NUM_BLOCKS_X = (width*height+threadsPerBlock.x-1)/threadsPerBlock.x;
    const int NUM_BLOCKS_Y = 1;
    dim3 gridDim(NUM_BLOCKS_X , NUM_BLOCKS_Y, 1);
    int KBLOCK_SIDE = k;
    dim3 kgridDim(1 , NUM_BLOCKS_Y, 1);
    dim3 kthreadsPerBlock(KBLOCK_SIDE, 1, 1);

    Point* means_device;
    Point* points_device;
    Point *data_scratch;
    int total_num_points = height*width;
    int scratch_len = nextPow2(total_num_points);
    size_t* assignments_device;
    uint8_t* new_img_device;
    uint8_t* rgb_img_device;
    uint8_t* new_img = (uint8_t*)malloc(sizeof(uint8_t) * height * width * CHANNEL_NUM);
    int* init_mean_nums;
    //printf("ENTERED");
    cudaMalloc(&means_device, sizeof(Point) * k);
    cudaMalloc(&points_device, sizeof(Point) * height * width );
    cudaMalloc(&data_scratch, sizeof(Point) * scratch_len);

    cudaMalloc(&init_mean_nums, sizeof(int) * k );
    cudaMalloc(&assignments_device, sizeof(size_t) * height * width);
    cudaMalloc(&new_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    cudaMalloc(&rgb_img_device, sizeof(uint8_t) * height * width*CHANNEL_NUM );
    
    cudaMemcpy(rgb_img_device, rgb_image, sizeof(uint8_t) * height * width*CHANNEL_NUM, cudaMemcpyHostToDevice);
    //printf("COPIED");
    //cudaMemcpy(&new_img_device, new_img, sizeof(uint8_t) * height * width, cudaMemcpyDeviceToHost);
    /* Set seed */
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    /* Generate n floats on device */
    //curandGenerator_t gen;
    //CURAND_CALL(curandGenerateUniform(gen, init_mean_nums, height * width));
    //(Point* &points, Point* means, size_t* assignments, int number_of_iterations, int k, int height, int width, uint8_t* new_img, int* init_mean_nums)
    double start_time_exc = currentSeconds();
    k_means_main(kgridDim, kthreadsPerBlock, gridDim, threadsPerBlock, points_device, means_device, assignments_device, 
    number_of_iterations, k,  height, width, rgb_img_device, new_img_device, init_mean_nums, data_scratch);
    //printf("DONE");
    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    printf("Time: %f\n", duration_exc);
    cudaMemcpy(new_img, new_img_device, sizeof(uint8_t) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);

    //cudaMemcpy(new_img, new_img_device, sizeof(Point) * height * width * CHANNEL_NUM, cudaMemcpyDeviceToHost);
    stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
    //printf("Finished k-means\n");
}

int main(int argc, char **argv){
    //printf("Starting off ... \n");
    const char *img_file = "cs_test1.jpg";
    int width, height, bpp;
    //printf("READING");
    uint8_t* rgb_image = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    //printf("READ");
    //Point* df = get_df(rgb_image, width, height);
    
    k_means(rgb_image, width, height, 4, 2000);
    
    return 1;
    
}