//sample run ./edge <path> s|d|c <num_chunks-only for c>
/*
Analysis :
    check this with collapse(2)-for both kernel + image
            schedule dynamic 
            with+without vectorization
    do for multiple images otherwise too small
    different chunk size 
    do chunk size relation with the num threads
*/
/* 
    tried experiment- declare more private variables
    tried condition inside the inner for loops
    inner loop for jj shouldn't be simd for locality 
*/

#include <bits/stdc++.h>
#include "cycletimer.h"
#include <omp.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <math.h>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 1
#define MAX_INTENSITY 256
#define OMP_NESTED true
#define NT 10

// different convolutions
// 0: VSOBEL, 1: HSOBEL
//TODO maybe see if the kernel can be subdivided using SVD decomposition
//vertical edge dection
//float VSOBEL[9] = {-1.0/9, 0, 1.0/9, -2.0/9, 0, 2.0/9, -1.0/9, 0, 1.0/9};
//float VSOBEL_H[3] = {1.0/3, 1.0/3, 1.0/3};
//float VSOBEL_V[3] = {1.0/3, 1.0/3, 1.0/3};

float HSOBEL_H[3] = {-1.0, 0, 1.0};
float HSOBEL_V[3] = {1.0, 2.0, 1.0};
float HSOBEL[9] = {-1.0, -2.0, -1.0, 0, 0, 0, 1.0, 2.0, 1.0};


//TODO what to do about negative values??
//TODO check 
void convolve_one_pass(int nthreads, uint8_t* &old_img, uint8_t* &new_img, float kernel[9], int k_width, int k_height, int img_width, int img_height){
    float tmp, kernel_elem = 0.f;
    int i, j, ii, jj, id_w, id_h, k_h, k_w, ii_last, jj_last;
    // TODO SEE if beneficial to parallelize over nested fors
    
    #pragma omp parallel for collapse(2) num_threads(nthreads) private(i, ii, jj, ii_last, jj_last, tmp)
    for(j = 0; j< img_height; j++){
        //#pragma omp simd private(tmp)
        for(i = 0; i< img_width; i++){
            tmp = 0.f;
            ii_last = std::min(i+k_width, img_width);
            jj_last = std::min(j+k_height, img_height);
            #pragma omp simd reduction(+:tmp)
            for(jj = j; jj<jj_last; jj++){
                //#pragma omp simd
                for(ii = i; ii< ii_last; ii++){
                    tmp += kernel[(jj-j) * k_width + (ii-i)] * (old_img[jj * img_width + ii]);
                }
            }     
            new_img[j * img_width + i] = (uint8_t)sqrt(tmp*tmp);
        }
    }
    
    stbi_write_png("one_pass.png", img_width, img_height, CHANNEL_NUM, new_img, img_width*CHANNEL_NUM);  
}


void convolve_two_pass_locality(int nthreads, uint8_t* &old_img, uint8_t* &new_img, float kernel_h[3], float kernel_w[3], int k_width, int k_height, int img_width, int img_height){
    float tmp, kernel_elem = 0.f;
    int i, j, ii, jj, id_w, id_h, k_h, k_w, ii_last, jj_last;
    float* tmp_buf = (float*)calloc((img_height) * img_width, sizeof(float));

    #pragma omp parallel for num_threads(nthreads) private(i, ii, tmp, ii_last) //schedule(dynamic)
    for(j = 0; j< img_height; j++){
        //#pragma omp simd private(tmp)
        for(i = 0; i<img_width; i++){
            tmp = 0.f;
            ii_last = std::min(i+k_width, img_width);
            //#pragma omp simd reduction(+: tmp)
            for(ii = i; ii< ii_last; ii++){
                tmp += kernel_w[ii-i] * old_img[j*img_width + ii];
            }   
            tmp_buf[j * img_width + i] = (tmp);
        }
    }
    
    #pragma omp parallel for num_threads(nthreads) private(i, jj, tmp, jj_last) //schedule(dynamic)
    for(j = 0; j< img_height; j++){
        //#pragma omp simd private(tmp)
        for(i = 0; i<img_width; i++){
            tmp = 0.f;
            jj_last = std::min(j+k_height, img_height);
            //#pragma omp simd reduction(+: tmp)
            for(jj = j; jj< jj_last; jj++){
                tmp += kernel_h[jj-j] * tmp_buf[jj*img_width + i];
            }
            new_img[j * img_width + i] = (uint8_t)sqrt(tmp*tmp);
        }
    }
    stbi_write_png("two_pass.png", img_width, img_height, CHANNEL_NUM, new_img, img_width*CHANNEL_NUM);  
    //printf("Finished edge\n");
}

//TODO confirm the +2 parts

void convolve_chunk(int nthreads, uint8_t* &old_img, uint8_t* &new_img, int chunk_size, float kernel_h[3], float kernel_w[3], int k_width, int k_height, int img_width, int img_height){
    float tmp, kernel_elem = 0.f;
    int i, j, ii, jj, id_w, id_h, k_h, k_w;
    //chunk_size = img_height;
    int ii_last, j_last, jj_last;
    int min_j2 = -1;
    int j2;
    #pragma omp parallel for num_threads(nthreads) private(ii_last, j_last, i, j2, j, ii, jj, tmp, id_h, id_w, k_w, k_h, kernel_elem) schedule(dynamic)
    for(j = 0; j< img_height; j+=chunk_size){
        float* tmp_buf = (float*)calloc((chunk_size+2) * img_width, sizeof(float));
        //#pragma omp simd
        j_last = std::min(j+chunk_size+2, img_height);
        //TODO 2
        for(j2 = j; j2<j_last; j2++){
            //#pragma omp simd private(tmp)
            for(i = 0; i<img_width; i++){
                tmp = 0.f;
                ////#pragma omp simd
                ii_last = std::min(i+k_width, img_width);
                for(ii = i; ii< ii_last; ii++){                  
                    tmp += kernel_w[ii-i] * old_img[j2*img_width + ii];
                }   
                tmp_buf[(j2-j) * img_width + i] = tmp;
            }
        }
        //#pragma omp simd
        for(j2 = j; j2<j_last-2; j2++){
            //#pragma omp simd
            for(i = 0; i<img_width; i++){
                tmp = 0.f;
                jj_last = std::min(j2+k_height, img_height);
                //#pragma omp simd
                for(jj = j2; jj< jj_last; jj++){
                    tmp += kernel_h[jj-j2] * tmp_buf[(jj-j)*img_width + i];
                }
                new_img[j2 * img_width + i] = (uint8_t)sqrt(tmp*tmp);
            }
        }
    }
    stbi_write_png("chunk_pass.png", img_width, img_height, CHANNEL_NUM, new_img, img_width*CHANNEL_NUM);  
}

int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = argv[1];
    int width, height, bpp;
    uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM); 
    uint8_t* new_img = (uint8_t*)malloc(width * height * sizeof(uint8_t) * CHANNEL_NUM);
    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);
    char type = argv[3][0];

    double start_time_exc = currentSeconds();
    for(int i=0; i<200; i++){
        std::cout<<i<<" "<< type << "\n";
        if(type=='s'){
            convolve_one_pass(num_threads, gray_img, new_img, HSOBEL, 3, 3, width, height);
        }
        else if(type=='d'){
            convolve_two_pass_locality(num_threads, gray_img, new_img, HSOBEL_H, HSOBEL_V, 3, 3, width, height);
        }
        else if(type=='c'){
            int chunk_size = atoi(argv[4]);
            convolve_chunk(num_threads, gray_img, new_img, chunk_size, HSOBEL_H, HSOBEL_V, 3, 3, width, height);
        }
    }
    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time Without Startup: %f\n", duration_exc);
    return 1;
}
