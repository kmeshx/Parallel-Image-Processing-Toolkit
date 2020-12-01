#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 1
#define MAX_INTENSITY 256

// different convolutions
// 0: VSOBEL, 1: HSOBEL

//vertical edge dection
float VSOBEL[9] = {-1.0/8, 0, 1.0/8, -2.0/8, 0, 2.0/8, -1.0/8, 0, 1.0/8};
//horizontal edge detection
//float HSOBEL[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
float HSOBEL[9] = {-1.0/8, -2.0/8, -1.0/8, 0, 0, 0, 1.0/8, 2.0/8, 1.0/8};


int otsu_binarization(uint8_t* &gray_img, int width, int height){
    double histogram[MAX_INTENSITY];    
    int index;
    double total_sum=0.0;
    double p1_sum, p2_sum, p1_num, p2_num = 0;
    double p1_mu, p2_mu, var, threshold = 0;
    int total_pts = width * height;
    double max_var = -1;
    uint8_t *new_img = (uint8_t *) malloc(total_pts * sizeof(uint8_t));

    for(int i=0; i< MAX_INTENSITY; i++){
        histogram[i] = 0.0;
    }
    for (int i=0; i<width* height; i++) {
        histogram[gray_img[i]] += 1.0;
    }
    
    for(int i=0; i<MAX_INTENSITY; i++){
        total_sum += double(i) * (histogram[i]) ;
    }

    for(int thresh=0; thresh<MAX_INTENSITY; thresh++){
        //printf("%d\n", histogram[thresh]);
        p1_num += histogram[thresh];
        p2_num = total_pts - p1_num;
        if(p1_num == 0 || p2_num==0){
            continue;
        }
        p1_sum += thresh * histogram[thresh];
        p2_sum = total_sum - p1_sum;
        p1_mu = p1_sum/p1_num;
        p2_mu = p2_sum/p2_num;
        double mu_diff = (p1_mu - p2_mu);
        var = p1_num * p2_num * mu_diff * mu_diff;
        if(var > max_var){
            //printf("SET %f %f \n", threshold, max_var);
            threshold = thresh;
            max_var = var;
        }
    }
    //printf("THRESHOLD: %f %f %f", threshold, total_sum, max_var);

    for(int i=0; i< total_pts; i++){
        if(gray_img[i]> threshold){
            new_img[i] = 255;
        }
        else{
            new_img[i] = 0;
        }
    }
  
  stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
  printf("Finished otsu\n");
  return 1;
}


//TODO what to do about negative values??
void convolve_one_pass(uint8_t* &old_img, uint8_t* &new_img, float kernel[9], int k_width, int k_height, int img_width, int img_height){
    float tmp, kernel_elem = 0.f;
    int i, j, ii, jj, id_w, id_h, k_h, k_w;
    for(j = 0; j< img_height; j++){

        for(i = 0; i< img_width; i++){
            tmp = 0.f;
            for(jj = 0; jj<k_height; jj++){
                for(ii = 0; ii< k_width; ii++){
                    id_h = j + jj;
                    id_w = i + ii;
                    k_w = k_width - ii - 1;
                    k_h = k_height - jj - 1;
                    if(id_w<img_width && id_h <img_height){
                        kernel_elem = kernel[k_h * k_width + k_w];
                        tmp += kernel_elem * (old_img[id_h * img_width + id_w]);
                    }
                }
            }            
            new_img[j * img_width + i] = (uint8_t)(tmp);
        }
    }
    otsu_binarization(new_img, img_width, img_height);
    //stbi_write_png("cs_test1_out.png", img_width, img_height, CHANNEL_NUM, new_img, img_width*CHANNEL_NUM);  
    printf("Finished edge\n");
}

int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = "building.jpg";
    int width, height, bpp;

    uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM); 
    uint8_t* new_img = (uint8_t*)malloc(width * height * sizeof(uint8_t) * CHANNEL_NUM);
    convolve_one_pass(gray_img, new_img, HSOBEL, 3, 3, width, height);
    return 1;
}
