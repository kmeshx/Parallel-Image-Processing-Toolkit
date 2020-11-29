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

int otsu_binarization(uint8_t* &gray_img, int width, int height){
    // Pick centroids as random points from the dataset.
    double histogram[MAX_INTENSITY];
    
    //vector<int> histogram(MAX_INTENSITY); //gray scale image
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


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = "cs_test1.jpg";
    int width, height, bpp;
    uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);  
    otsu_binarization(gray_img, width, height);
    return 1;
}