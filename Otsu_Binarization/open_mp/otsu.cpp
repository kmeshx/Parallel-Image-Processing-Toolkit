#include <bits/stdc++.h>
//#include "cycletimer.h"
#include <omp.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <math.h>

//#include <random>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define CHANNEL_NUM 1
#define MAX_INTENSITY 256
#define OMP_NESTED true


void scan_parallel(double* &cum_histogram, double* &cum_sum){
    int i, d, k;

    // up sweep
    for(d = 0; d<num_levels; d++){
        #pragma omp parallel for shared(cum_histogram, cum_sum)
        for(k=0; k<MAX_INTENSITY/1<<(d+1); k++){
            cum_histogram[k + pow(2, d+1) -1] = cum_histogram[k+pow(2, d) -1] + cum_histogram[k+pow(2, d+1)-1];
            cum_sum[k + 1<<(d+1) -1] = cum_sum[k+(1<<d) -1] + cum_sum[k+ 1<<(d+1)-1];
        }
    }

    // down sweep 
    double tmp_sum, tmp_histogram;
    a[MAX_INTENSITY-1] = 0;
    for (d = num_levels-1; d=0; d--) {
        #pragma omp parallel for shared(cum_histogram, cum_sum)
        for (k = 0; i < MAX_INTENSITY/1<<(d+1); i++){
            tmp_histogram = cum_histogram[k + 1<<d];
            tmp_sum = cum_sum[k + 1<<d];
            cum_histogram[k + 1<<d -1] = cum_histogram[k+1<<(d+1) -1];
            cum_histogram[k + 1<<(d+1) -1] = tmp_histogram + cum_histogram[k+1<<(d+1) -1];
            cum_sum[k + 1<<d -1] = cum_sum[k+1<<(d+1) -1];
            cum_sum[k + 1<<(d+1) -1] = tmp_sum + cum_sum[k+1<<(d+1) -1];
        }
    }

}


int otsu_binarization(uint8_t* &gray_img, int width, int height){
    // Pick centroids as random points from the dataset.
    double histogram[MAX_INTENSITY];
    double cum_histogram[MAX_INTENSITY];
    double cum_sum[MAX_INTENSITY];

    
    //vector<int> histogram(MAX_INTENSITY); //gray scale image
    int index;
    double total_sum=0.0;
    double hist_sum=0.0;
    double hist_val_sum=0.0;
    int i=0;
    double p1_sum, p2_sum, p1_num, p2_num = 0;
    double p1_mu, p2_mu, var, threshold = 0;
    int total_pts = width * height;
    double max_var = -1;
    double mu_diff;
    int total_num = 0;
    int thresh = 0;
    uint8_t *new_img = (uint8_t *) malloc(total_pts * sizeof(uint8_t));

    #pragma omp parallel for shared(histogram) private(i)
    for(i=0; i< MAX_INTENSITY; i++){
        histogram[i] = 0.0;
    }
  
    //TEST ??
    #pragma omp parallel for shared(histogram, gray_img)
    for (i=0; i<width* height; i++) {
        #pragma omp atomic
        histogram[gray_img[i]] += 1.0;
    }
    
    #pragma omp parallel for shared(histogram) reduction(+: total_sum)
    for(i=0; i<MAX_INTENSITY; i++){
        total_sum += double(i) * (histogram[i]) ;
        // set initial values for the reduction
        cum_histogram[i] = histogram[i];
        cum_sum[i] = i * histogram[i];
    }

    int num_levels = ceil(log(MAX_INTENSITY)/log(2));
    scan_parallel(cum_histogram, cum_sum);

    #pragma omp parallel for private(p1_num, p2_num, p1_sum, p2_sum, p1_mu, p2_mu, var, mu_diff) shared(cum_histogram, threshold, total_pts, total_sum) 
    for(thresh=0; thresh<MAX_INTENSITY; thresh++){
        p1_num = cum_histogram[thresh];
        p2_num = total_pts - p1_num;
        if(p1_num == 0 || p2_num==0){
            continue;
        }
        p1_sum = cum_sum[thresh];
        p2_sum = total_sum - p1_sum;
        p1_mu = p1_sum/p1_num;
        p2_mu = p2_sum/p2_num;
        mu_diff = (p1_mu - p2_mu);
        var = p1_num * p2_num * mu_diff * mu_diff;
        #pragma omp critical
        if(var > max_var){
            threshold = thresh;
            max_var = var;
        }
    }

    #pragma omp parallel for shared(new_img, threshold)
    for(i=0; i< total_pts; i++){
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