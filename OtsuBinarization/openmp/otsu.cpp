#include <bits/stdc++.h>
#include <algorithm>
#include <omp.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <math.h>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "../../utils/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../utils/stb_image_write.h"
#include "../../utils/cycletimer.h"
#define CHANNEL_NUM 1
#define MAX_INTENSITY 256
#define OMP_NESTED true
#define NT 24
#define NL 16

void scan_parallel(double* &cum_histogram, double* &cum_sum, int nthreads){
    for(int d = 0; d<8; d++){
        #pragma omp parallel for
        for(int k=0; k<MAX_INTENSITY; k+=(1<<(d+1)))
        {
            cum_histogram[(1<<(d+1))+k-1] += cum_histogram[(1<<d) + k-1] ;
            cum_sum[(1<<(d+1))+k-1] += cum_sum[(1<<d )+ k-1];
        }
    }
    
    // down sweep
    double tmp_sum, tmp_histogram;
    cum_histogram[MAX_INTENSITY-1] = 0;
    cum_sum[MAX_INTENSITY-1] = 0;
    for (int d = 7; d>=0; d--) {
        #pragma omp parallel for
        for (int k = 0; k < MAX_INTENSITY; k+= (1<<(d+1)))
        {
            tmp_histogram = cum_histogram[(1<<d)+k-1];
            tmp_sum = cum_sum[(1<<d)+k-1];
            cum_histogram[(1<<d)+k-1] = cum_histogram[(1<<(d+1)) + k-1];
            cum_histogram[(1<<(d+1))+k-1] += tmp_histogram ;
            cum_sum[(1<<d)+k-1] = cum_sum[(1<<(d+1)) + k-1];
            cum_sum[(1<<(d+1))+k-1] += tmp_sum ;
        }
    }


}


int otsu_binarization(uint8_t* &gray_img, int width, int height, int nthreads){
    // Pick centroids as random points from the dataset.
    double histogram[MAX_INTENSITY];
    double* cum_histogram = (double*)malloc(sizeof(double) * MAX_INTENSITY);
    double* cum_sum = (double*)malloc(sizeof(double) * MAX_INTENSITY);
    double* max_vars = (double*)malloc(sizeof(double)*MAX_INTENSITY);

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
    omp_lock_t lock[NL];
    #pragma omp parallel for
    for(i=0; i< MAX_INTENSITY; i++){
        histogram[i] = 0.0;
    }
    //DEL 
    /*
    FOR TESTING
    #pragma omp parallel for
    for (int i=0; i<NL; i++)
        omp_init_lock(&(lock[i]));
    */


    
    for (i=0; i<width* height; i++) {
        int item = gray_img[i];
        histogram[item] += 1.0;
    }


    #pragma omp parallel for num_threads(nthreads) shared(histogram) reduction(+: total_sum)
    for(i=0; i<MAX_INTENSITY; i++){
        total_sum += double(i) * (histogram[i]) ;
        // set initial values for the reduction
        cum_histogram[i] = histogram[i];
        cum_sum[i] = i * histogram[i];
    }

    scan_parallel(cum_histogram, cum_sum, nthreads);

    #pragma omp parallel for num_threads(nthreads) private(p1_num, p2_num, p1_sum, p2_sum, p1_mu, p2_mu, var, mu_diff) shared(cum_histogram, threshold, total_pts, total_sum)
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
	    max_vars[thresh] = var;
    }
    

    #pragma omp parallel for num_threads(nthreads) shared(max_vars) reduction(max: max_var)
    for(thresh=0; thresh<MAX_INTENSITY; thresh++){
	    max_var = std::max(max_var, max_vars[thresh]);
	}
    #pragma omp parallel for num_threads(nthreads) shared(max_vars)
    for(thresh=0; thresh<MAX_INTENSITY; thresh++){
	if(max_vars[thresh]==max_var) threshold=thresh;
    }
    

    #pragma omp parallel for num_threads(nthreads) shared(new_img, threshold)
    for(i=0; i< total_pts; i++){
        if(gray_img[i]> threshold){
            new_img[i] = 255;
        }
        else{
            new_img[i] = 0;
        }
    }

  stbi_write_png("check.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);
  //printf("Finished otsu\n");
  return 1;
}


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = argv[1];
    int width, height, bpp;
    uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);
    int nthreads = atoi(argv[2]);
    omp_set_num_threads(nthreads);
    double start_time_exc = currentSeconds();
    for(int i =0; i<400; i++)
    otsu_binarization(gray_img, width, height, nthreads);
    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time: %f\n", duration_exc);
    return 1;
}
