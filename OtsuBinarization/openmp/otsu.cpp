/*
./otsu /home/sbali/proj/images/building.jpg <nt>
./otsu /home/sbali/proj/images/cs_test1.jpg <nt> 
*/

/*
Results

Buiding

(1, 2, 4, 8, 16, 24, 32)
0.456641
0.356257
0.322506
0.284558
0.318712
0.322246
0.354643

Dynamic
0.456174
0.380942
0.319642
0.315810
0.342669
0.365078
0.436242

(other)
(1, 2, 4, 8, 16, 24, 32)
0.082043
0.081858
0.074505
0.080633
0.101323
0.110661
0.131641

Dynamic OpenMP
(1, 2, 4, 8, 16, 24, 32)

0.082314
 0.114312
0.097692
 0.102615
 0.126754
0.112121
0.171893

*/

#include <bits/stdc++.h>
#include <algorithm>
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
#include "../../utils/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../utils/stb_image_write.h"
#include "../../utils/cycletimer.h"
#define CHANNEL_NUM 1
#define MAX_INTENSITY 256
#define OMP_NESTED true
#define NT 24

void scan_eff_parallel(int* &cum_histogram, int* &cum_sum, int nthreads){
    for(int d = 0; d<8; d++){
        #pragma omp parallel for num_threads(nthreads)
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
        #pragma omp parallel for num_threads(nthreads)
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

void scan_parallel(int* &cum_histogram, int* &cum_sum, int nthreads){
    int i, d, k;
    int log_max = 8;
    for(int i = 0; i< 10; i++){
        printf("%d ", cum_histogram[i]);
    }
    /*

    for(int k = 1; k<MAX_INTENSITY; k++){
        cum_histogram[k] = cum_histogram[k] + cum_histogram[k-1];
        cum_sum[k] = cum_sum[k] + cum_sum[k-1];

    }
    */
    for(d = 1; d<=8; d++){
        //#pragma omp parallel for num_threads(nthreads) 
        for(k=d-1; k<MAX_INTENSITY; k+= d){
            cum_histogram[k] += cum_histogram[k-d] ;
            cum_sum[k] += cum_sum[k-d] ;
        }
    }

    for(d = d>>1; d>1; d>>=1){
        //#pragma omp parallel for num_threads(nthreads) 
        for(k=d-1; k<MAX_INTENSITY-d; k+= d){
            cum_histogram[d+k] += cum_histogram[k] ;
            cum_sum[d+k] += cum_sum[k] ;
        }
    }

    printf("\n");
    for(int i = 0; i< 10; i++){
        printf("%d ", cum_histogram[i]);
    }

    return;

}


int otsu_binarization(uint8_t* &gray_img, int width, int height, int nthreads){
    // Pick centroids as random points from the dataset.
    int histogram[MAX_INTENSITY];
    int* cum_histogram = (int*)malloc(sizeof(double) * MAX_INTENSITY);
    int* cum_sum = (int*)malloc(sizeof(double) * MAX_INTENSITY);
    double* max_vars = (double*)malloc(sizeof(double)*MAX_INTENSITY);

    //vector<int> histogram(MAX_INTENSITY); //gray scale image
    int index;
    double total_sum=0.0;
    int hist_sum=0.0;
    int hist_val_sum=0.0;
    double p1_sum, p2_sum, p1_num, p2_num = 0;
    double p1_mu, p2_mu, var, threshold = 0;
    int total_pts = width * height;
    double max_var = -1;
    double mu_diff;
    int total_num = 0;
    //int thresh = 0;
    uint8_t *new_img = (uint8_t *) malloc(total_pts * sizeof(uint8_t));
    for(int iter = 0; iter<400; iter++){
    
    #pragma omp parallel for num_threads(nthreads) 
    for(int i=0; i< MAX_INTENSITY; i++){
        histogram[i] = 0;
    }

    //TEST ??
    //#pragma omp parallel for num_threads(nthreads) shared(histogram, gray_img)
    for (int i=0; i<width* height; i++) {
       // #pragma omp atomic
        histogram[gray_img[i]] += 1;
    }

    #pragma omp parallel for num_threads(nthreads) reduction(+: total_sum) schedule(dynamic)
    for(int i=0; i<MAX_INTENSITY; i++){
        total_sum += double(i) * (histogram[i]) ;
        // set initial values for the reduction
        cum_histogram[i] = histogram[i];
        cum_sum[i] = i * histogram[i];
    }

    scan_eff_parallel(cum_histogram, cum_sum, nthreads);
    
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    for(int thresh=0; thresh<MAX_INTENSITY; thresh++){
        int p1_num = cum_histogram[thresh];
        int p2_num = total_pts - p1_num;
        if(p1_num == 0 || p2_num==0){
            continue;
        }
        double p1_sum = cum_sum[thresh];
        double p2_sum = total_sum - p1_sum;
        double p1_mu = p1_sum/p1_num;
        double p2_mu = p2_sum/p2_num;
        double mu_diff = (p1_mu - p2_mu);
        double var = p1_num * p2_num * mu_diff * mu_diff;
        /*#pragma omp critical
        if(var > max_var){
            threshold = thresh;
            max_var = var;
        }*/
	    max_vars[thresh] = var;
    }


    #pragma omp parallel for num_threads(nthreads) reduction(max: max_var) 
    for(int thresh=0; thresh<MAX_INTENSITY; thresh++){
	    max_var = std::max(max_var, max_vars[thresh]);
	}
    #pragma omp parallel for num_threads(nthreads)
    for(int thresh=0; thresh<MAX_INTENSITY; thresh++){
	    if(max_vars[thresh]==max_var) threshold=thresh;
    }   
    

    #pragma omp parallel for num_threads(nthreads) 
    for(int i=0; i< total_pts; i++){
        if(gray_img[i]> threshold){
            new_img[i] = 255;
        }
        else{
            new_img[i] = 0;
        }
    }
    }

  //stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);
  //printf("Finished otsu\n");
  return 1;
}


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = argv[1];
    int width, height, bpp;
    uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);
    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);
    double start_time_exc = currentSeconds();
    
    otsu_binarization(gray_img, width, height, num_threads);
    double end_time = currentSeconds();
    double duration_exc = end_time - start_time_exc;
    fprintf(stdout, "Time: %f\n", duration_exc);
    return 1;
}