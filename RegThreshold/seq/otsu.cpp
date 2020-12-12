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

int thresh(uint8_t* &gray_img, int width, int height, int threshold, int num_threads){    
    int total_pts = width * height;
    uint8_t *new_img = (uint8_t *) malloc(total_pts * sizeof(uint8_t));
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
    const char *img_file = argv[1];
    int width, height, bpp;
    uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM, argv[2]);  
    thresh_segmentation(gray_img, width, height);
    return 1;
}
