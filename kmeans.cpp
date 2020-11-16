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
#define CHANNEL_NUM 3

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

DataFrame get_df(uint8_t *rgb_image, int width, int height){
    DataFrame points;
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
        //printf("r: %f, g: %f, b: %f\n", r, g, b);
        points.push_back(Point(x, y, r, g, b));
        //points[i] = Point(x, y, r, g, b);
    }
   return points; 
}


uint8_t *get_raw(std::vector<size_t> assignments, DataFrame& df){
    int len = assignments.size();
    uint8_t *rgb_image = (uint8_t *) malloc(len*CHANNEL_NUM);
    Point p;
    int c;
    for(int i=0; i < len; i++){
        c = assignments.at(i);
        p = df.at(c);
        rgb_image[CHANNEL_NUM*i] = p.r;
        rgb_image[CHANNEL_NUM*i+1] = p.g;
        rgb_image[CHANNEL_NUM*i+2] = p.b;
    }
    return rgb_image;
}

DataFrame print_df(DataFrame& points, int width, int height){
    int l = width*height;
    Point p;
    printf("Size: %d\n", points.size());
    for(int i = 0; i < l; i++){
        //printf("i: %d\n", i);
        p = points[i];

        printf("X: %f, Y: %f, R: %f, G: %f, B: %f\n",
                p.x, p.y, p.r, p.g, p.b);
    }
    printf("Printed DF\n");
}

DataFrame k_means(DataFrame& data, int width, int height,
                  size_t k,
                  size_t number_of_iterations) {
  static std::random_device seed;
  static std::mt19937 random_number_generator(seed());
  std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

  // Pick centroids as random points from the dataset.
  DataFrame means(k);
  int index;
  for (int i=0; i<k; i++) {
    index = (int)(indices(random_number_generator)/CHANNEL_NUM);
    means.at(i) = data[index];
  }
  //check means
  //printf("Means Size: %d\n", means.size());
  //for(int i=0; i<k; i++){ printf("Mean[%d] Index in Data: %f\n", i, means.at(i).x);}
  std::vector<size_t> assignments(data.size());
  for(size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
    // Find assignments.
    for (size_t point = 0; point < data.size(); ++point) {
      double best_distance = std::numeric_limits<double>::max();
      size_t best_cluster = 0;
      for (size_t cluster = 0; cluster < k; ++cluster) {
        double distance = data[point].color_distance(means[cluster]);
        //squared_l2_distance(data[point], means[cluster]);
        if (distance < best_distance) {
          best_distance = distance;
          best_cluster = cluster;
        }
      }
      assignments[point] = best_cluster;
    }

    // Sum up and count points for each cluster.
    DataFrame new_means(k);
    std::vector<size_t> counts(k, 0);
    for (size_t point = 0; point < data.size(); ++point) {
      const auto cluster = assignments[point];
      new_means[cluster].x += data[point].x;
      new_means[cluster].y += data[point].y;
      //also assign average color
      new_means[cluster].r += data[point].r;
      new_means[cluster].g += data[point].g;
      new_means[cluster].b += data[point].b;
      counts[cluster] += 1;
    }

    // Divide sums by counts to get new centroids.
    for (size_t cluster = 0; cluster < k; ++cluster) {
      // Turn 0/0 into 0/1 to avoid zero division.
      const auto count = std::max<size_t>(1, counts[cluster]);
      means[cluster].x = new_means[cluster].x / count;
      means[cluster].y = new_means[cluster].y / count;
      //also for colors
      means[cluster].r = new_means[cluster].r / count;
      means[cluster].g = new_means[cluster].g / count;
      means[cluster].b = new_means[cluster].b / count;
    }

    }

  uint8_t *new_img = get_raw(assignments, means);
  stbi_write_png("cs_test1_out.png", width, height, CHANNEL_NUM, new_img, width*CHANNEL_NUM);  
  //check means
  //printf("Means Size: %d\n", means.size());
  //for(int i=0; i<k; i++){ printf("Mean[%d] Index in Data: %f\n", i, means.at(i).x);}
  printf("Finished k-means\n");
  return means;
}


int main(int argc, char **argv){
    printf("Starting off ... \n");
    const char *img_file = "cs_test1.jpg";
    int width, height, bpp;

    uint8_t* rgb_image = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);
    //raw_print(rgb_image, width, height);    
    DataFrame df = get_df(rgb_image, width, height);
    //print_df(df, width, height);
    k_means(df, width, height, 3, 2);
    //stbi_image_free(rgb_image);
    //rgb_image = (uint8_t *) malloc(width*height*CHANNEL_NUM);
    //stbi_write_png("write_test1.png", width, height, CHANNEL_NUM, rgb_image, width*CHANNEL_NUM);

    return 1;
    //exit(1);
    
}
