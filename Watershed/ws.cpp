
#include <queue>
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

using namespace std;
struct Point;
using DataFrame = std::vector<Point>;

struct Point {
    double x, y;     // coordinates
    double bw;

    Point() : 
        x(0.0), 
        y(0.0),
        bw(0.0){}
        
    Point(double x, double y, double bw) : 
        x(x), 
        y(y),
		bw(bw) {}
};

struct Compare
{
    bool operator() (Point const& p1, Point const& p2)
    {
		//why was this gr
        return p1.bw < p2.bw;
    }
};


void watershed(priority_queue<Point, vector<Point>, Compare> pq, uint8_t **markerMap, Point **image,
		int8_t *dx, int8_t *dy, int rows, int cols, uint8_t **inPq) {

	while (!pq.empty()) {
		Point top = (pq.top());
		pq.pop();
		int tx = (int)top.x;
		int ty = (int)top.y;
		bool canLabel = true;
		int neighboursLabel = 0;
		for (int i = 0; i < 4; i++) {
			int nextX = top.x + dx[i];
			int nextY = top.y + dy[i];
			if (nextX < 0 || nextY < 0 || nextX >= rows || nextY >= cols) {
				continue;
			}
			//Pixel next = Pixel((int)image.at<uchar>(nextX, nextY), nextX, nextY);
			Point next = image[nextX][nextY];
			int nx = (int)next.x;
			int ny = (int)next.y;


			// Must check if all surrounding marked have the same color
			if (markerMap[nx][ny] != 0 && next.bw != 0) {
				if (neighboursLabel == 0) {
					neighboursLabel = markerMap[nx][ny];
				}
				else {
					if (markerMap[nx][ny] != neighboursLabel) {
						canLabel = false;
					}
				}
			}
			else {
				if (inPq[nextX][nextY] == false) {
					inPq[nx][ny] = true;
					pq.push(next);
				}
			}
		}

		if (canLabel) {
			markerMap[tx][ty] = neighboursLabel;
		}
	}
}


int main(int argc, char **argv){
	printf("Starting off ... \n");
	const char *img_file = "cs_test1.jpg";
	int width, height, bpp;
	uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);
	//double start_time_exc = currentSeconds();
	priority_queue<Point, vector<Point>, Compare> pq;
	//otsu_binarization(gray_img, width, height);
	//double end_time = currentSeconds();
	//double duration_exc = end_time - start_time_exc;
	//fprintf(stdout, "Time: %f\n", duration_exc);
	return 1;
}
