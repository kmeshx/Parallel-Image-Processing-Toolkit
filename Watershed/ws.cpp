
#include <queue>


void watershed(priority_queue<Point> pq, uint8_t **markerMap, Point **image,
		int8_t *dx, int8_t *dy, int rows, int cols, uint8_t **inPq){

	while (!pq.empty()) {
		Point top = pq.top();
		pq.pop();

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
			// Must check if all surrounding marked have the same color
			if (markerMap[next.x][next.y] != 0 && next.bw != 0) {
				if (neighboursLabel == 0) {
					neighboursLabel = markerMap[next.x][next.y];
				}
				else {
					if (markerMap[next.x][next.y] != neighboursLabel) {
						canLabel = false;
					}
				}
			}
			else {
				if (inPq[nextX][nextY] == false) {
					inPq[next.x][next.y] = true;
					pq.push(next);
				}
			}
		}

		if (canLabel) {
			markerMap[top.x][top.y] = neighboursLabel;
		}
	}
}

class Compare
{
public:
    bool operator() (Point const& p1, Point const& p2)
    {
        return p1.gr < p2.gr;
    }
};

int main(int argc, char **argv){
	printf("Starting off ... \n");
	const char *img_file = "cs_test1.jpg";
	int width, height, bpp;
	uint8_t* gray_img = stbi_load(img_file, &width, &height, &bpp, CHANNEL_NUM);
	double start_time_exc = currentSeconds();
	priority_queue<Point, vector<Point>, Compare> pq;
	//otsu_binarization(gray_img, width, height);
	double end_time = currentSeconds();
	double duration_exc = end_time - start_time_exc;
	fprintf(stdout, "Time: %f\n", duration_exc);
	return 1;
}
