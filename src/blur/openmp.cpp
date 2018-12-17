/*
* Created by Aartee Kasliwal on 2018-12-09.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <sys/time.h>
#include "omp.h"
#include <vector>

using namespace cv;
using namespace std;

int RADIUS = 3;

void my_blur(Mat &original_image, int &i, int &j, int &newBlue, int &newGreen, int &newRed, int& r,int&m,int&n) {
    newBlue = 0;
    newGreen = 0;
    newRed = 0;
    int sum = 0;
    Vec3b point_color;
    int di, dj;
    for (di = 0; di <= 2*r+1; di++) {
        if(i -r + di<0||i -r + di>=n)
            continue;
        for (dj = 0; dj <= 2*r+1; dj++) {
            if(j -r + dj<0||j -r + dj>=m)
                continue;
            point_color = original_image.at<Vec3b>(Point(i -r + di, j -r + dj));
            newBlue +=  point_color[0];
            newGreen += point_color[1];
            newRed += point_color[2];
            sum++;
        }
    }

    newBlue /=  sum;
    newGreen /= sum;
    newRed /= sum;

    if (newBlue < 0)
        newBlue = 0;
    if (newBlue > 255)
        newBlue = 255;

    if (newGreen < 0)
        newGreen = 0;
    if (newGreen > 255)
        newGreen = 255;

    if (newRed < 0)
        newRed = 0;
    if (newRed > 255)
        newRed = 255;
}

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat original_image, concatenated_image;
    original_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if (!original_image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int i, j, n = original_image.cols, m = original_image.rows;

    Mat blured_image(m, n, CV_8UC3, Scalar(255, 255, 255));
#pragma openmp parallel for
    for (i = 0; i < n; i++) {
        int newBlue, newGreen, newRed;
        for (j = 0; j < m; j++) {
            my_blur(original_image, i, j, newBlue, newGreen, newRed,RADIUS,m,n);
            blured_image.at<Vec3b>(Point(i, j)) = Vec3b(static_cast<uchar>(newBlue), static_cast<uchar>(newGreen),
                                                        static_cast<uchar>(newRed));

        }
    }
    gettimeofday(&end, NULL);

    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                   end.tv_usec - start.tv_usec) / 1.e6;
    cout<<delta;
//    imwrite("pic.jpg", blured_image);                   // Show our image inside it.
    return 0;
}
