//
// Created by Arpit Mathur on 2018-12-09.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "omp.h"
#include <sys/time.h>

using namespace cv;
using namespace std;

int SHARPEN_FACTOR[3][3] = {
        {-1, -1, -1},
        {-1,  9,  -1},
        {-1,  -1,  -1}
};

void my_sharpen(Mat &original_image, int &i, int &j, int &newBlue, int &newGreen, int &newRed) {
    newBlue = 0;
    newGreen = 0;
    newRed = 0;
    Vec3b point_color;
    int di, dj;
    for (di = 0; di < 3; di++) {
        for (dj = 0; dj < 3; dj++) {
            point_color = original_image.at<Vec3b>(Point(i + di - 1, j + dj - 1));
            newBlue += SHARPEN_FACTOR[dj][di] * point_color[0];
            newGreen += SHARPEN_FACTOR[dj][di] * point_color[1];
            newRed += SHARPEN_FACTOR[dj][di] * point_color[2];
        }
    }

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

//    namedWindow("Display window", WINDOW_NORMAL);// Create a window for display.
    struct timeval start, end;
    gettimeofday(&start, NULL);

    int i, j, n = original_image.cols, m = original_image.rows;

    Mat sharpened_image(m, n, CV_8UC3, Scalar(255, 255, 255));

    #pragma openmp parallel for
    for (i = 0; i < n; i++) {
        int newBlue, newGreen, newRed;
        for (j = 0; j < m; j++) {
            if (i <= 0 || i >= n - 1 || j <= 0 || j >= m - 1) {
                sharpened_image.at<Vec3b>(Point(i, j)) = original_image.at<Vec3b>(Point(i, j));
                continue;
            }
            my_sharpen(original_image, i, j, newBlue, newGreen, newRed);
            sharpened_image.at<Vec3b>(Point(i, j)) = Vec3b(static_cast<uchar>(newBlue), 
                                                          static_cast<uchar>(newGreen),
                                                           static_cast<uchar>(newRed));

        }
    }
    gettimeofday(&end, NULL);

    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                   end.tv_usec - start.tv_usec) / 1.e6;
    cout<<delta;

//    hconcat(original_image, sharpened_image, concatenated_image);
//    imshow("Display window", sharpened_image);                   // Show our image inside it.
//    waitKey(0);
    return 0;
}
