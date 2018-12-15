//
// Created by Arpit Mathur on 2018-12-09.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "omp.h"

using namespace cv;
using namespace std;

int SHARPEN_FACTOR[3][3] = {
        {-1, -1, -1},
        {-1, 8,  -1},
        {-1, -1, -1}
};

int my_sharpen(Mat& original_image, int i, int j, int index) {
    int result = 0;
    int di, dj;
    for (di = 0; di < 3; di++) {
        for (dj = 0; dj < 3; dj++) {
            result += SHARPEN_FACTOR[dj][di] * original_image.at<Vec3b>(Point(i + di - 1, j + dj - 1))[index];
        }
    }

    if (result < 0)
        return 0;
    if (result > 255)
        return 255;

    return result;
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

    namedWindow("Display window", WINDOW_NORMAL);// Create a window for display.

    int i, j, n = original_image.cols, m = original_image.rows;

    Mat sharpened_image(m, n, CV_8UC3, Scalar(255, 255, 255));

    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (i <= 0 || i >= n - 1 || j <= 0 || j >= m - 1) {
                sharpened_image.at<Vec3b>(Point(i, j)) = original_image.at<Vec3b>(Point(i, j));
                continue;
            }

            int newBlue, newGreen, newRed;
            newBlue = my_sharpen(original_image, i, j, 0);
            newGreen = my_sharpen(original_image, i, j, 1);
            newRed = my_sharpen(original_image, i, j, 2);
            sharpened_image.at<Vec3b>(Point(i, j)) = Vec3b(static_cast<uchar>(newBlue), static_cast<uchar>(newGreen),
                                                           static_cast<uchar>(newRed));

        }
    }

    hconcat(original_image, sharpened_image, concatenated_image);
    imshow("Display window", concatenated_image);                   // Show our image inside it.
    waitKey(0);
    return 0;
}
