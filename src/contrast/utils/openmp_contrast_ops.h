#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include "utils.h"

using namespace cv;
using namespace std;

Mat contrast_sequantial(Mat original_image) { 
    
    Mat contrast_image(original_image.rows, original_image.cols, CV_8UC3, Scalar(255, 255, 255));    

    #pragma omp parallel for
    for (int i = 0; i < original_image.cols; i++) {
        for (int j = 0; j < original_image.rows; j++) {
            Vec3b color = original_image.at<Vec3b>(Point(i, j));
            int newBlue = Truncate(factor * (int(color[0]) - 128) + 128);
            int newGreen = Truncate(factor * (int(color[1]) - 128) + 128);
            int newRed = Truncate(factor * (int(color[2]) - 128) + 128);

            contrast_image.at<Vec3b>(Point(i, j)) = Vec3b(newBlue, newGreen, newRed);
        }
    }
    return contrast_image;
}