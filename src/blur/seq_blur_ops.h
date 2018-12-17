//
// Created by Aartee Kasliwal on 2018-12-09.
//

#include <cstdio>
#include <iostream>
#include <string>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>
#include "omp.h"

using namespace cv;
using namespace std;

const int GAUSSIAN_RADIUS = 20;

int Truncate(int value) {
    if (value > 255)
        return 255;
    if (value < 0)
        return 0;
    return value;
}

Mat blur_sequential(Mat originalImage, int multiplication_matrix[], int radius) {
    Mat blurred_image = originalImage.clone();
    int h = originalImage.rows;
    int w = originalImage.cols;

    for(int i = 0; i < w; i++) {
        if(i + 2 *  radius > w - 1) {
            continue;
        }
        for (int j = 0; j < h; j++) {
            if(j + 2 * radius > h - 1) {
                continue;
            }

            double red = 0;
            double green = 0;
            double blue = 0;
            double sum = 0;

            for(int iy = 0; iy < 2 * radius + 1; iy++) {
                if(j + iy + radius > h - 1) {
                    continue;
                }

                for (int ix = 0; ix < 2 * radius + 1; ix++) {
                    if(i + ix + radius > w - 1) {
                        continue;
                    }

                    Vec3b channels = originalImage.at<Vec3b>(Point(i + ix + radius, j + iy + radius));

                    blue += double(channels[0]) * multiplication_matrix[ix * radius + iy];
                    green += double(channels[1]) * multiplication_matrix[ix * radius + iy];
                    red += double(channels[2]) * multiplication_matrix[ix * radius + iy];
                    sum += multiplication_matrix[ix * radius + iy];
                }
            }
            blurred_image.at<Vec3b>(Point(i + 2 * radius, j + 2 * radius)) = Vec3b(Truncate(blue/sum), Truncate(green/sum), Truncate(red/sum));
        }
    }
    return blurred_image;
}

Mat blur_openmp(Mat originalImage, int multiplication_matrix[], int radius) {
    Mat blurred_image = originalImage.clone();
    int h = originalImage.rows;
    int w = originalImage.cols;

    #pragma omp parallel for
    for(int i = 0; i < w; i++) {
        if(i + 2 *  radius > w - 1) {
            continue;
        }
        for (int j = 0; j < h; j++) {
            if(j + 2 * radius > h - 1) {
                continue;
            }

            double red = 0;
            double green = 0;
            double blue = 0;
            double sum = 0;

            for(int iy = 0; iy < 2 * radius + 1; iy++) {
                if(j + iy + radius > h - 1) {
                    continue;
                }

                for (int ix = 0; ix < 2 * radius + 1; ix++) {
                    if(i + ix + radius > w - 1) {
                        continue;
                    }

                    Vec3b channels = originalImage.at<Vec3b>(Point(i + ix + radius, j + iy + radius));

                    blue += double(channels[0]) * multiplication_matrix[ix * radius + iy];
                    green += double(channels[1]) * multiplication_matrix[ix * radius + iy];
                    red += double(channels[2]) * multiplication_matrix[ix * radius + iy];
                    sum += multiplication_matrix[ix * radius + iy];
                }
            }
            blurred_image.at<Vec3b>(Point(i + 2 * radius, j + 2 * radius)) = Vec3b(Truncate(blue/sum), Truncate(green/sum), Truncate(red/sum));
        }
    }
    return blurred_image;
}

double blur_image(string originalImagePath, string sequentialOrOpenmp, bool isToImageWrite){

    Mat originalImage = imread(originalImagePath, CV_LOAD_IMAGE_COLOR);

    //check whether the image is loaded or not
    if (!originalImage.data) {
        printf("Error : No Image Data.\n");
        return -1;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int size = (2 * GAUSSIAN_RADIUS + 1) * (2 * GAUSSIAN_RADIUS + 1);
    int multiplication_matrix[size];

    for(int i = 0; i < size; i++){
        multiplication_matrix[i] = 1;
    }

    Mat blurred_image;
    if(sequentialOrOpenmp == "sequential"){
        blurred_image = blur_sequential(originalImage, multiplication_matrix, GAUSSIAN_RADIUS);
    }
    else if(sequentialOrOpenmp == "openmp"){
        blurred_image = blur_openmp(originalImage, multiplication_matrix, GAUSSIAN_RADIUS);
    }
    
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
         end.tv_usec - start.tv_usec) / 1.e6;

    //showing original and blurred_image in one window
    if(isToImageWrite){
        Mat concatenated_image;
        namedWindow("Display window", WINDOW_AUTOSIZE);
        hconcat(originalImage, blurred_image, concatenated_image);
        imshow("Display window", concatenated_image);
        waitKey(0);
    }

    return delta;
}