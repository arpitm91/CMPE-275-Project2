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
#include <boost/chrono.hpp>

using namespace boost::chrono;
using namespace cv;
using namespace std;

const int GAUSSIAN_RADIUS = 20;
Mat blur_sequential(Mat image, int r);

int main(int argc, const char** argv){
    if (argc < 2) {
        printf("Usage: ./executable originalImagePath\n");
        return -1;
    }

    //read the image
    string originalImagePath = argv[1];
    Mat originalImage = imread(originalImagePath, CV_LOAD_IMAGE_COLOR);

    //check whether the image is loaded or not
    if (!originalImage.data) {
        printf("Error : No Image Data.\n");
        return -1;
    }

    // getting blurred image
    auto dt_s = high_resolution_clock::now();

    Mat blurred_image;
    for(int i = 0; i < 1; i++){
        blurred_image = blur_sequential(originalImage, GAUSSIAN_RADIUS);
    }


    // Time spent in blur_sequential
    auto dt = duration_cast<seconds> (high_resolution_clock::now() - dt_s);
    std::cout << "\ndt seq = " << dt.count() << " sec" << "\n";

    //showing original and blurred_image in one window
    Mat concatenated_image;
    namedWindow("Display window", WINDOW_AUTOSIZE);
    hconcat(originalImage, blurred_image, concatenated_image);
    imshow("Display window", concatenated_image);
    waitKey(0);
    return 0;
}

int Truncate(int value) {
    if (value > 255)
        return 255;
    if (value < 0)
        return 0;
    return value;
}

Mat blur_sequential(Mat originalImage, int r) {
//    double multiplication_matrix[7][7] = {
//            {0.000036,0.000363,0.001446,0.002291,0.001446,0.000363,0.000036},
//            {0.000363,0.003676,0.014662,0.023226,0.014662,0.003676,0.000363},
//            {0.001446,0.014662,0.058488,0.092651,0.058488,0.014662,0.001446},
//            {0.002291,0.023226,0.092651,0.146768,0.092651,0.023226,0.002291},
//            {0.001446,0.014662,0.058488,0.092651,0.058488,0.014662,0.001446},
//            {0.000363,0.003676,0.014662,0.023226,0.014662,0.003676,0.000363},
//            {0.000036,0.000363,0.001446,0.002291,0.001446,0.000363,0.000036}
//    };

//    double multiplication_matrix[9][9] = {
//        {0,0.000001,0.000014,0.000055,0.000088,0.000055,0.000014,0.000001,0},
//        {0.000001,0.000036,0.000362,0.001445,0.002289,0.001445,0.000362,0.000036,0.000001},
//        {0.000014,0.000362,0.003672,0.014648,0.023205,0.014648,0.003672,0.000362,0.000014},
//        {0.000055,0.001445,0.014648,0.058434,0.092566,0.058434,0.014648,0.001445,0.000055},
//        {0.000088,0.002289,0.023205,0.092566,0.146634,0.092566,0.023205,0.002289,0.000088},
//        {0.000055,0.001445,0.014648,0.058434,0.092566,0.058434,0.014648,0.001445,0.000055},
//        {0.000014,0.000362,0.003672,0.014648,0.023205,0.014648,0.003672,0.000362,0.000014},
//        {0.000001,0.000036,0.000362,0.001445,0.002289,0.001445,0.000362,0.000036,0.000001},
//        {0,0.000001,0.000014,0.000055,0.000088,0.000055,0.000014,0.000001,0}
//    };


//    double multiplication_matrix[11][11] = {
//        {0,0,0,0,0.000001,0.000001,0.000001,0,0,0,0},
//        {0,0,0.000001,0.000014,0.000055,0.000088,0.000055,0.000014,0.000001,0,0},
//        {0,0.000001,0.000036,0.000362,0.001445,0.002289,0.001445,0.000362,0.000036,0.000001,0},
//        {0,0.000014,0.000362,0.003672,0.014648,0.023204,0.014648,0.003672,0.000362,0.000014,0},
//        {0.000001,0.000055,0.001445,0.014648,0.058433,0.092564,0.058433,0.014648,0.001445,0.000055,0.000001},
//        {0.000001,0.000088,0.002289,0.023204,0.092564,0.146632,0.092564,0.023204,0.002289,0.000088,0.000001},
//        {0.000001,0.000055,0.001445,0.014648,0.058433,0.092564,0.058433,0.014648,0.001445,0.000055,0.000001},
//        {0,0.000014,0.000362,0.003672,0.014648,0.023204,0.014648,0.003672,0.000362,0.000014,0},
//        {0,0.000001,0.000036,0.000362,0.001445,0.002289,0.001445,0.000362,0.000036,0.000001,0},
//        {0,0,0.000001,0.000014,0.000055,0.000088,0.000055,0.000014,0.000001,0,0},
//        {0,0,0,0,0.000001,0.000001,0.000001,0,0,0,0}
//    };

    double multiplication_matrix[41][41] = {{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000001,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000014,0.000055,0.000088,0.000055,0.000014,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000036,0.000362,0.001445,0.002289,0.001445,0.000362,0.000036,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000014,0.000362,0.003672,0.014648,0.023204,0.014648,0.003672,0.000362,0.000014,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000055,0.001445,0.014648,0.058433,0.092564,0.058433,0.014648,0.001445,0.000055,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000088,0.002289,0.023204,0.092564,0.146632,0.092564,0.023204,0.002289,0.000088,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000055,0.001445,0.014648,0.058433,0.092564,0.058433,0.014648,0.001445,0.000055,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000014,0.000362,0.003672,0.014648,0.023204,0.014648,0.003672,0.000362,0.000014,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000036,0.000362,0.001445,0.002289,0.001445,0.000362,0.000036,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000014,0.000055,0.000088,0.000055,0.000014,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000001,0.000001,0.000001,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                            {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    };


    Mat blurred_image = originalImage.clone();
    int h = originalImage.rows;
    int w = originalImage.cols;

#pragma omp parallel for schedule(guided)
    for(int i = 0; i < w; i++) {
        if(i+2*r>w - 1) {
            continue;
        }
        for (int j = 0; j < h; j++) {
            if(j+2*r>h - 1) {
                continue;
            }

            double red = 0;
            double green = 0;
            double blue = 0;
            double sum = 0;

            for(int iy = 0; iy < 2 * r + 1; iy++) {
                if(j + iy+r > h - 1) {
                    continue;
                }

                for (int ix = 0; ix < 2 * r + 1; ix++) {
                    if(i + ix +r> w - 1) {
                        continue;
                    }

                    Vec3b channels = originalImage.at<Vec3b>(Point(i + ix + r, j + iy + r));

                    blue += double(channels[0]) * multiplication_matrix[ix][iy];
                    green += double(channels[1]) * multiplication_matrix[ix][iy];
                    red += double(channels[2]) * multiplication_matrix[ix][iy];
                    sum+=multiplication_matrix[ix][iy];
                }
            }
            blurred_image.at<Vec3b>(Point(i+2*r, j+2*r)) = Vec3b(Truncate(blue/sum), Truncate(green/sum), Truncate(red/sum));
        }
    }
    return blurred_image;
}