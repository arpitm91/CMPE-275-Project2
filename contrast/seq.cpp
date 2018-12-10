//
// Created by Anuj Chaudhari on 2018-12-08.
//

/*
 g++ seq.cpp -o seq_contrast `pkg-config --cflags --libs opencv`
 */

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <unistd.h>

using namespace cv;
using namespace std;

int Truncate(int value) {
    if (value > 255)
        return 255;
    if (value < 0)
        return 0;
    return value;
}

int main(int argc, char **argv) {
    int contrast = 127;

    if (argc < 2) {
        printf("Usage: ./executable filename [contrast]\n");
        return -1;
    }
    if (argc >= 3) {
        contrast = atoi(argv[2]);
    }

    string inputFile = argv[1];

    Mat original_image = imread(inputFile, CV_LOAD_IMAGE_COLOR);
    if (!original_image.data) {
        cout << "Could not open or find the image." << endl;
        return -1;
    }
    cout << "Image Resolution: " << original_image.rows << "x" << original_image.cols << endl;
    float factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

    Mat contrast_image(original_image.cols, original_image.rows, CV_8UC3, Scalar(255, 255, 255));

    for (int i = 0; i < original_image.cols; i++) {
        for (int j = 0; j < original_image.rows; j++) {
            Vec3b color = original_image.at<Vec3b>(Point(i, j));
            int newBlue = Truncate(factor * (int(color[0]) - 128) + 128);
            int newGreen = Truncate(factor * (int(color[1]) - 128) + 128);
            int newRed = Truncate(factor * (int(color[2]) - 128) + 128);

            contrast_image.at<Vec3b>(Point(i, j)) = Vec3b(newBlue, newGreen, newRed);

        }
    }

    Mat concatenated_image;
    namedWindow("Display window", WINDOW_AUTOSIZE);
    hconcat(original_image, contrast_image, concatenated_image);
    imshow("Display window", concatenated_image);
    waitKey(0);
    return 0;
}