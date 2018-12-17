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
#include <sys/time.h>

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

    string inputFile = argv[1];

    if (argc < 2) {
        printf("Usage: ./executable filename [contrast]\n");
        return -1;
    }
    if (argc >= 3) {
        contrast = atoi(argv[2]);
    }

    Mat original_image = imread(inputFile, CV_LOAD_IMAGE_COLOR);
    if (!original_image.data) {
        cout << "Could not open or find the image." << endl;
        return -1;
    }
    float factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < original_image.cols; i++) {
        for (int j = 0; j < original_image.rows; j++) {
            Vec3b color = original_image.at<Vec3b>(Point(i, j));

            int newBlue = Truncate(factor * (int(color[0]) - 128) + 128);
            int newGreen = Truncate(factor * (int(color[1]) - 128) + 128);
            int newRed = Truncate(factor * (int(color[2]) - 128) + 128);

            original_image.at<Vec3b>(Point(i, j)) = Vec3b(newBlue, newGreen, newRed);
        }
    }

    gettimeofday(&end, NULL);

    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u + 
                     end.tv_usec - start.tv_usec) / 1.e6;
    cout << delta;
    // imwrite("./output/contrast-omp.jpg", concatenated_image);
    return 0;
}
