#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <unistd.h>

using namespace cv;
using namespace std;

Mat contrast_sequantial(Mat original_image, int contrast) { 
    //uchar* image = convertImage(original_image);
    float factor = (259 * (contrast + 255)) / (255 * (259 - contrast));
    for (int i = 0; i < original_image.rows; i++) {
        for (int j = 0; j < original_image.cols; j++) {

            Vec3b color = original_image.at<Vec3b>(Point(j, i));
            int newBlue = Truncate(factor * (int(color[0]) - 128) + 128);
            int newGreen = Truncate(factor * (int(color[1]) - 128) + 128);
            int newRed = Truncate(factor * (int(color[2]) - 128) + 128);

            original_image.at<Vec3b>(Point(j, i)) = Vec3b(newBlue, newGreen, newRed);

            //image[j*original_image.rows+i] = Truncate(factor * (int(image[j*original_image.rows+i]-128)+128 ));
        }
    }
    //original_image.data = image;
    return original_image;
}