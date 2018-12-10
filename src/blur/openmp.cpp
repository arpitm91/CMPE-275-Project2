//
// Created by Aartee Kasliwal on 2018-12-09.
//

#include <cstdio>
#include <iostream>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/chrono.hpp>

using namespace boost::chrono;
using namespace cv;

#define PI 3.14159265358979323846
const int GAUSSIAN_RADIUS = 5;

typedef struct {
    double value[3];
    double weight[3];
} Weights;

IplImage* blur_openmp(IplImage* image, double r);

int main(int argc, const char** argv){
    const char *originalImagePath = argv[1];
    IplImage* originalImage = cvLoadImage(originalImagePath, CV_LOAD_IMAGE_COLOR);

    //check whether the image is loaded or not
    if (originalImage != NULL && originalImage->imageSize <= 0) {
        printf("Error : No Image Data.\n");
        return -1;
    }

    // getting blurred image
    IplImage* blurred_image;
    auto dt_s = high_resolution_clock::now();

    blurred_image = blur_openmp(originalImage, GAUSSIAN_RADIUS);

    // Time spent in blur_openmp
    auto dt = duration_cast<seconds> (high_resolution_clock::now() - dt_s);
    std::cout << "\ndt seq = " << dt.count() << " sec" << "\n";

    //showing original and blurred_image in one window

    IplImage* concatenated_image = cvCreateImage(cvSize(2 * originalImage->width,originalImage->height), IPL_DEPTH_8U, 3);

    // Copy originalImage to concatenated_image
    cvSetImageROI(concatenated_image, cvRect(0, 0, originalImage->width, originalImage->height));
    cvCopy(originalImage, concatenated_image, NULL);
    cvResetImageROI(concatenated_image);

    // Copy blurred_image to concatenated_image
    cvSetImageROI(concatenated_image, cvRect(blurred_image->width, 0, blurred_image->width, blurred_image->height));
    cvCopy(blurred_image, concatenated_image, NULL);
    cvResetImageROI(concatenated_image);

    namedWindow("Blurred Image", WINDOW_AUTOSIZE );
    cvShowImage("Blurred Image", concatenated_image);

    waitKey(0);
    return 0;
}

IplImage* blur_openmp(IplImage* image, double r) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;

    double rs = ceil(r * 2.57);     // significant radius
    #pragma omp parallel for schedule(guided)
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = i-rs; iy<i+rs+1; iy++) {
                for (int ix = j - rs; ix < j + rs + 1; ix++) {
                    int x = min(w - 1, max(0, ix));
                    int y = min(h - 1, max(0, iy));

                    double dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i);
                    double wght = exp(-dsq / (2 * r * r)) / (PI * 2 * r * r);
                    CvScalar channels = cvGet2D(image, y, x);

                    // calculate the value for each channel
                    for (int c = 0; c < 3; c++) {
                        weights.value[c] += channels.val[c] * wght;
                        weights.weight[c] += wght;
                    }
                }
            }

            // set the value for each channel in the resulting image.
//            printf("i=%d, j=%d, r=%f, g=%f, b=%f\n", i, j, weights.value[0], weights.value[1], weights.value[2]);
            CvScalar resultingChannels = cvGet2D(result, i, j);
            for(int c=0; c < 3; c++) {
                resultingChannels.val[c] = round(weights.value[c] / weights.weight[c]);
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return result;
}
