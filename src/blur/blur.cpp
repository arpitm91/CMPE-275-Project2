#include "opencv2/highgui/highgui.hpp"
#include <cstdio>
#include <cstring>
#include <opencv2/imgproc/imgproc.hpp>
#include "omp.h"

using namespace cv;

#define PI 3.14159265358979323846

//g++ -I /usr/local/Cellar/opencv/3.4.3_1/include -L /usr/local/Cellar/opencv/3.4.3_1/lib build.cpp -o build
//g++ build.cpp -o build `pkg-config --cflags --libs opencv`

static const char *GAUSSIAN_FILTER_NAME = "gaussian";
const int GAUSSIAN_RADIUS = 3;
const int MOTION_X = 40;
const int MOTION_Y = 5;

typedef struct {
    double value[3];
    double weight[3];
} Weights;

bool isParallel(const char * parallel);


IplImage* gaussian_blur(IplImage* image, double r);
IplImage* gaussian_blur_parallel(IplImage* image, double r);

IplImage* motion_blur(IplImage* image, int deltaX,  int deltaY);
IplImage* motion_blur_parallel(IplImage* image, int deltaX,  int deltaY);


int myMin(int a, int b) {
    return a < b ? a : b;
}

int myMax(int a, int b) {
    return a > b ? a : b;
}

int main( int argc, const char** argv )
{
    if (argc < 5) {
        printf("Should be like so: executable inputPath outputPath [gaussian|motion] [true|false]\n");
        printf("Specifying gaussian will apply the gaussian build\n");
        printf("True for parallel, false otherwise.\n");
        return -1;
    }

    // Gather arguments
    const char *inputPath = argv[1];
    const char *outputPath = argv[2];
    printf(inputPath);
    printf(outputPath);
    const char *filterName = argv[3];
    bool inParallel = isParallel(argv[4]);
    bool isGaussianFilter = strcmp(filterName, GAUSSIAN_FILTER_NAME) == 0;

    IplImage* image = cvLoadImage(inputPath, CV_LOAD_IMAGE_UNCHANGED);
    //check whether the image is loaded or not
    if (image != NULL && image->imageSize <= 0) {
        printf("Error : Image cannot be loaded..!!\n");
        return -1;
    }

    IplImage* result;
    if (inParallel) {
        if (isGaussianFilter) {
            result = gaussian_blur_parallel(image, GAUSSIAN_RADIUS);
        } else {
            result = motion_blur_parallel(image, MOTION_X, MOTION_Y);
        }
    } else {
        if (isGaussianFilter) {
            result = gaussian_blur(image, GAUSSIAN_RADIUS);
        } else {
            result = motion_blur(image, MOTION_X, MOTION_Y);
        }
    }

//    Code to see the resulting image in a window.
    namedWindow("Original Image", CV_WINDOW_AUTOSIZE); //create a window with the name "Original Image"
    namedWindow("Blurred Image", CV_WINDOW_AUTOSIZE); //create a window with the name "Blurred Image"

    cvShowImage("Original Image", image); //display the image which is stored in the 'img' in the "Original Image" window
    cvShowImage("Blurred Image", result); //display the image which is stored in the 'img' in the "Blurred Image" window

    waitKey(0); //wait infinite time for a keypress
    destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"

    return 0;
}

IplImage* gaussian_blur(IplImage* image, double r) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;
    printf("h=%d, w=%d", h, w);
    double rs = ceil(r * 2.57);     // significant radius
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = i-rs; iy<i+rs+1; iy++) {
                for (int ix = j - rs; ix < j + rs + 1; ix++) {
                    int x = myMin(w - 1, myMax(0, ix));
                    int y = myMin(h - 1, myMax(0, iy));
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


IplImage* gaussian_blur_parallel(IplImage* image, double r) {
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
                    int x = myMin(w - 1, myMax(0, ix));
                    int y = myMin(h - 1, myMax(0, iy));
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

IplImage* motion_blur(IplImage* image, int deltaX,  int deltaY) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = myMin(i + deltaY, h -1) ; iy > i; iy--) {
                CvScalar channels = cvGet2D(image, iy, j);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c] * 2;
                    weights.weight[c] += 2;
                }
            }

            for(int ix = myMin(j + deltaX, w-1); ix > j; ix--) {
                CvScalar channels = cvGet2D(image, i, ix);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c];
                    weights.weight[c] += 1;
                }
            }

            // set the value for each channel in the resulting image.
            CvScalar resultingChannels = cvGet2D(result, i, j);
            for(int c=0; c < 3; c++) {
                resultingChannels.val[c] = weights.value[c] / weights.weight[c];
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return result;
}

IplImage* motion_blur_parallel(IplImage* image, int deltaX,  int deltaY) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;
    #pragma omp parallel for schedule(guided)
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = myMin(i + deltaY, h -1) ; iy > i; iy--) {
                CvScalar channels = cvGet2D(image, iy, j);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c] * 2;
                    weights.weight[c] += 2;
                }
            }

            for(int ix = myMin(j + deltaX, w-1); ix > j; ix--) {
                CvScalar channels = cvGet2D(image, i, ix);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c];
                    weights.weight[c] += 1;
                }
            }

            // set the value for each channel in the resulting image.
            CvScalar resultingChannels = cvGet2D(result, i, j);
            for(int c=0; c < 3; c++) {
                resultingChannels.val[c] = weights.value[c] / weights.weight[c];
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return result;
}

bool isParallel(const char * parallel) {
    return strcmp(parallel, "true") == 0;
}