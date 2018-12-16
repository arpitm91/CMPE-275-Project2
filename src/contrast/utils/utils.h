#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <unistd.h>

using namespace cv;
using namespace std;

uchar* convertImage(Mat mat) {
    // uchar *array = new uchar[mat.rows * mat.cols];
    uchar *array;
    if (mat.isContinuous())
        array = mat.data;
    return array;
}