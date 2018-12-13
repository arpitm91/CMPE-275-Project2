//
// Created by Anuj Chaudhari on 2018-12-08.
//

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <ctime>

using namespace cv;
using namespace std;

int Truncate(int value) {
	if (value > 255)
		return 255;
	if (value < 0)
		return 0;
    return value;
}

uchar* convertImage(Mat mat) {
    uchar *array = new uchar[mat.rows * mat.cols];
    if (mat.isContinuous())
            array = mat.data;
    return array;
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
    uchar* image = convertImage(original_image);

    cout << "Image Resolution: " << original_image.rows << "x" << original_image.cols << endl;
    float factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

    const clock_t begin_time = clock();

    for (int i = 0; i < original_image.rows; i++) {
        for (int j = 0; j < original_image.cols * 3; j++) {
            image[j*original_image.rows+i] = Truncate(factor * (int(image[j*original_image.rows+i]-128)+128 ));
        }
    }
  
    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl;


    imwrite("./output/contrast_seq.jpg", original_image);
}
