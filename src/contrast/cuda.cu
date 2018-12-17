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
#include <math.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

__device__
int TruncateDevice(int value) {
	if (value > 255)
		return 255;
	if (value < 0)
		return 0;
    return value;
}

int Truncate(int value) {
    if (value > 255)
        return 255;
    if (value < 0)
        return 0;
    return value;
}

uchar* convertImage(Mat mat) {
    uchar *array;
    if (mat.isContinuous())
        array = mat.data;
    return array;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void contrast_image(uchar* image, int rows, int cols, int factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    long index = (idx + idy * cols) * 3;

    if (idx >= cols || idy >= rows)
        return;

    image[index] = TruncateDevice( factor * ((int(image[index]) - 128) + 128) );
    image[index+1] = TruncateDevice( factor * ((int(image[index+1]) - 128) + 128) );
    image[index+2] = TruncateDevice( factor * ((int(image[index+2]) - 128) + 128) );
}

int main(int argc, char **argv) {
    int contrast = 127;
    if (argc < 2) {
        printf("Usage: ./executable filename [contrast]\n");
        return -1;
    }

    string inputFile = argv[1];
    Mat original_image = imread(inputFile, CV_LOAD_IMAGE_COLOR);
    uchar* image = convertImage(original_image);
    uchar* device_image;

    float factor = (259 * (contrast + 255)) / (255 * (259 - contrast));

    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    dim3 dimBlock(32,32);
    dim3 dimGrid;
    dimGrid.x = ceil(float(original_image.cols) / 32);
    dimGrid.y = ceil(float(original_image.rows) / 32);

    gpuErrchk(cudaMalloc((void**) &device_image, 3 *  original_image.rows * original_image.cols *sizeof(uchar))); 
    gpuErrchk(cudaMemcpy(device_image, image, 3 * original_image.rows * original_image.cols *sizeof(uchar), cudaMemcpyHostToDevice));
    contrast_image<<< dimGrid, dimBlock >>>(device_image, original_image.rows, original_image.cols, factor);
    gpuErrchk(cudaMemcpy(image, device_image, 3 * original_image.rows * original_image.cols *sizeof(uchar), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_image));

    gettimeofday(&end, NULL);
    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                             end.tv_usec - start.tv_usec) / 1.e6;
    cout << delta;
    original_image.data = image;
    // imwrite("./output/contrast_cuda.jpg", original_image);
    return 0;
}
