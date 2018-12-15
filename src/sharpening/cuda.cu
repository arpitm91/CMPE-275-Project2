//
// Created by Aartee Kasliwal on 2018-12-09.
//

#include<stdio.h>
#include <iostream>
#include <string>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace std;

uchar* convertImage(Mat mat) {
    uchar *array = new uchar[mat.rows * mat.cols];
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

__device__
int Truncate_device(int value) {
    if (value > 255)
        return 255;
    if (value < 0)
        return 0;
    return value;
}

__global__
void sharpen_image(uchar* device_image, int rows, int cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx + 2 > cols - 1 || idy + 2 > rows - 1) {
        return;
    }

    double multiplication_matrix[3][3] = {
            {-1,-1,-1},
            {-1,9,-1},
            {-1,-1,-1},
    };


    double red = 0;
    double green = 0;
    double blue = 0;
    double sum = 0;

    for(int iy = 0; iy < 3; iy++) {
        if(idy + iy + 1 > rows - 1) {
            continue;
        }

        for (int ix = 0; ix < 3; ix++) {
            if(idx + ix + 1 > cols - 1) {
                continue;
            }

            int index = 3 * ((idx + ix + 1) + (idy + iy + 1) * cols);

            blue += double(device_image[index + 0]) * multiplication_matrix[ix][iy];
            green += double(device_image[index + 1]) * multiplication_matrix[ix][iy];
            red += double(device_image[index + 2]) * multiplication_matrix[ix][iy];
            sum+=multiplication_matrix[ix][iy];
        }
    }

    int index = 3 * ((idx + 2) + (idy + 2) * cols);

    device_image[index + 0] = Truncate_device(blue / sum);
    device_image[index + 1] = Truncate_device(green / sum);
    device_image[index + 2] = Truncate_device(red / sum);

}

int main(int argc, const char** argv){

    if (argc < 2) {
        printf("Usage: ./executable originalImagePath\n");
        return -1;
    }

    //read the image
    string originalImagePath = argv[1];
    Mat originalImage = imread(originalImagePath, CV_LOAD_IMAGE_COLOR);
    Mat outputImage = originalImage.clone();

    //check whether the image is loaded or not
    if (!originalImage.data) {
        printf("Error : No Image Data.\n");
        return -1;
    }
    printf("Image resolution: %d * %d \n", originalImage.rows, originalImage.cols);


    uchar* host_image = convertImage(originalImage);
    uchar* device_image;
    uchar* blured_image = (uchar *) malloc(3 * originalImage.rows * originalImage.cols * sizeof(uchar));


    gpuErrchk(cudaMalloc((void**) &device_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar)));
    gpuErrchk(cudaMemcpy(device_image, host_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar), cudaMemcpyHostToDevice));

    dim3 dimGrid(32, 32);
    dim3 dimBlock(32, 32);

    sharpen_image<<< dimGrid, dimBlock >>>(device_image, originalImage.rows, originalImage.cols);

    gpuErrchk(cudaMemcpy(blured_image, device_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_image));

    outputImage.data = blured_image;
    imwrite("output/bluredimage_cuda.png", outputImage);
    return 0;
}
