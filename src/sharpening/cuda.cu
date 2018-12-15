//
// Created by Arpit Mathur on 2018-12-09.
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

uchar *convertImage(Mat mat) {
    uchar *array = new uchar[mat.rows * mat.cols];
    if (mat.isContinuous())
        array = mat.data;
    return array;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void sharpen_image(uchar *device_image,uchar *output_image, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx + 2 > cols - 1 || idy + 2 > rows - 1) {
        return;
    }

    int multiplication_matrix[3][3] = {
            {-1, -1, -1},
            {-1, 9,  -1},
            {-1, -1, -1},
    };


    int red = 0;
    int green = 0;
    int blue = 0;

    for (int iy = 0; iy < 3; iy++) {
        if (idy + iy + 1 > rows - 1) {
            continue;
        }

        for (int ix = 0; ix < 3; ix++) {
            if (idx + ix + 1 > cols - 1) {
                continue;
            }

            int index = 3 * ((idx + ix + 1) + (idy + iy + 1) * cols);

            blue += device_image[index + 0] * multiplication_matrix[ix][iy];
            green += device_image[index + 1] * multiplication_matrix[ix][iy];
            red += device_image[index + 2] * multiplication_matrix[ix][iy];
        }
    }

    int index = 3 * ((idx + 2) + (idy + 2) * cols);

    if (blue > 255)
        output_image[index + 0] = 255;
    else if (blue < 0)
        output_image[index + 0] = 0;
    else
        output_image[index + 0] = blue;

    if (green > 255)
        output_image[index + 1] = 255;
    else if (green < 0)
        output_image[index + 1] = 0;
    else
        output_image[index + 1] = green;

    if (red > 255)
        output_image[index + 2] = 255;
    else if (red < 0)
        output_image[index + 2] = 0;
    else
        output_image[index + 2] = red;

}

int main(int argc, const char **argv) {

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


    uchar *host_image = convertImage(originalImage);
    uchar *device_image;
    uchar *output_image;
    uchar *sharpened_image = (uchar *) malloc(3 * originalImage.rows * originalImage.cols * sizeof(uchar));


    gpuErrchk(cudaMalloc((void **) &device_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar)));
    gpuErrchk(cudaMalloc((void **) &output_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar)));
    gpuErrchk(cudaMemcpy(device_image, host_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar),
                         cudaMemcpyHostToDevice));

    dim3 dimGrid(32, 32);
    dim3 dimBlock(32, 32);

    sharpen_image << < dimGrid, dimBlock >> > (device_image,output_image, originalImage.rows, originalImage.cols);

    gpuErrchk(cudaMemcpy(sharpened_image, output_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_image));
    gpuErrchk(cudaFree(output_image));

    outputImage.data = sharpened_image;
    imwrite("output/sharpenedimage_cuda.png", outputImage);
    return 0;
}
