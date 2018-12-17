//
// Created by Arpit Mathur on 2018-12-09.
//

#include <stdio.h>
#include <iostream>
#include <string>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <sys/time.h>

using namespace cv;
using namespace std;

int kernel_cpu[9] = {
        -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1
};

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
void sharpen_image(uchar *device_image, uchar *output_image, int *kernel, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx + 2 > cols - 1 || idy + 2 > rows - 1) {
        return;
    }


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

            blue += device_image[index + 0] * kernel[ix * 3 + iy];
            green += device_image[index + 1] * kernel[ix * 3 + iy];
            red += device_image[index + 2] * kernel[ix * 3 + iy];
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
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }
    Mat originalImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    //check whether the image is loaded or not
    if (!originalImage.data) {
        printf("Error : No Image Data.\n");
        return -1;
    }
    Mat outputImage = originalImage.clone();

    const clock_t begin_time = clock();
    uchar *host_image = convertImage(originalImage);
    uchar *device_image;
    uchar *output_image;
    int *kernel;

    gpuErrchk(cudaMalloc((void **) &device_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar)));
    gpuErrchk(cudaMalloc((void **) &output_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar)));
    gpuErrchk(cudaMalloc((void **) &kernel, 9 * sizeof(int)));
    gpuErrchk(cudaMemcpy(device_image, host_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar),
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kernel, kernel_cpu, 9 * sizeof(int), cudaMemcpyHostToDevice));

    dim3 dimBlock(32, 32);
    dim3 dimGrid;
    dimGrid.x = ceil(float(originalImage.cols) / 32);
    dimGrid.y = ceil(float(originalImage.rows) / 32);

    sharpen_image << < dimGrid, dimBlock >> >
                                (device_image, output_image, kernel, originalImage.rows, originalImage.cols);

    gpuErrchk(cudaMemcpy(host_image, output_image, 3 * originalImage.rows * originalImage.cols * sizeof(uchar),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(device_image));
    gpuErrchk(cudaFree(output_image));
    gpuErrchk(cudaFree(kernel));

    outputImage.data = host_image;
    gettimeofday(&end, NULL);

    float delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                   end.tv_usec - start.tv_usec) / 1.e6;
    cout<<delta;
//    imwrite("output/sharpenedimage_cuda.png", outputImage);

    return 0;
}
