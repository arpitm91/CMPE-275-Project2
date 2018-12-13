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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void add(uchar* image, int rows, int cols, int factor)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < rows; i+=stride){
        for(int j = 0; j < 3*cols; j++) {
            int x = factor * ((int(image[j*rows+i])-128)+128);
            if (x > 255)
                x = 255;
           else if(x < 0)
                x = 0;
            image[j*rows+i] = x;
         }
    }
}

int main(int argc, char **argv) {
    int contrast = 127;
    int threads = 1;
    if (argc < 2) {
        printf("Usage: ./executable filename [contrast]\n");
        return -1;
    }
    if (argc >= 3) {
        threads = atoi(argv[2]);
    }

    string inputFile = argv[1];
    Mat original_image = imread(inputFile, CV_LOAD_IMAGE_COLOR);
    uchar* image = convertImage(original_image);
    uchar* device_image;
    uchar* returned_image = (uchar *) malloc(3 * original_image.rows * original_image.cols *sizeof(uchar));
//    uchar* returned_image;

    cout << "Image Resolution: " << original_image.rows << "x" << original_image.cols << endl;
    
    float factor = (259 * (contrast + 255)) / (255 * (259 - contrast));
    printf("Calling cudaMallocManaged...\n");

    const clock_t begin_time = clock();

    cudaMalloc((void**) &device_image, 3 *  original_image.rows * original_image.cols *sizeof(uchar));
    
    //printf("Calling cudaMemCpy H->D...\n");
    gpuErrchk( cudaMemcpy(device_image, image, 3 * original_image.rows * original_image.cols *sizeof(uchar), cudaMemcpyHostToDevice));

    //printf("Calling Kernel...\n");
    add<<<1, threads>>>(device_image, original_image.rows, original_image.cols, factor);
   
    //printf("Waiting for Device Synchronize...\n");
 //   cudaDeviceSynchronize();
    //printf("Calling cudaMemCpy D->H...\n");

//    for(int i = 1000; i < 1010; i++)
//        cout<< int(returned_image[i]) << " " << int(image[i]) << endl;

    //cout << endl;
    gpuErrchk(cudaMemcpy(returned_image, device_image, 3 * original_image.rows * original_image.cols *sizeof(uchar), cudaMemcpyDeviceToHost));
    cudaFree(device_image);

    cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC;

    for (int i = 0; i < original_image.rows; i++) {
        for (int j = 0; j < original_image.cols * 3; j++) {
            image[j*original_image.rows+i] = Truncate(factor * (int(image[j*original_image.rows+i]-128)+128 ));
        }
    }
    
//    for(int i = 1000; i < 1010; i++)
//        cout<< int(returned_image[i]) << " " << int(image[i]) << endl;

    original_image.data = returned_image;

    imwrite("./output/contrast_cuda.jpg", original_image);
    return 0;
}
