/*
* Created by Aartee Kasliwal on 2018-12-09.
*/

#include <cuda_runtime.h>

#ifndef CUDA_BLUR_OPS_H
#define CUDA_BLUR_OPS_H

using namespace cv;

double blur_image_cuda(string originalImagePath, bool isToImageWrite);

#endif