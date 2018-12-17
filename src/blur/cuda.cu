/*
* Created by Aartee Kasliwal on 2018-12-09.
*/

#include "cuda_blur_ops.cu"

int main(int argc, const char** argv){

	if (argc < 2) {
		printf("Usage: ./executable originalImagePath\n");
		return -1;
	}

	printf("%f", blur_image_cuda(argv[1], false));
	return 0;
}
