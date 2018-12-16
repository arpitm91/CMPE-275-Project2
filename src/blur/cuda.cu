/*
* Created by Aartee Kasliwal on 2018-12-09.
*/

#include "blur_ops.cu"

int main(int argc, const char** argv){

	if (argc < 2) {
		printf("Usage: ./executable originalImagePath\n");
		return -1;
	}

	printf("cuda parallel time = %f \n", blur_image_cuda(argv[1], true));
	return 0;
}
