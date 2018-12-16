//
// Created by Aartee Kasliwal on 2018-12-09.
//

#include "seq_blur_ops.cpp"

int main(int argc, const char** argv){

	if (argc < 2) {
		printf("Usage: ./executable originalImagePath\n");
		return -1;
	}

	printf("blur image sequential time = %f \n", blur_image(argv[1], "sequential", true));
	return 0;
}