//
// Created by Aartee Kasliwal on 2018-12-09.
//

#include "seq_blur_ops.h"

int main(int argc, const char** argv){

	if (argc < 2) {
		printf("Usage: ./executable originalImagePath\n");
		return -1;
	}

	printf("%f", blur_image(argv[1], "openmp", false));
	return 0;
}
