cmake:
	mkdir -p build
	cd build && cmake -DCMAKE_CXX_COMPILER=g++-8 ..

compile:
	mkdir -p bin
	g++-8 ./src/contrast/seq.cpp `pkg-config --cflags --libs opencv` -o ./bin/contrast-seq
	g++-8 ./src/contrast/openmp.cpp `pkg-config --cflags --libs opencv` -o ./bin/contrast-omp
	g++-8 ./src/blur/seq.cpp `pkg-config --cflags --libs opencv` -o ./bin/blur-seq
	g++-8 ./src/blur/openmp.cpp `pkg-config --cflags --libs opencv` -o ./bin/blur-omp
	g++-8 ./src/sharpening/seq.cpp `pkg-config --cflags --libs opencv` -o ./bin/sharpen-seq
	g++-8 ./src/sharpening/openmp.cpp `pkg-config --cflags --libs opencv` -o ./bin/sharpen-omp
	g++-8 ./src/sharpening/openmp.cpp `pkg-config --cflags --libs opencv` -o ./bin/sharpen-omp
	nvcc `pkg-config --cflags --libs opencv`  src/contrast/cuda.cu -o ./bin/contrast-cuda
	nvcc `pkg-config --cflags --libs opencv`  src/blur/cuda.cu -o ./bin/blur-cuda
	nvcc `pkg-config --cflags --libs opencv`  src/sharpening/cuda.cu -o ./bin/sharpening-cuda
	
clean:
	cd build && make clean
	rm -rf build
