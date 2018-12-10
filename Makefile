cmake:
	mkdir -p build
	cd build && cmake -DCMAKE_CXX_COMPILER=g++-8 ..

compile:
	mkdir -p bin
	cd build && make all

clean:
	cd build && make clean
