//
// Created by Aartee Kasliwal on 2018-12-15.
//

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <unistd.h>
#include <ctime>
#include <libgen.h>
#include <math.h>
#include "seq_blur_ops.h"

#include "../lib/dirreader.h"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    vector<string> images;
    //read_directory("./dataset/jpg/", images);
	read_directory("./images/", images);

    double sequence_time = 0;
    double parallel_time = 0;

    for(int i = 0; i < images.size(); i++) {

        string inputFile = images[i];
        
        parallel_time += blur_image(inputFile, "openmp", false);
        //sequence_time += blur_image(inputFile, "sequential", false);
    }

    //cout << "\n\nTotal Sequenctial Time: " << sequence_time << endl;
    cout << "Total Parallel    Time: " << parallel_time << endl;

    return 0;
}
