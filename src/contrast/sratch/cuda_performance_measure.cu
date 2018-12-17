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
#include <libgen.h>
#include <math.h>

#include "./utils/utils.h"
#include "./utils/cude_contrast_ops.h"
#include "./utils/seq_contrast_ops.h"

#include "../lib/dirreader.h"


using namespace cv;
using namespace std;


int main(int argc, char **argv) {
    int contrast = 127;

    vector<string> images;
    read_directory("./images/", images);

    double sequence_time = 0;
    double parallel_time = 0;

    for(int i = 0; i < images.size(); i++) {

        string inputFile = images[i];
        string input_filename = basename(strdup(inputFile.c_str()));
        cout << "Processing Image: " << input_filename << endl;
        Mat original_image = imread(inputFile, CV_LOAD_IMAGE_COLOR);
        Mat seq_image = original_image.clone();
        
        const clock_t begin_time = clock();
        Mat contrast_image_cuda = contrast_cuda(original_image, contrast);            
        const clock_t end_time = clock();
        parallel_time += float( end_time - begin_time ) /  CLOCKS_PER_SEC;
        cout << "Cuda   : "<< float( end_time - begin_time ) /  CLOCKS_PER_SEC << endl;

        const clock_t begin_time_seq = clock();
        Mat contrast_image_seq = contrast_sequantial(seq_image, contrast);        
        const clock_t end_time_seq = clock();
        sequence_time += float( end_time_seq - begin_time_seq ) /  CLOCKS_PER_SEC;
        cout << "Seq    : "<< float( end_time_seq - begin_time_seq ) /  CLOCKS_PER_SEC << endl;
        
        string output_filename = "./output/contrast_seq_" + input_filename;
        imwrite(output_filename, original_image);

        output_filename = "./output/contrast_cuda_" + input_filename;
        imwrite(output_filename, contrast_image_cuda);
    }

    cout << "Total Sequenctial Time: " << sequence_time << endl;
    cout << "Total Cuda        Time: " << parallel_time << endl;

    return 0;
}
