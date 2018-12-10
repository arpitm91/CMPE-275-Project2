//
// Created by Arpit Mathur on 2018-12-09.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int SHARPEN_FACTOR[9] = {
        -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1
};

int my_sharpen(Vec3b color[], int index) {
    int result = 0;

    for (int i = 0; i < 9; i++) {
        result += SHARPEN_FACTOR[i] * int(color[i][index]);
    }

    if (result < 0)
        return  0;
    if (result > 255)
        return  255;
    return result;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
        return -1;
    }

    Mat original_image, concatenated_image;
    original_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if (!original_image.data)                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    namedWindow("Display window", WINDOW_NORMAL);// Create a window for display.

    Vec3b color[9];
    int i, j, newBlue, newGreen, newRed, n = original_image.cols, m = original_image.rows;
    int factor = 2;

    Mat sharpened_image(m, n, CV_8UC3, Scalar(255, 255, 255));
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            if (i <= 0|| i >= n - 1 || j <= 0 || j >= m - 1) {
                sharpened_image.at<Vec3b>(Point(i, j)) = original_image.at<Vec3b>(Point(i, j));
                continue;
            }
            color[0] = original_image.at<Vec3b>(Point(i - 1, j - 1));
            color[1] = original_image.at<Vec3b>(Point(i - 1, j));
            color[2] = original_image.at<Vec3b>(Point(i - 1, j + 1));
            color[3] = original_image.at<Vec3b>(Point(i, j - 1));
            color[4] = original_image.at<Vec3b>(Point(i, j));
            color[5] = original_image.at<Vec3b>(Point(i, j + 1));
            color[6] = original_image.at<Vec3b>(Point(i + 1, j - 1));
            color[7] = original_image.at<Vec3b>(Point(i + 1, j));
            color[8] = original_image.at<Vec3b>(Point(i + 1, j + 1));
            newBlue = my_sharpen(color, 0);
            newGreen = my_sharpen(color, 1);
            newRed = my_sharpen(color, 2);

            sharpened_image.at<Vec3b>(Point(i, j)) = Vec3b(newBlue, newGreen, newRed);

        }
    }

    hconcat(original_image, sharpened_image, concatenated_image);
    imshow("Display window", concatenated_image);                   // Show our image inside it.

//    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
