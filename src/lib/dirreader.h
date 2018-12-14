#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>

using namespace std;

void read_directory(const std::string& name, vector<string> &image_names)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        if(strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0)
            continue;
        string image_path = name + dp->d_name;
        image_names.push_back(image_path);
    }
    closedir(dirp);
}

/*
int main(int argc, char **argv) {
    vector<string> images;
    read_directory("./images/", images);
    for(int i=0; i<images.size(); i++) {
        std::cout << images[i] << std::endl;
    }
}
*/
