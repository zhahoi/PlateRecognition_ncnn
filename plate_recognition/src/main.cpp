#include <iostream>
#include "plate_recognition.h"
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <dirent.h>

std::string getHouZhui(std::string fileName)
{
    int pos = fileName.find_last_of(std::string("."));
    std::string houZui = fileName.substr(pos + 1);
    return houZui;
}

int readFileList(char *basePath, std::vector<std::string> &fileList, std::vector<std::string> fileType)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir = opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8)
        {   ///file
            if (fileType.size())
            {
                std::string houZui = getHouZhui(std::string(ptr->d_name));
                for (auto &s : fileType)
                {
                    if (houZui == s)
                    {
                        fileList.push_back(std::string(basePath) + "/" + std::string(ptr->d_name));
                        break;
                    }
                }
            }
            else
            {
                fileList.push_back(std::string(basePath) + "/" + std::string(ptr->d_name));
            }
        }
        else if (ptr->d_type == 10)    ///link file
            printf("d_name:%s/%s\n", basePath, ptr->d_name);
        else if (ptr->d_type == 4)    ///dir
        {
            memset(base, '\0', sizeof(base));
            strcpy(base, basePath);
            strcat(base, "/");
            strcat(base, ptr->d_name);
            readFileList(base, fileList, fileType);
        }
    }
    closedir(dir);
    return 1;
}

int main()
{
    PlateRecognition plate_recognition;
    
    int string_index = 0;
    double time_count = 0.0;
    
    std::string imagepath = "/home/dell/Code/c++/PlateRecognition_detect/data";
    std::vector<std::string> imagList;
    std::vector<std::string> fileType{"jpg", "png"};
    
    readFileList(const_cast<char *>(imagepath.c_str()), imagList, fileType);

    std::cout << "Found " << imagList.size() << " images to process." << std::endl;

    for (size_t i = 0; i < imagList.size(); i++)
    {
        cv::Mat plate_img = cv::imread(imagList[i]);
        if (plate_img.empty()) continue;
        
        string_index++;
        std::string plate_num;
        std::string plate_color;
        
        auto start = std::chrono::high_resolution_clock::now();
        int ret = plate_recognition.recognize(plate_img, plate_num, plate_color);
        auto end = std::chrono::high_resolution_clock::now();
        time_count += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        if (ret == 0) {
            // 关键：强制刷新输出缓冲区
            std::cout.flush();
            std::cout << "Image: " << imagList[i] << " - plate: " << plate_num << " plate_color: " << plate_color << std::endl;
            std::cout.flush(); // 再次刷新确保输出
        }
    }
    
    if (string_index > 0) {
        double average_time = time_count / string_index;
        std::cout << "\n=== Processing Summary ===" << std::endl;
        std::cout << "Total images processed: " << string_index << std::endl;
        std::cout << "Average time: " << average_time << " us" << std::endl;
        std::cout << "Average time: " << average_time / 1000.0 << " ms" << std::endl;
    }
    
    return 0;
}