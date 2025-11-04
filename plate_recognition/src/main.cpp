#include <iostream>
#include "plate_recognition.h"
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

// 用于读取指定文件夹下特定类型文件
void readFileList(const std::string& path, std::vector<std::string>& fileList, const std::vector<std::string>& extensions)
{
    fileList.clear();
    for (const auto& entry : fs::directory_iterator(path))
    {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        for (auto& e : extensions)
        {
            if (ext == "." + e)
            {
                fileList.push_back(entry.path().string());
                break;
            }
        }
    }
}

int main()
{
    PlateRecognition plate_recognition;
    
    int string_index = 0;
    double time_count = 0.0;
    
    std::string imagepath = "../data";
    std::vector<std::string> imagList;
    std::vector<std::string> fileType{"jpg", "png"};
    
    readFileList(imagepath, imagList, fileType);

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
            std::cout << "Image: " << imagList[i] 
                      << " - plate: " << plate_num 
                      << " plate_color: " << plate_color << std::endl;
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