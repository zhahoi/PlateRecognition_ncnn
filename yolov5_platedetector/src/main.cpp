#include "platedetector_yolov5.h"
#include <iostream>
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
    std::string imageFolder = "/home/dell/Code/c++/PlateRecognition_ncnn/data";
    std::string outputFolder = "/home/dell/Code/c++/PlateRecognition_ncnn/result";
    std::vector<std::string> fileTypes{"jpg","png"};
    std::vector<std::string> imageList;

    readFileList(imageFolder, imageList, fileTypes);
    if (imageList.empty())
    {
        std::cerr << "No images found in folder: " << imageFolder << std::endl;
        return -1;
    }

    // 初始化 YOLOv5_plate detector
    Detector_Yolov5plate yolov5_plate;

    for (const auto& imgPath : imageList)
    {
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) continue;

        std::vector<Detection_Yolov5> objects;
        auto start = std::chrono::high_resolution_clock::now();
        yolov5_plate.detect(img, objects);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << imgPath << " inference time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " us" << std::endl;

        // 使用已有 draw 函数绘制 bbox 和 landmark
        yolov5_plate.draw(img, objects);

        // 保存结果
        std::string imageName = fs::path(imgPath).filename().string();
        std::string savePath = outputFolder + "/" + imageName;
        cv::imwrite(savePath, img);
        std::cout << "Saved result to: " << savePath << std::endl;

        // 可视化（可选）
        // cv::imshow("YOLOv5_plate", img);
        // cv::waitKey(1);
    }

    return 0;
}
