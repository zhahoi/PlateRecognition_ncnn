#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/freetype.hpp>

#include "plate_detection_recognition.h"

// ========== 配置选项 ==========
#define USE_YOLOV7      0       // 1: 使用YOLOv7, 0: 使用YOLOv5
#define PLATE_MAX       10      // 每张图片最大车牌数量

// ========== 工具函数 ==========
int readFileList(const std::string& path, std::vector<std::string>& fileList, 
                 const std::vector<std::string>& extensions)
{
    fileList.clear();
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "Error: Cannot open directory: " << path << std::endl;
        return -1;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            std::string filename = entry->d_name;
            size_t dotPos = filename.find_last_of(".");
            if (dotPos != std::string::npos) {
                std::string ext = filename.substr(dotPos + 1);
                for (const auto& e : extensions) {
                    if (ext == e) {
                        fileList.push_back(path + "/" + filename);
                        break;
                    }
                }
            }
        }
    }
    closedir(dir);
    return 0;
}

void drawBboxes(cv::Mat& img, const PlateDet* PlateDets, cv::Ptr<cv::freetype::FreeType2>& ft2)
{
    if (PlateDets[0].plate_index <= 0) return;

    static cv::Scalar kp_colors[4] = {
        cv::Scalar(255, 255, 0),  // 左上 - 黄
        cv::Scalar(0, 0, 255),    // 右上 - 红
        cv::Scalar(0, 255, 0),    // 右下 - 绿
        cv::Scalar(255, 0, 255)   // 左下 - 紫
    };

    for (int f = 0; f < PlateDets[0].plate_index && f < PLATE_MAX; f++)
    {
        std::string plate_str(PlateDets[f].plate_license);
        std::string plate_color(PlateDets[f].plate_color);
        std::string label = plate_str + " " + plate_color;

        int xmin = static_cast<int>(PlateDets[f].bbox.xmin);
        int ymin = static_cast<int>(PlateDets[f].bbox.ymin);
        int xmax = static_cast<int>(PlateDets[f].bbox.xmax);
        int ymax = static_cast<int>(PlateDets[f].bbox.ymax);

        // 绘制边界框
        cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), 
                     cv::Scalar(255, 0, 0), 2);

        // 绘制关键点
        for (int k = 0; k < 4; k++) {
            int x = static_cast<int>(PlateDets[f].key_points[2 * k]);
            int y = static_cast<int>(PlateDets[f].key_points[2 * k + 1]);
            cv::circle(img, cv::Point(x, y), 4, kp_colors[k], -1);
        }

        // 绘制标签
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(ymin, labelSize.height + 10);
        
        cv::rectangle(img, 
                     cv::Point(xmin, top - labelSize.height - 10),
                     cv::Point(xmin + labelSize.width + 10, top),
                     cv::Scalar(255, 255, 255), cv::FILLED);

        if (ft2) {
            ft2->putText(img, label, cv::Point(xmin + 5, top - 5), 
                        20, cv::Scalar(0, 0, 0), -1, 8, true);
        } else {
            cv::putText(img, label, cv::Point(xmin + 5, top - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        // 置信度
        char conf_text[32];
        snprintf(conf_text, sizeof(conf_text), "%.2f", PlateDets[f].confidence);
        cv::putText(img, conf_text, cv::Point(xmin, ymax + 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

void initPlateDet(PlateDet* PlateDets)
{
    for (int i = 0; i < PLATE_MAX; i++)
    {
        PlateDets[i].bbox.xmin = 0;
        PlateDets[i].bbox.ymin = 0;
        PlateDets[i].bbox.xmax = 0;
        PlateDets[i].bbox.ymax = 0;
        PlateDets[i].confidence = 0.0f;
        PlateDets[i].label = -1;
        PlateDets[i].plate_index = 0;
        
        memset(PlateDets[i].key_points, 0, sizeof(PlateDets[i].key_points));
        memset(PlateDets[i].plate_license, 0, sizeof(PlateDets[i].plate_license));
        memset(PlateDets[i].plate_color, 0, sizeof(PlateDets[i].plate_color));
    }
}

// ========== 主函数 ==========
int main(int argc, char* argv[])
{
    std::cout << "========================================" << std::endl;
    std::cout << "  Plate Detection & Recognition System  " << std::endl;
    std::cout << "========================================" << std::endl;

    // 配置路径
    std::string image_dir = "../data";
    std::string output_dir = "../result";
    std::string font_path = "../font/NotoSansCJK-Regular.otf";
    
    if (argc > 1) image_dir = argv[1];
    if (argc > 2) output_dir = argv[2];
    if (argc > 3) font_path = argv[3];

    mkdir(output_dir.c_str(), 0755);

    // 初始化算法
    std::cout << "\nInitializing..." << std::endl;
    PlateAlgorithm plate_algo;

    // 加载字体
    cv::Ptr<cv::freetype::FreeType2> ft2;
    try {
        ft2 = cv::freetype::createFreeType2();
        ft2->loadFontData(font_path, 0);
        std::cout << "Font loaded: " << font_path << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "Warning: Cannot load font" << std::endl;
        ft2 = nullptr;
    }

    // 读取图像列表
    std::vector<std::string> imageList;
    std::vector<std::string> fileTypes{"jpg", "png", "jpeg", "JPG", "PNG", "JPEG"};
    
    if (readFileList(image_dir, imageList, fileTypes) != 0 || imageList.empty()) {
        std::cerr << "Error: No images found in " << image_dir << std::endl;
        return -1;
    }

    std::cout << "Found " << imageList.size() << " images" << std::endl;
    std::cout << "Using detector: " << (USE_YOLOV7 ? "YOLOv7" : "YOLOv5") << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 处理图像
    PlateDet PlateDets[PLATE_MAX];
    Plate_ImageData image_data;
    
    int total_plates = 0;
    double total_time = 0.0;
    int processed_images = 0;

    for (size_t i = 0; i < imageList.size(); i++)
    {
        cv::Mat img = cv::imread(imageList[i]);
        if (img.empty()) {
            std::cerr << "Warning: Cannot read " << imageList[i] << std::endl;
            continue;
        }

        // 准备输入
        image_data.image = img.data;
        image_data.width = img.cols;
        image_data.height = img.rows;
        image_data.channels = img.channels();

        initPlateDet(PlateDets);

        // 检测识别
        auto start = std::chrono::high_resolution_clock::now();
        
        int ret = -1;
#if USE_YOLOV7
        ret = plate_algo.PlateRecognition_yolov7(&image_data, PlateDets);
#else
        ret = plate_algo.PlateRecognition_yolov5(&image_data, PlateDets);
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if (ret != 0) {
            std::cerr << "Warning: Processing failed for " << imageList[i] << std::endl;
            continue;
        }

        processed_images++;
        total_time += elapsed;

        int num_plates = PlateDets[0].plate_index;
        total_plates += num_plates;

        // 输出结果（保留关键信息）
        std::cout << "[" << processed_images << "/" << imageList.size() << "] " 
                  << imageList[i] << std::endl;
        std::cout << "  Time: " << elapsed / 1000.0 << " ms | Plates: " << num_plates << std::endl;

        for (int j = 0; j < num_plates && j < PLATE_MAX; j++) {
            std::cout << "  [" << j + 1 << "] " 
                      << PlateDets[j].plate_license << " "
                      << PlateDets[j].plate_color
                      << " (conf: " << std::fixed << std::setprecision(2) 
                      << PlateDets[j].confidence << ")" << std::endl;
        }

        // 绘制并保存
        drawBboxes(img, PlateDets, ft2);

        size_t lastPos = imageList[i].find_last_of("/");
        std::string image_name = (lastPos == std::string::npos) ? 
                                imageList[i] : imageList[i].substr(lastPos + 1);
        std::string output_path = output_dir + "/" + image_name;
        
        cv::imwrite(output_path, img);
    }

    // 统计信息
    std::cout << "\n========================================" << std::endl;
    std::cout << "         Processing Summary             " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total images:      " << processed_images << std::endl;
    std::cout << "Total plates:      " << total_plates << std::endl;
    
    if (processed_images > 0) {
        std::cout << "Avg plates/image:  " << std::fixed << std::setprecision(2)
                  << static_cast<float>(total_plates) / processed_images << std::endl;
        std::cout << "Avg time/image:    " << std::fixed << std::setprecision(1)
                  << total_time / processed_images / 1000.0 << " ms" << std::endl;
        std::cout << "FPS:               " << std::fixed << std::setprecision(2)
                  << processed_images / (total_time / 1000000.0) << std::endl;
    }
    
    std::cout << "Total time:        " << std::fixed << std::setprecision(1)
              << total_time / 1000.0 << " ms" << std::endl;
    std::cout << "Results saved to:  " << output_dir << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
