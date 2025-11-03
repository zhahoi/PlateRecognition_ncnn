#ifndef __PLATE_DETECTION_RECOGNITION_H__
#define __PLATE_DETECTION_RECOGNITION_H__

#include <iostream>
#include <memory>  // 添加智能指针头文件
#include <opencv2/opencv.hpp>

#include "plate_recognition.h"
#include "platedetector_yolov5.h"
#include "platedetector_yolov7.h"

typedef struct 
{
    PlateRect bbox;                   // bbox
    int label;                        // 0->单层车牌 1->双层车牌
    float confidence;                 // 检测置信度
    float key_points[8];              // 关键点坐标
    char plate_color[32];             // 车牌颜色（固定长度，避免指针）
    char plate_license[32];           // 车牌号（固定长度，避免指针）
    int plate_index;                  // 表示一张图像里面车牌数量        
} PlateDet;

// 传入图像的数据结构
typedef struct
{
    unsigned char* image;      
    int width;
    int height;
    int channels;              
} Plate_ImageData;

class PlateAlgorithm
{
public:
    PlateAlgorithm();
    ~PlateAlgorithm();
    
    // 禁止拷贝和赋值（因为使用了unique_ptr）
    PlateAlgorithm(const PlateAlgorithm&) = delete;
    PlateAlgorithm& operator=(const PlateAlgorithm&) = delete;
    
    // 支持移动语义
    PlateAlgorithm(PlateAlgorithm&&) = default;
    PlateAlgorithm& operator=(PlateAlgorithm&&) = default;

    int PlateRecognition_yolov5(Plate_ImageData* img, PlateDet* PlateDets);
    int PlateRecognition_yolov7(Plate_ImageData* img, PlateDet* PlateDets);

private:
    // 使用 unique_ptr 管理检测器和识别器
    std::unique_ptr<Detector_Yolov5plate> yolov5_plate;
    std::unique_ptr<Detector_Yolov7plate> yolov7_plate;
    std::unique_ptr<PlateRecognition> plate_recognition;

private:
    cv::Mat get_split_merge(cv::Mat &img);   // 双层车牌 分割 拼接
    cv::Mat getTransForm(cv::Mat &src_img, cv::Point2f order_rect[4]); // 透视变换
    float getNorm2(float x, float y);
};

#endif // __PLATE_DETECTION_RECOGNITION_H__
