#ifndef __DETECTOR_YOLOV7PLATE_H__
#define __DETECTOR_YOLOV7PLATE_H__

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <net.h>
#include <vector>
#include <algorithm>
#include <iostream>

#ifndef YOLOV7_PARAM
#define YOLOV7_PARAM "/home/dell/Code/c++/PlateRecognition_ncnn/yolov7_platedetector/weights/yolov7plate.ncnn.param"
#endif

#ifndef YOLOV7_BIN
#define YOLOV7_BIN "/home/dell/Code/c++/PlateRecognition_ncnn/yolov7_platedetector/weights/yolov7plate.ncnn.bin"
#endif

typedef struct 
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
} PlateRect;

struct Detection_Yolov7
{
    PlateRect bbox;
    int label;
    float confidence;
    std::vector<float> key_points;    // 关键点 (8个值: x,y × 4)
};

class Detector_Yolov7plate
{
public:
    Detector_Yolov7plate();
    ~Detector_Yolov7plate();

    int detect(const cv::Mat& rgb, std::vector<Detection_Yolov7>& objects);
    int detect_batch(const std::vector<cv::Mat>& rgb_batch, std::vector<std::vector<Detection_Yolov7>>& batch_objects);

    int draw(cv::Mat& rgb, const std::vector<Detection_Yolov7>& objects);

private:
    ncnn::Net yolov7_plate;

    const float norm_vals[3] = {1.f/255.f, 1.f/255.f, 1.f/255.f};
    const int target_size = 640;
    const int num_classes = 2;         // 单层车牌和双层车牌
    const int ckpt_num = 4;            // 4个关键点
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.5f;
    const bool use_gpu = true;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // __DETECTOR_YOLOV7PLATE_H__