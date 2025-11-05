#ifndef __DETECTOR_YOLOV5PLATE_H__
#define __DETECTOR_YOLOV5PLATE_H__

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <net.h>

#ifndef YOLOV5_PARAM
#define YOLOV5_PARAM "/home/dell/Code/c++/PlateRecognition_ncnn/yolov5_platedetector/weights/yolov5plate.ncnn.param"
#endif

#ifndef YOLOV5_BIN
#define YOLOV5_BIN "/home/dell/Code/c++/PlateRecognition_ncnn/yolov5_platedetector/weights/yolov5plate.ncnn.bin"
#endif

struct Detection_Yolov5
{
    float bbox[4];  //x1 y1 x2 y2
    float class_confidence;
    int label;
    float landmark[8];
};

class Detector_Yolov5plate
{
public:
    Detector_Yolov5plate();
    ~Detector_Yolov5plate();

    int detect(const cv::Mat& rgb, std::vector<Detection_Yolov5>& objects);
    int detect_batch(const std::vector<cv::Mat>& rgb_batch, std::vector<std::vector<Detection_Yolov5>>& batch_objects);

    int draw(cv::Mat& rgb, const std::vector<Detection_Yolov5>& objects);

private:
    ncnn::Net yolov5_plate;

    const int target_size = 640;
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.5f;
    const bool use_gpu = true;
    const int max_stride = 32;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // __DETECTOR_YOLOV5PLATE_H__