#ifndef __PLATE_RECOGNITION_H__
#define __PLATE_RECOGNITION_H__

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <net.h>
#include <vector>
#include <algorithm>
#include <iostream>

// 使用CMake传递的路径
#ifndef PLATE_RECOGNITION_PARAM
#define PLATE_RECOGNITION_PARAM "/home/dell/Code/c++/PlateRecognition_ncnn/plate_recognition/weights/plate_recognition_color.ncnn.param"
#endif

#ifndef PLATE_RECOGNITION_BIN
#define PLATE_RECOGNITION_BIN "/home/dell/Code/c++/PlateRecognition_ncnn/plate_recognition/weights/plate_recognition_color.ncnn.bin"
#endif

extern std::vector<std::string> plate_color_list;
extern std::string plate_chr[78];

class PlateRecognition
{
public:
    PlateRecognition();
    ~PlateRecognition();

    int recognize(const cv::Mat& img, std::string& plate_str, std::string& plate_color);

private:
    ncnn::Net plate_recognition;

    const float mean_vals[3] = {149.94f, 149.94f, 149.94f};
    const float norm_vals[3] = {0.020746888f, 0.020746888f, 0.020746888f}; // 1/(0.193*255)

    const int INPUT_W = 168;
    const int INPUT_H = 48;
    const int POS = 21;
    const int CLS = 78;
    const bool use_gpu = true;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // __PLATE_RECOGNITION_H__