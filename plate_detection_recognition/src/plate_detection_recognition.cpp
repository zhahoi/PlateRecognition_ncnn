#include "plate_detection_recognition.h"

PlateAlgorithm::PlateAlgorithm()
{
    try {
        yolov5_plate = std::make_unique<Detector_Yolov5plate>();
        std::cout << "[PlateAlgorithm] YOLOv5 detector initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[PlateAlgorithm] Failed to initialize YOLOv5: " << e.what() << std::endl;
        throw;
    }
    
    try {
        yolov7_plate = std::make_unique<Detector_Yolov7plate>();
        std::cout << "[PlateAlgorithm] YOLOv7 detector initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[PlateAlgorithm] Failed to initialize YOLOv7: " << e.what() << std::endl;
        throw;
    }
    
    try {
        plate_recognition = std::make_unique<PlateRecognition>();
        std::cout << "[PlateAlgorithm] Plate recognition initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[PlateAlgorithm] Failed to initialize recognition: " << e.what() << std::endl;
        throw;
    }
}

PlateAlgorithm::~PlateAlgorithm()
{
    std::cout << "[PlateAlgorithm] Destructing..." << std::endl;
}

int PlateAlgorithm::PlateRecognition_yolov5(Plate_ImageData* img, PlateDet* PlateDets)
{
    if (img == nullptr || img->image == nullptr)
    {
        std::cerr << "[YOLOv5] Image is null" << std::endl;
        return -1;
    }
    
    if (!yolov5_plate || !plate_recognition)
    {
        std::cerr << "[YOLOv5] Detector or recognizer not initialized" << std::endl;
        return -1;
    }

    cv::Mat image_temp(img->height, img->width, CV_8UC3, img->image);
    if (image_temp.empty())
    {
        std::cerr << "[YOLOv5] Image is empty" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> ImgVec;
    std::vector<std::vector<Detection_Yolov5>> dets;
    ImgVec.push_back(image_temp);
    
    yolov5_plate->detect_batch(ImgVec, dets);

    for (size_t i = 0; i < dets.size(); i++)
    {
        for (size_t j = 0; j < dets[i].size(); j++)
        {
            cv::Point2f order_rect[4];
            for (int k = 0; k < 4; k++)
            {
                order_rect[k] = cv::Point(dets[i][j].landmark[2 * k], 
                                         dets[i][j].landmark[2 * k + 1]);
                PlateDets[j].key_points[2 * k] = dets[i][j].landmark[2 * k];
                PlateDets[j].key_points[2 * k + 1] = dets[i][j].landmark[2 * k + 1];
            }
            
            cv::Mat roiImg = getTransForm(image_temp, order_rect);
            int label = dets[i][j].label;
            if (label)  // 双层车牌
            {
                roiImg = get_split_merge(roiImg);
            }
            
            std::string plate_str;
            std::string plate_color;
            plate_recognition->recognize(roiImg, plate_str, plate_color);
            
            // 使用成员变量而不是数组索引
            PlateDets[j].bbox.xmin = dets[i][j].bbox[0];  // 改为成员变量
            PlateDets[j].bbox.ymin = dets[i][j].bbox[1];
            PlateDets[j].bbox.xmax = dets[i][j].bbox[2];
            PlateDets[j].bbox.ymax = dets[i][j].bbox[3];
            PlateDets[j].confidence = dets[i][j].class_confidence;
            PlateDets[j].label = dets[i][j].label;
            
            // 安全的字符串拷贝
            strncpy(PlateDets[j].plate_license, plate_str.c_str(), 31);
            PlateDets[j].plate_license[31] = '\0';
            strncpy(PlateDets[j].plate_color, plate_color.c_str(), 31);
            PlateDets[j].plate_color[31] = '\0';
            
            PlateDets[j].plate_index = dets[i].size();
        }
    }

    return 0;
}

int PlateAlgorithm::PlateRecognition_yolov7(Plate_ImageData* img, PlateDet* PlateDets)
{
    if (img == nullptr || img->image == nullptr)
    {
        std::cerr << "[YOLOv7] Image is null" << std::endl;
        return -1;
    }
    
    if (!yolov7_plate || !plate_recognition)
    {
        std::cerr << "[YOLOv7] Detector or recognizer not initialized" << std::endl;
        return -1;
    }

    cv::Mat image_temp(img->height, img->width, CV_8UC3, img->image);
    if (image_temp.empty())
    {
        std::cerr << "[YOLOv7] Image is empty" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> ImgVec;
    std::vector<std::vector<Detection_Yolov7>> dets;
    ImgVec.push_back(image_temp);
    
    yolov7_plate->detect_batch(ImgVec, dets);

    for (size_t i = 0; i < dets.size(); i++)
    {
        for (size_t j = 0; j < dets[i].size(); j++)
        {
            cv::Point2f order_rect[4];
            for (int k = 0; k < 4; k++)
            {
                order_rect[k] = cv::Point(dets[i][j].key_points[2 * k], 
                                         dets[i][j].key_points[2 * k + 1]);
                PlateDets[j].key_points[2 * k] = dets[i][j].key_points[2 * k];
                PlateDets[j].key_points[2 * k + 1] = dets[i][j].key_points[2 * k + 1];
            }
            
            cv::Mat roiImg = getTransForm(image_temp, order_rect);
            int label = dets[i][j].label;
            if (label)  // 双层车牌
            {
                roiImg = get_split_merge(roiImg);
            }
            
            std::string plate_str;
            std::string plate_color;
            plate_recognition->recognize(roiImg, plate_str, plate_color);
            
            // YOLOv7 部分可能已经是正确的成员变量访问方式
            PlateDets[j].bbox.xmin = dets[i][j].bbox.xmin;  // 保持原样
            PlateDets[j].bbox.ymin = dets[i][j].bbox.ymin;
            PlateDets[j].bbox.xmax = dets[i][j].bbox.xmax;
            PlateDets[j].bbox.ymax = dets[i][j].bbox.ymax;
            PlateDets[j].confidence = dets[i][j].confidence;
            PlateDets[j].label = dets[i][j].label;
            
            // 安全的字符串拷贝
            strncpy(PlateDets[j].plate_license, plate_str.c_str(), 31);
            PlateDets[j].plate_license[31] = '\0';
            strncpy(PlateDets[j].plate_color, plate_color.c_str(), 31);
            PlateDets[j].plate_color[31] = '\0';
            
            PlateDets[j].plate_index = dets[i].size();
        }
    }

    return 0;
}

float PlateAlgorithm::getNorm2(float x, float y)
{
    return sqrt(x * x + y * y);
}

cv::Mat PlateAlgorithm::getTransForm(cv::Mat &src_img, cv::Point2f order_rect[4])
{
    cv::Point2f w1 = order_rect[0] - order_rect[1];
    cv::Point2f w2 = order_rect[2] - order_rect[3];
    auto width1 = getNorm2(w1.x, w1.y);
    auto width2 = getNorm2(w2.x, w2.y);
    auto maxWidth = std::max(width1, width2);

    cv::Point2f h1 = order_rect[0] - order_rect[3];
    cv::Point2f h2 = order_rect[1] - order_rect[2];
    auto height1 = getNorm2(h1.x, h1.y);
    auto height2 = getNorm2(h2.x, h2.y);
    auto maxHeight = std::max(height1, height2);

    std::vector<cv::Point2f> pts_ori(4);
    std::vector<cv::Point2f> pts_std(4);

    pts_ori[0] = order_rect[0];
    pts_ori[1] = order_rect[1];
    pts_ori[2] = order_rect[2];
    pts_ori[3] = order_rect[3];

    pts_std[0] = cv::Point2f(0, 0);
    pts_std[1] = cv::Point2f(maxWidth, 0);
    pts_std[2] = cv::Point2f(maxWidth, maxHeight);
    pts_std[3] = cv::Point2f(0, maxHeight);

    cv::Mat M = cv::getPerspectiveTransform(pts_ori, pts_std);
    cv::Mat dstimg;
    cv::warpPerspective(src_img, dstimg, M, cv::Size(maxWidth, maxHeight));

    return dstimg;
}

cv::Mat PlateAlgorithm::get_split_merge(cv::Mat &img)
{
    cv::Rect upper_rect_area = cv::Rect(0, 0, img.cols, int(5.0 / 12 * img.rows));
    cv::Rect lower_rect_area = cv::Rect(0, int(1.0 / 3 * img.rows), img.cols, 
                                        img.rows - int(1.0 / 3 * img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower = img(lower_rect_area);
    cv::resize(img_upper, img_upper, img_lower.size());
    cv::Mat out(img_lower.rows, img_lower.cols + img_upper.cols, CV_8UC3, 
               cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0, 0, img_upper.cols, img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols, 0, img_lower.cols, img_lower.rows)));

    return out;
}