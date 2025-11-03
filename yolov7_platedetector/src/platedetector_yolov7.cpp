#include "platedetector_yolov7.h"
#include "cpu.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

static void qsort_descent_inplace(std::vector<Detection_Yolov7>& objects)
{
    std::sort(objects.begin(), objects.end(),
              [](const Detection_Yolov7& a, const Detection_Yolov7& b)
              { return a.confidence > b.confidence; });
}

static inline float box_iou(float aleft, float atop, float aright, float abottom,
                           float bleft, float btop, float bright, float bbottom)
{
    float cleft = std::max(aleft, bleft);
    float ctop = std::max(atop, btop);
    float cright = std::min(aright, bright);
    float cbottom = std::min(abottom, bbottom);
    
    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;
    
    float a_area = std::max(0.0f, aright - aleft) * std::max(0.0f, abottom - atop);
    float b_area = std::max(0.0f, bright - bleft) * std::max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static void nms_sorted_bboxes(const std::vector<Detection_Yolov7>& objects,
                              std::vector<int>& picked,
                              float nms_threshold)
{
    picked.clear();
    const int n = objects.size();
    if (n == 0) return;

    for (int i = 0; i < n; i++)
    {
        bool keep = true;
        
        for (int j : picked)
        {
            // 同类别才进行NMS
            if (objects[i].label != objects[j].label)
                continue;
            
            // 计算IOU 
            float iou = box_iou(
                objects[i].bbox.xmin, objects[i].bbox.ymin,
                objects[i].bbox.xmax, objects[i].bbox.ymax,
                objects[j].bbox.xmin, objects[j].bbox.ymin,
                objects[j].bbox.xmax, objects[j].bbox.ymax
            );

            if (iou > nms_threshold)
            {
                keep = false;
                break;
            }
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_plate_proposals(const ncnn::Mat& pred,
                                     float prob_threshold,
                                     std::vector<Detection_Yolov7>& objects,
                                     int num_classes,
                                     int ckpt_num)
{
    const int num_bboxes = pred.h;
    for (int i = 0; i < num_bboxes; i++)
    {
        const float* pitem = pred.row(i);
        float objectness = pitem[4];
        if (objectness < prob_threshold) continue;

        const float* class_conf = pitem + 5;
        float confidence = class_conf[0];
        int label = 0;
        for (int c = 1; c < num_classes; c++)
        {
            if (class_conf[c] > confidence)
            {
                confidence = class_conf[c];
                label = c;
            }
        }

        confidence *= objectness;
        if (confidence < prob_threshold) continue;

        float cx = pitem[0], cy = pitem[1], w = pitem[2], h = pitem[3];

        Detection_Yolov7 det;
        det.bbox.xmin = cx - w*0.5f;
        det.bbox.ymin = cy - h*0.5f;
        det.bbox.xmax = cx + w*0.5f;
        det.bbox.ymax = cy + h*0.5f;
        det.confidence = confidence;
        det.label = label;

        det.key_points.resize(ckpt_num * 2);
        const float* landmarks = pitem + 5 + num_classes;
        for (int k = 0; k < ckpt_num; k++)
        {
            det.key_points[2*k]   = landmarks[3*k];
            det.key_points[2*k+1] = landmarks[3*k+1];
        }

        objects.push_back(det);
    }
}


Detector_Yolov7plate::Detector_Yolov7plate()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    yolov7_plate.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolov7_plate.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov7_plate.opt.use_vulkan_compute = use_gpu;
#endif

    yolov7_plate.opt.num_threads = ncnn::get_big_cpu_count();
    yolov7_plate.opt.blob_allocator = &blob_pool_allocator;
    yolov7_plate.opt.workspace_allocator = &workspace_pool_allocator;

    yolov7_plate.load_param(YOLOV7_PARAM);
    yolov7_plate.load_model(YOLOV7_BIN);
}

Detector_Yolov7plate::~Detector_Yolov7plate()
{
    yolov7_plate.clear();
}

int Detector_Yolov7plate::detect(const cv::Mat& rgb, std::vector<Detection_Yolov7>& objects)
{
    objects.clear();

    int orig_w = rgb.cols;
    int orig_h = rgb.rows;

    // 缩放到 target_size
    int w = rgb.cols;
    int h = rgb.rows;
    float scale = target_size / (float)std::max(w, h);
    int resized_w = w * scale;
    int resized_h = h * scale;

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resized_w, resized_h));

    cv::Mat input_mat(target_size, target_size, CV_8UC3, cv::Scalar(114,114,114));
    resized.copyTo(input_mat(cv::Rect(0,0,resized_w,resized_h)));

    // 转 ncnn
    ncnn::Mat in = ncnn::Mat::from_pixels(input_mat.data, ncnn::Mat::PIXEL_RGB, target_size, target_size);
    in.substract_mean_normalize(0, norm_vals);

    // 推理
    ncnn::Extractor ex = yolov7_plate.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    // 生成候选框
    generate_plate_proposals(out, prob_threshold, objects, num_classes, ckpt_num);

    // NMS
    qsort_descent_inplace(objects);
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, nms_threshold);

    std::vector<Detection_Yolov7> nms_objects;
    for (int idx : picked) nms_objects.push_back(objects[idx]);
    objects.swap(nms_objects);

    // 限制在原图范围
    for (auto& det : objects)
    {
        det.bbox.xmin = std::max(0.f, std::min((float)(orig_w-1), det.bbox.xmin/scale));
        det.bbox.ymin = std::max(0.f, std::min((float)(orig_h-1), det.bbox.ymin/scale));
        det.bbox.xmax = std::max(0.f, std::min((float)(orig_w-1), det.bbox.xmax/scale));
        det.bbox.ymax = std::max(0.f, std::min((float)(orig_h-1), det.bbox.ymax/scale));

        for (size_t k = 0; k < det.key_points.size(); k+=2)
        {
            det.key_points[k]   = std::max(0.f, std::min((float)(orig_w-1), det.key_points[k]/scale));
            det.key_points[k+1] = std::max(0.f, std::min((float)(orig_h-1), det.key_points[k+1]/scale));
        }
    }

    return 0;
}

int Detector_Yolov7plate::detect_batch(const std::vector<cv::Mat>& rgb_batch, 
                                       std::vector<std::vector<Detection_Yolov7>>& batch_objects)
{
    batch_objects.clear();
    batch_objects.resize(rgb_batch.size());
    
    if (rgb_batch.empty()) {
        std::cerr << "[Batch Detect] Input batch is empty!" << std::endl;
        return -1;
    }
    
    int batch_size = rgb_batch.size();
    std::cout << "[Batch Detect] Processing " << batch_size << " images with OpenMP parallel..." << std::endl;
    
    // OpenMP并行处理
    #pragma omp parallel for num_threads(4)
    for (int b = 0; b < batch_size; b++)
    {
        if (rgb_batch[b].empty() || rgb_batch[b].data == nullptr) {
            #pragma omp critical
            {
                std::cerr << "[Batch Detect] Image " << b << " is empty, skipping..." << std::endl;
            }
            continue;
        }
        
        // 调用单张检测（线程安全）
        int ret = detect(rgb_batch[b], batch_objects[b]);
        
        #pragma omp critical
        {
            if (ret != 0) {
                std::cerr << "[Batch Detect] Image " << b << " detection failed!" << std::endl;
            } else {
                std::cout << "[Batch Detect] Image " << b << ": " 
                          << batch_objects[b].size() << " objects detected" << std::endl;
            }
        }
    }
    
    std::cout << "[Batch Detect] Batch processing completed!" << std::endl;
    return 0;
}

int Detector_Yolov7plate::draw(cv::Mat& rgb, const std::vector<Detection_Yolov7>& objects)
{
    // 4个关键点对应颜色：左上-黄, 右上-红, 右下-绿, 左下-紫
    static cv::Scalar lm_colors[] = {
        cv::Scalar(255, 255, 0),  // 左上 - 黄色
        cv::Scalar(0, 0, 255),    // 右上 - 红色
        cv::Scalar(0, 255, 0),    // 右下 - 绿色
        cv::Scalar(255, 0, 255)   // 左下 - 紫色
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Detection_Yolov7& obj = objects[i];

        int x0 = static_cast<int>(obj.bbox.xmin);
        int y0 = static_cast<int>(obj.bbox.ymin);
        int x1 = static_cast<int>(obj.bbox.xmax);
        int y1 = static_cast<int>(obj.bbox.ymax);

        // 绘制bbox
        cv::rectangle(rgb, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 0, 0), 2);

        // 绘制4个landmark
        for (size_t k = 0; k + 1 < obj.key_points.size(); k += 2)
        {
            int lm_idx = k / 2; // 第几个关键点
            if (lm_idx >= 4) break; // 防止越界
            int lm_x = static_cast<int>(obj.key_points[k]);
            int lm_y = static_cast<int>(obj.key_points[k + 1]);
            cv::circle(rgb, cv::Point(lm_x, lm_y), 4, lm_colors[lm_idx], -1);
        }

        // 绘制标签 + 置信度
        char text[64];
        sprintf(text, "%d:%.2f", obj.label, obj.confidence);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int tx = x0;
        int ty = y0 - label_size.height - baseLine;
        if (ty < 0) ty = 0;
        if (tx + label_size.width > rgb.cols) tx = rgb.cols - label_size.width;

        // 标签背景
        cv::rectangle(rgb, cv::Rect(cv::Point(tx, ty), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        // 绘制文字
        cv::putText(rgb, text, cv::Point(tx, ty + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }

    return 0;
}
