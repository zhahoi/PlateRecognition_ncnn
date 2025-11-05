#include "platedetector_yolov5.h"
#include "cpu.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

static void qsort_descent_inplace(std::vector<Detection_Yolov5>& objects)
{
    std::sort(objects.begin(), objects.end(),
              [](const Detection_Yolov5& a, const Detection_Yolov5& b)
              { return a.class_confidence > b.class_confidence; });
}

static void nms_sorted_bboxes(const std::vector<Detection_Yolov5>& objects,
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
            // 计算IOU 
            // bbox格式: [cx, cy, w, h]
            float lbox[4] = {objects[i].bbox[0], objects[i].bbox[1], objects[i].bbox[2], objects[i].bbox[3]};
            float rbox[4] = {objects[j].bbox[0], objects[j].bbox[1], objects[j].bbox[2], objects[j].bbox[3]};
            
            float interBox[4] = {
                std::max(lbox[0] - lbox[2]/2, rbox[0] - rbox[2]/2), // left
                std::min(lbox[0] + lbox[2]/2, rbox[0] + rbox[2]/2), // right
                std::max(lbox[1] - lbox[3]/2, rbox[1] - rbox[3]/2), // top
                std::min(lbox[1] + lbox[3]/2, rbox[1] + rbox[3]/2)  // bottom
            };
            
            if (interBox[2] > interBox[3] || interBox[0] > interBox[1]) {
                continue;
            }
            
            float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
            float iou = interBoxS / ((lbox[2]) * (lbox[3]) + (rbox[2]) * (rbox[3]) - interBoxS + 0.000001f);

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
                                     std::vector<Detection_Yolov5>& objects)
{
    const int OUTPUT_TENSOR = pred.h;         // 25200
    const int OUTPUT_CANDIDATES = pred.w;     // 15
    
    for (int i = 0; i < OUTPUT_TENSOR; i++)
    {
        const float* ptr = pred.row(i);
        
        int class_num = OUTPUT_CANDIDATES - 13;  // 15 - 13 = 2
        float max_conf = 0.0f;
        int class_idx = -1;
        
        // 计算最大置信度: obj_conf * cls_conf
        float obj_conf = ptr[4];
        for (int j = 0; j < class_num; j++)
        {
            // conf = obj_conf * cls_conf 
            float conf_temp = obj_conf * ptr[j + 13];
            if (max_conf < conf_temp)
            {
                max_conf = conf_temp;
                class_idx = j;
            }
        }
        
        // 置信度过滤
        if (max_conf < prob_threshold)
            continue;
        
        Detection_Yolov5 det;
        
        // bbox: [cx, cy, w, h] 
        det.bbox[0] = ptr[0];  // cx
        det.bbox[1] = ptr[1];  // cy
        det.bbox[2] = ptr[2];  // w
        det.bbox[3] = ptr[3];  // h
        
        det.class_confidence = max_conf;
        det.label = class_idx;
        
        // landmark: 8个值
        for (int k = 0; k < 8; k++) {
            det.landmark[k] = ptr[5 + k];
        }
        
        objects.push_back(det);
    }
}

Detector_Yolov5plate::Detector_Yolov5plate() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    yolov5_plate.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolov5_plate.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov5_plate.opt.use_vulkan_compute = use_gpu;
#endif

    yolov5_plate.opt.num_threads = ncnn::get_big_cpu_count();
    yolov5_plate.opt.blob_allocator = &blob_pool_allocator;
    yolov5_plate.opt.workspace_allocator = &workspace_pool_allocator;

    yolov5_plate.load_param(YOLOV5_PARAM);
    yolov5_plate.load_model(YOLOV5_BIN);
}

Detector_Yolov5plate::~Detector_Yolov5plate()
{
    yolov5_plate.clear();
}

int Detector_Yolov5plate::detect(const cv::Mat& rgb, std::vector<Detection_Yolov5>& objects) 
{
    objects.clear();

    int orig_w = rgb.cols;
    int orig_h = rgb.rows;

    float r_w = (float)target_size / orig_w;
    float r_h = (float)target_size / orig_h;
    
    int w, h, x, y;
    if (r_h > r_w) 
    {
        w = target_size;
        h = static_cast<int>(r_w * orig_h);
        x = 0;
        y = (target_size - h) / 2;
    }
    else 
    {
        w = static_cast<int>(r_h * orig_w);
        h = target_size;
        x = (target_size - w) / 2;
        y = 0;
    }

    // resize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, orig_w, orig_h, w, h);

    // letterbox padding
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, y, target_size - h - y, x, target_size - w - x, 
                          ncnn::BORDER_CONSTANT, 114.f);

    // 归一化 (除以255)
    in_pad.substract_mean_normalize(0, norm_vals);

    // 推理
    ncnn::Extractor ex = yolov5_plate.create_extractor();
    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    // 生成候选框
    generate_plate_proposals(out, prob_threshold, objects);

    std::cout << "Before NMS: " << objects.size() << " objects" << std::endl;

    // 按置信度排序 
    qsort_descent_inplace(objects);

    // NMS
    std::vector<int> picked;
    nms_sorted_bboxes(objects, picked, nms_threshold);

    std::cout << "After NMS: " << picked.size() << " objects" << std::endl;

    // 保留 NMS 后的结果
    std::vector<Detection_Yolov5> nms_objects;
    for (int idx : picked)
        nms_objects.push_back(objects[idx]);
    objects.swap(nms_objects);

    // 恢复到原图尺寸 
    for (auto& det : objects)
    {
        float bbox_center[4] = {det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]};
        float lmk[8];
        for (int k = 0; k < 8; k++) {
            lmk[k] = det.landmark[k];
        }
        
        int l, r, t, b;
        if (r_h > r_w) 
        {
            l = (bbox_center[0] - bbox_center[2]/2) / r_w;
            r = (bbox_center[0] + bbox_center[2]/2) / r_w;
            t = (bbox_center[1] - bbox_center[3]/2 - (target_size - r_w * orig_h) / 2) / r_w;
            b = (bbox_center[1] + bbox_center[3]/2 - (target_size - r_w * orig_h) / 2) / r_w;
            
            // 转换landmark
            for (int k = 0; k < 8; k += 2) 
            {
                det.landmark[k] = lmk[k] / r_w;
                det.landmark[k+1] = (lmk[k+1] - (target_size - r_w * orig_h) / 2) / r_w;
            }
        } 
        else 
        {
            l = (bbox_center[0] - bbox_center[2]/2 - (target_size - r_h * orig_w) / 2) / r_h;
            r = (bbox_center[0] + bbox_center[2]/2 - (target_size - r_h * orig_w) / 2) / r_h;
            t = (bbox_center[1] - bbox_center[3]/2) / r_h;
            b = (bbox_center[1] + bbox_center[3]/2) / r_h;
            
            // 转换landmark
            for (int k = 0; k < 8; k += 2) 
            {
                det.landmark[k] = (lmk[k] - (target_size - r_h * orig_w) / 2) / r_h;
                det.landmark[k+1] = lmk[k+1] / r_h;
            }
        }
        
        // 转换为角点格式并边界检查
        det.bbox[0] = std::max(1.f, (float)l);
        det.bbox[1] = std::max(1.f, (float)t);
        det.bbox[2] = (r > l) ? r : l + 1;
        det.bbox[2] = std::min((float)(orig_w - 1), det.bbox[2]);
        det.bbox[3] = (b > t) ? b : t + 1;
        det.bbox[3] = std::min((float)(orig_h - 1), det.bbox[3]);
        
        // landmark边界检查
        for (int k = 0; k < 8; k += 2) {
            det.landmark[k] = std::max(0.f, std::min((float)(orig_w - 1), det.landmark[k]));
            det.landmark[k+1] = std::max(0.f, std::min((float)(orig_h - 1), det.landmark[k+1]));
        }
    }

    return 0;
}

int Detector_Yolov5plate::detect_batch(const std::vector<cv::Mat>& rgb_batch, 
                                       std::vector<std::vector<Detection_Yolov5>>& batch_objects)
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

int Detector_Yolov5plate::draw(cv::Mat& rgb, const std::vector<Detection_Yolov5>& objects) 
{
    static cv::Scalar lm_colors[] = {
        cv::Scalar(255, 255, 0),  // 左上 - 黄色
        cv::Scalar(0, 0, 255),    // 右上 - 红色
        cv::Scalar(0, 255, 0),    // 右下 - 绿色
        cv::Scalar(255, 0, 255)   // 左下 - 紫色
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Detection_Yolov5& obj = objects[i];

        int x0 = static_cast<int>(obj.bbox[0]);
        int y0 = static_cast<int>(obj.bbox[1]);
        int x1 = static_cast<int>(obj.bbox[2]);
        int y1 = static_cast<int>(obj.bbox[3]);

        cv::rectangle(rgb, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 0, 0), 2);

        // 绘制 landmark (4个关键点)
        for (int k = 0; k < 8; k += 2)
        {
            int color_idx = k / 2;
            int lm_x = static_cast<int>(obj.landmark[k]);
            int lm_y = static_cast<int>(obj.landmark[k + 1]);
            cv::circle(rgb, cv::Point(lm_x, lm_y), 4, lm_colors[color_idx], -1);
        }

        // 绘制标签
        char text[64];
        sprintf(text, "%d:%.2f", obj.label, obj.class_confidence);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

        int tx = x0;
        int ty = y0 - label_size.height - baseLine;
        if (ty < 0) ty = 0;
        if (tx + label_size.width > rgb.cols) tx = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(tx, ty), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(rgb, text, cv::Point(tx, ty + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1.5);
    }

    return 0;
}