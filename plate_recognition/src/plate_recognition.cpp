#include "plate_recognition.h"
#include "cpu.h"
#include <iostream>

// 定义全局变量
std::vector<std::string> plate_color_list = {"黑色", "蓝色", "绿色", "白色", "黄色"};

std::string plate_chr[78] = {"#","京","沪","津","渝","冀","晋","蒙","辽","吉",
    "黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵",
    "云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民",
    "航","危","0","1","2","3","4","5","6","7","8","9","A","B","C","D","E",
    "F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X",
    "Y","Z","险","品"};

PlateRecognition::PlateRecognition()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    plate_recognition.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    plate_recognition.opt = ncnn::Option();

#if NCNN_VULKAN
    plate_recognition.opt.use_vulkan_compute = use_gpu;
#endif

    plate_recognition.opt.num_threads = ncnn::get_big_cpu_count();
    plate_recognition.opt.blob_allocator = &blob_pool_allocator;
    plate_recognition.opt.workspace_allocator = &workspace_pool_allocator;

    plate_recognition.load_param(PLATE_RECOGNITION_PARAM);
    plate_recognition.load_model(PLATE_RECOGNITION_BIN);
}

PlateRecognition::~PlateRecognition()
{
    plate_recognition.clear();
}

int PlateRecognition::recognize(const cv::Mat& img, std::string& plate_str, std::string& plate_color)
{
    plate_str.clear();
    plate_color.clear();

    if (img.empty())
    {
        std::cerr << "[PlateRecognition] Input image is empty" << std::endl;
        return -1;
    }

    // 预处理：resize 到模型输入尺寸
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_W, INPUT_H));

    // 转换为 ncnn::Mat 并归一化
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, INPUT_W, INPUT_H);
    in.substract_mean_normalize(mean_vals, norm_vals);

    // 推理
    ncnn::Extractor ex = plate_recognition.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out0, out1;
    ex.extract("out0", out0);  // 字符识别输出
    ex.extract("out1", out1);  // 颜色分类输出

    // ===== 颜色识别 =====
    if (out1.empty() || out1.total() < 5)
    {
        std::cerr << "[PlateRecognition] Color output format error" << std::endl;
        return -1;
    }

    // 找到最大概率的颜色索引
    int max_color_idx = 0;
    float max_color_prob = out1[0];
    for (int i = 1; i < 5; ++i)
    {
        if (out1[i] > max_color_prob)
        {
            max_color_prob = out1[i];
            max_color_idx = i;
        }
    }

    if (max_color_idx >= 0 && max_color_idx < (int)plate_color_list.size())
    {
        plate_color = plate_color_list[max_color_idx];
    }
    else
    {
        plate_color = "未知";
    }

    // ===== 字符识别 =====
    if (out0.empty() || out0.w != CLS || out0.h != POS)
    {
        std::cerr << "[PlateRecognition] Character output format error" << std::endl;
        return -1;
    }

    // 对每个位置找到最大概率的字符
    std::vector<int> plate_index;
    plate_index.reserve(POS);
    
    for (int pos = 0; pos < POS; ++pos)
    {
        const float* probs = out0.row(pos);
        int max_idx = 0;
        float max_prob = probs[0];
        
        for (int cls = 1; cls < CLS; ++cls)
        {
            if (probs[cls] > max_prob)
            {
                max_prob = probs[cls];
                max_idx = cls;
            }
        }
        plate_index.push_back(max_idx);
    }

    // CTC 解码（去除重复和空白符）
    int pre = 0;
    for (size_t j = 0; j < plate_index.size(); ++j)
    {
        if (plate_index[j] != 0 && plate_index[j] != pre)
        {
            if (plate_index[j] > 0 && plate_index[j] < 78)
            {
                plate_str += plate_chr[plate_index[j]];
            }
        }
        pre = plate_index[j];
    }

    return 0;
}