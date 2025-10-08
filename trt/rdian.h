#pragma once 

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include <opencv2/dnn.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <numeric>
#include <chrono>

#include "trt_utils.h"


class RDIAN  {
private:
    // Model parameters
    int input_w;  // Input image width expected by the model
    int input_h;  // Input image height expected by the model
    int num_detections;  // Number of detections output by the model
    int detection_attribute_size;  // Attributes (e.g., bbox, class) per detection
    int num_classes = 80;  // Number of classes (e.g., COCO dataset has 80 classes)
    static const int nc{10}; // number of class

    // Maximum supported image size (used for memory allocation checks)

    // Confidence threshold for filtering detections
    float conf_threshold = 0.3f; //0.3

    // Non-Maximum Suppression (NMS) threshold to remove duplicate boxes
    float nms_threshold = 0.4f; // 0.4


    int img_w;
    int img_h;
    void* iresized{nullptr};
    cudaStream_t stream;
    std::string engine_file_path;
    std::unique_ptr<nvinfer1::IExecutionContext,TRTDestroy> context;
    std::unique_ptr<nvinfer1::ICudaEngine,TRTDestroy> engine;
    Logger gLogger;
    std::vector<void*> buffers;
    float* output_buffer;
    float* blob;
    
    cv::Rect2i win;
    float scale;

    int fd_blob {-1};
    // NvBufferSession nbs;
    
    void loadEngine();
    void cudaBlobFromImageRGB(void* img_rgb, float* blob, int pitch);
    void cudaCropAndBlobFromImageRGB(void* img_rgb, float* blob);

public:

    RDIAN(int img_w, int img_h);
    ~RDIAN();
    /* std::vector<Object> */void apply(int fd);
    static void gen_engine_from_onnx();
};





