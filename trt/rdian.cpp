#include "rdian.h"

#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }


static inline float intersection_area1(const cv::Rect_<float> &a_rect, const cv::Rect_<float> &b_rect)
{
    cv::Rect_<float> inter = a_rect & b_rect;
    return inter.area();
}


void RDIAN::loadEngine()
{
    char *trtModelStream{nullptr};
    std::size_t size{0};

    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroy> runtime{nvinfer1::createInferRuntime(gLogger)};
    assert(runtime != nullptr);
    engine.reset(runtime->deserializeCudaEngine(trtModelStream, size));
    assert(engine != nullptr); 
    context.reset( engine->createExecutionContext() );
    assert(context != nullptr);
    delete[] trtModelStream;
}

RDIAN::RDIAN(int img_w ,int img_h)
    :img_w{img_w},img_h{img_h}
{
    engine_file_path = "/home/user/rdian.engine";
    loadEngine();
    buffers.reserve(engine->getNbBindings());
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    // detection_attribute_size = engine->getBindingDimensions(1).d[1];
    // num_detections = engine->getBindingDimensions(1).d[2];

    std::cout << "----->RDIAN " << input_h << " "<< input_w << " "<< detection_attribute_size << " "<< num_detections << std::endl;
    // ----->YOLO12 640 640 84 8400
    // num_classes = detection_attribute_size - 4;

    cudaMalloc(&blob, 1 * input_w * input_h * sizeof(float));
    // CUDA_CHECK();
    // // Initialize output buffer
    cudaMallocManaged((void **) &output_buffer, 1 * input_w * input_h * sizeof(float));

    buffers[0] = (void *)blob;
    buffers[1] = (void *) output_buffer;

    // NvBufferCreateParams input_params = {0};
    // input_params.payloadType = NvBufferPayload_SurfArray;
    // input_params.width = input_w;
    // input_params.height = input_h;
    // input_params.layout = NvBufferLayout_Pitch;
    // input_params.colorFormat = NvBufferColorFormat_ABGR32;
    // input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT; 

    // if (-1 == NvBufferCreateEx(&fd_blob, &input_params))
    //     std::cout<<"Failed to create NvBuffer fd_blob"<<std::endl;

    // nbs = NvBufferSessionCreate();

    CUDA_CHECK(cudaStreamCreate(&stream));
}

RDIAN::~RDIAN() {

}

void RDIAN::apply(int fd) {
    // std::vector<Object> objects;

    // win = param->getSearchWindow();
    // // scale = param->getYolo12Scale();

    // int src_dmabuf_fds[1];
    // src_dmabuf_fds[0] = fd;

    // NvBufferCompositeParams composite_params;
    // memset(&composite_params, 0, sizeof(composite_params));

    // composite_params.composite_flag = NVBUFFER_COMPOSITE;
    // composite_params.input_buf_count = 1;
    // composite_params.composite_filter[0] = NvBufferTransform_Filter_Bilinear;// NvBufferTransform_Filter_Nicest; 
    // composite_params.dst_comp_rect_alpha[0] = 1.0f;

    // composite_params.src_comp_rect[0].left = win.x; // vid_io_param->inConf[vid_idx].crop_left;
    // composite_params.src_comp_rect[0].top = win.y;
    // composite_params.src_comp_rect[0].width = win.width;
    // composite_params.src_comp_rect[0].height = win.height;

    // composite_params.dst_comp_rect[0].top = 0; 
    // composite_params.dst_comp_rect[0].left = 0; 
    // composite_params.dst_comp_rect[0].width = input_w;
    // composite_params.dst_comp_rect[0].height = input_h; 
    // composite_params.session = nbs;
    // NvBufferComposite(src_dmabuf_fds, fd_blob, &composite_params);

    // CudaProcess cup{fd_blob};
    // auto blob_ptr = cup.getImgPtr();
    
    // cudaBlobFromImageRGB(blob_ptr, blob, cup.getPitch()/4);
    // cup.freeImage();

    context->enqueueV2(buffers.data(), stream, nullptr);
    cudaStreamSynchronize(stream);
    // postprocess(objects);

    return;
}
