#include <assert.h>
#include <vector>
#include <iostream>
#include "yololayer.h"
#include "cuda_utils.h"


#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <cmath>

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>


namespace Tn
{
    template<typename T> 
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T> 
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }

        size_t YoloLen = sizeof(float) * mMaxOutObject * 66;
        CUDA_CHECK(cudaMallocHost(&yolo_output, 4 * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&yolo_output[0], YoloLen));  //score
        CUDA_CHECK(cudaMalloc(&yolo_output[1], YoloLen * 4)); //box
        CUDA_CHECK(cudaMalloc(&yolo_output[2], YoloLen));  //class
        CUDA_CHECK(cudaMalloc(&yolo_output[3], YoloLen * 4)); //point
    }
    YoloLayerPlugin::~YoloLayerPlugin()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
        cudaFree((void*)yolo_output[0]);
        cudaFree(yolo_output[1]);
        cudaFree(yolo_output[2]);
        cudaFree(yolo_output[3]);
        cudaFreeHost(yolo_output);
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer) const noexcept
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const noexcept
    {
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int YoloLayerPlugin::initialize() noexcept
    {
        return 0;
    }

    // DimsExprs YoloLayerPlugin::getOutputDimensions(int outputIndex, const DimsExprs *inputs,
    // int nbInputs, IExprBuilder &exprBuilder) noexcept override 
    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
    {
        //output the result to channel
        int totalsize = ((index == 1 || index==3) ? 4 : 1);
        return Dims3(mMaxOutObject, totalsize, 1);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const noexcept
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) noexcept
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() noexcept {}

    const char* YoloLayerPlugin::getPluginType() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const noexcept
    {
        return "1";
    }

    void YoloLayerPlugin::destroy() noexcept
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPlugin::clone() const noexcept
    {
        YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    size_t YoloLayerPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
    {
        return this->nms_fun(maxBatchSize, nullptr, nullptr, mMaxOutObject, mMaxOutObject, 0.2f, 
        nullptr, 0, nullptr);
    }

#define CUDA_ALIGN 256
    template <typename T>
    inline size_t get_size_aligned(size_t num_elem) {
        size_t size = num_elem * sizeof(T);
        size_t extra_align = 0;
        if (size % CUDA_ALIGN != 0) {
            extra_align = CUDA_ALIGN - size % CUDA_ALIGN;
        }
        return size + extra_align;
    }
        template <typename T>
    inline T *get_next_ptr(size_t num_elem, void *&workspace, size_t &workspace_size) {
        size_t size = get_size_aligned<T>(num_elem);
        if (size > workspace_size) {
            throw std::runtime_error("Workspace is too small!");
        }
        workspace_size -= size;
        T *ptr = reinterpret_cast<T *>(workspace);
        workspace = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(workspace) + size);
        return ptr;
    }

    __global__ void nms_kernel(
        const int num_per_thread, const float threshold, const int num_detections,
        const int *indices, float *scores, const float *classes, const float4 *boxes) {

        // Go through detections by descending score
        for (int m = 0; m < num_detections; m++) {
            for (int n = 0; n < num_per_thread; n++) {
                int i = threadIdx.x * num_per_thread + n;
                if (i < num_detections && m < i && scores[m] > 0.0f) {
                    int idx = indices[i];
                    int max_idx = indices[m];
                    int icls = classes[idx];
                    int mcls = classes[max_idx];
                    if (mcls == icls) {
                        float4 ibox = boxes[idx];
                        float4 mbox = boxes[max_idx];
                        float x1 = max(ibox.x, mbox.x);
                        float y1 = max(ibox.y, mbox.y);
                        float x2 = min(ibox.z, mbox.z);
                        float y2 = min(ibox.w, mbox.w);
                        float w = max(0.0f, x2 - x1 + 1);
                        float h = max(0.0f, y2 - y1 + 1);
                        float iarea = (ibox.z - ibox.x + 1) * (ibox.w - ibox.y + 1);
                        float marea = (mbox.z - mbox.x + 1) * (mbox.w - mbox.y + 1);
                        float inter = w * h;
                        float overlap = inter / (iarea + marea - inter);
                        if (overlap > threshold) {
                            scores[i] = 0.0f;
                        }
                    }
                }
            }
            // Sync discarded detections
            __syncthreads();
        }
    }

    int YoloLayerPlugin::nms_fun(int batch_size, void **inputs, void *const* outputs, size_t count, int detections_per_im, float nms_thresh, void *workspace, size_t workspace_size, cudaStream_t stream) const {

        if (!workspace || !workspace_size) {
            // Return required scratch space size cub style
            workspace_size  = get_size_aligned<bool>(count);  // flags
            workspace_size += get_size_aligned<int>(count);   // indices
            workspace_size += get_size_aligned<int>(count);   // indices_sorted
            workspace_size += get_size_aligned<float>(count); // scores
            workspace_size += get_size_aligned<float>(count); // scores_sorted
        
            size_t temp_size_flag = 0;
            cub::DeviceSelect::Flagged((void *)nullptr, temp_size_flag,
            cub::CountingInputIterator<int>(count),
            (bool *)nullptr, (int *)nullptr, (int *)nullptr, count);
            size_t temp_size_sort = 0;
            cub::DeviceRadixSort::SortPairsDescending((void *)nullptr, temp_size_sort,
            (float *)nullptr, (float *)nullptr, (int *)nullptr, (int *)nullptr, count);
            workspace_size += std::max(temp_size_flag, temp_size_sort);

            return workspace_size;
        }

        auto on_stream = thrust::cuda::par.on(stream);

        auto flags = get_next_ptr<bool>(count, workspace, workspace_size);
        auto indices = get_next_ptr<int>(count, workspace, workspace_size);
        auto indices_sorted = get_next_ptr<int>(count, workspace, workspace_size);
        auto scores = get_next_ptr<float>(count, workspace, workspace_size);
        auto scores_sorted = get_next_ptr<float>(count, workspace, workspace_size);

        // printf("nms batch %d \n", batch_size);

        for (int batch = 0; batch < batch_size; batch++) {
            auto in_scores = static_cast<const float *>(inputs[0]) + batch * count;
            auto in_boxes = static_cast<const float4 *>(inputs[1]) + batch * count;
            auto in_classes = static_cast<const float *>(inputs[2]) + batch * count;
            auto in_points = static_cast<const float4 *>(inputs[3]) + batch * count;
            
            auto out_scores = static_cast<float *>(outputs[0]) + batch * detections_per_im;
            auto out_boxes = static_cast<float4 *>(outputs[1]) + batch * detections_per_im;
            auto out_classes = static_cast<float *>(outputs[2]) + batch * detections_per_im;
            auto out_points = static_cast<float4 *>(outputs[3]) + batch * detections_per_im;
            

            // float tmp[10];
            // cudaMemcpyAsync(tmp, in_scores, 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            // printf("input %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n", tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9]);
            // cudaMemcpyAsync(tmp, out_scores, 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            // printf("output %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n", tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9]);

            // Discard null scores
            thrust::transform(on_stream, in_scores, in_scores + count,
            flags, thrust::placeholders::_1 > 0.0f);

            int *num_selected = reinterpret_cast<int *>(indices_sorted);
            cub::DeviceSelect::Flagged(workspace, workspace_size, cub::CountingInputIterator<int>(0),
            flags, indices, num_selected, count, stream);
            cudaStreamSynchronize(stream);
            int num_detections = *thrust::device_pointer_cast(num_selected);

            // Sort scores and corresponding indices
            thrust::gather(on_stream, indices, indices + num_detections, in_scores, scores);
            cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
            scores, scores_sorted, indices, indices_sorted, num_detections, 0, sizeof(*scores)*8, stream);

            // Launch actual NMS kernel - 1 block with each thread handling n detections
            const int max_threads = 1024;
            int num_per_thread = ceil((float)num_detections / max_threads);
            nms_kernel<<<1, max_threads, 0, stream>>>(num_per_thread, nms_thresh, num_detections,
            indices_sorted, scores_sorted, in_classes, in_boxes);

            // Re-sort with updated scores
            cub::DeviceRadixSort::SortPairsDescending(workspace, workspace_size,
            scores_sorted, scores, indices_sorted, indices, num_detections, 0, sizeof(*scores)*8, stream);

            // Gather filtered scores, boxes, classes
            num_detections = min(detections_per_im, num_detections);
            cudaMemcpyAsync(out_scores, scores, num_detections * sizeof *scores, cudaMemcpyDeviceToDevice, stream);
            if (num_detections < detections_per_im) {
                thrust::fill_n(on_stream, out_scores + num_detections, detections_per_im - num_detections, 0);
            }
            thrust::gather(on_stream, indices, indices + num_detections, in_boxes, out_boxes);
            thrust::gather(on_stream, indices, indices + num_detections, in_classes, out_classes);
            thrust::gather(on_stream, indices, indices + num_detections, in_points, out_points);

            // printf("num_detections %d \n", num_detections);
            // cudaMemcpyAsync(tmp, out_scores, 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            // printf("output %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n", tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9]);
        }
        
        return 0;
    }

    __device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float **output, int noElements,
        const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes, int outputElem)
    {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid * bnIdx;
        int info_len_i = 5 + classes + Yolo::N2DP;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < CHECK_COUNT; ++k) {
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < 5 + classes; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float *fscore = output[0] + bnIdx * maxoutobject;
            float *fboxes = output[1] + bnIdx * maxoutobject * 4;
            float *fclass = output[2] + bnIdx * maxoutobject;
            float *fpoints = output[3] + bnIdx * maxoutobject * 4;
            int count = (int)atomicAdd(fscore+maxoutobject-1, 1);
            if (count >= maxoutobject) return;
            // char *data = (char*)res_count + sizeof(float) + count * sizeof(Detection);
            // Detection *det = (Detection*)(data);
            fboxes = fboxes + count * 4;
            fpoints = fpoints + count * 4;
            fscore = fscore + count;
            fclass = fclass + count;

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            // pytorch:
            //  y = x[i].sigmoid()
            //  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            //  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            //  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth
            // det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
            // det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;
            fboxes[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
            fboxes[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

            // W: (Pw * e^tw) / FeaturemapW * netwidth
            // v5: https://github.com/ultralytics/yolov5/issues/471
            fboxes[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            fboxes[2] = fboxes[2] * fboxes[2] * anchors[2 * k];
            fboxes[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            fboxes[3] = fboxes[3] * fboxes[3] * anchors[2 * k + 1];
            fboxes[0] -= fboxes[2]/2.0f;
            fboxes[1] -= fboxes[3]/2.0f;
            fboxes[2] += fboxes[0];
            fboxes[3] += fboxes[1];
            fscore[0] = box_prob * max_cls_prob;
            fclass[0] = class_id;

            for(int i=0; i<4; i++){
                fpoints[i] = Logist(curInput[idx + k * info_len_i * total_grid + (5 + classes + i) * total_grid]) * 4.0f - 2.0f;
                if(i%2==0){
                    fpoints[i] = fpoints[i] * anchors[2 * k] + col * netwidth / yoloWidth;
                }else{
                    fpoints[i] = fpoints[i] * anchors[2 * k + 1] + row * netheight / yoloHeight;
                }
                //printf("%f %f %f %f %d %d\n", fpoints[i], Logist(curInput[idx + k * info_len_i * total_grid + (4+i) * total_grid]), anchors[2 * k], anchors[2 * k+1], col, row);
            }
        }
    }

    void YoloLayerPlugin::forwardGpu(const float* const* inputs, float** output, cudaStream_t stream, int batchSize)
    {
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemsetAsync(output[0] + idx * mMaxOutObject, 0, sizeof(float)*mMaxOutObject, stream));
            CUDA_CHECK(cudaMemsetAsync(output[1] + idx * mMaxOutObject*4, 0, sizeof(float)*mMaxOutObject*4, stream));
            CUDA_CHECK(cudaMemsetAsync(output[2] + idx * mMaxOutObject, 0, sizeof(float)*mMaxOutObject, stream));
            CUDA_CHECK(cudaMemsetAsync(output[3] + idx * mMaxOutObject*4, 0, sizeof(float)*mMaxOutObject*4, stream));
        }
        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width * yolo.height * batchSize;
            if (numElem < mThreadCount) mThreadCount = numElem;

            //printf("Net: %d  %d \n", mYoloV5NetWidth, mYoloV5NetHeight);
            CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
                (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem);
        }
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemsetAsync(output[0] + idx * mMaxOutObject + mMaxOutObject - 1, 0, sizeof(float), stream));
        }
    }


    int32_t YoloLayerPlugin::enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,cudaStream_t stream) noexcept
    {
        forwardGpu((const float* const*)inputs, yolo_output, stream, batchSize);

        nms_fun(batchSize, (void **)yolo_output, outputs, mMaxOutObject, mMaxOutObject, 0.2f, workspace, getWorkspaceSize(batchSize), stream);
 
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const noexcept
    {
        return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const noexcept
    {
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames() noexcept
    {
        return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
    {
        assert(fc->nbFields == 2);
        assert(strcmp(fc->fields[0].name, "netinfo") == 0);
        assert(strcmp(fc->fields[1].name, "kernels") == 0);
        int *p_netinfo = (int*)(fc->fields[0].data);
        int class_count = p_netinfo[0];
        int input_w = p_netinfo[1];
        int input_h = p_netinfo[2];
        int max_output_object_count = p_netinfo[3];
        std::vector<Yolo::YoloKernel> kernels(fc->fields[1].length);
        memcpy(&kernels[0], fc->fields[1].data, kernels.size() * sizeof(Yolo::YoloKernel));
        YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}

