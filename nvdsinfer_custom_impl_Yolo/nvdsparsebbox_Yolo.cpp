/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "nvdsinfer_custom_impl.h"
#include <time.h>
#include <map>

#define kNMS_THRESH 0.45
#define MAX_OUTPUT_BBOX_COUNT 1000

static constexpr int LOCATIONS = 4;
struct alignas(float) Detection{
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(Detection& a, Detection& b) {
    return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float *output, float conf_thresh, float nms_thresh) {
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

/* This is a sample bounding box parsing function for the sample YoloV5 detector model */
static bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    const float kCONF_THRESH = detectionParams.perClassThreshold[0];

    // std::vector<Detection> res;
    //nms(res, (float*)(outputLayersInfo[0].buffer), kCONF_THRESH, kNMS_THRESH);
    float *scores = (float*)(outputLayersInfo[0].buffer);
    float *boxes = (float*)(outputLayersInfo[1].buffer);
    float *classes = (float*)(outputLayersInfo[2].buffer);
    float *points = (float*)(outputLayersInfo[3].buffer);
    
    //for(auto& r : res) 
    for(int i = 0; i<MAX_OUTPUT_BBOX_COUNT; i++)
    {
        if(scores[i] <= 0.01) break;

	    NvDsInferParseObjectInfo oinfo;        
	    oinfo.classId = classes[i];
	    oinfo.left    = static_cast<unsigned int>(boxes[4*i+0]);
	    oinfo.top     = static_cast<unsigned int>(boxes[4*i+1]);
	    oinfo.width   = static_cast<unsigned int>(boxes[4*i+2]-boxes[4*i+0]);
	    oinfo.height  = static_cast<unsigned int>(boxes[4*i+3]-boxes[4*i+1]);
	    oinfo.detectionConfidence = scores[i];

        oinfo.points[0] = static_cast<unsigned int>(points[4*i+0]);
        oinfo.points[1] = static_cast<unsigned int>(points[4*i+1]);
        oinfo.points[2] = static_cast<unsigned int>(points[4*i+2]);
        oinfo.points[3] = static_cast<unsigned int>(points[4*i+3]);

	    objectList.push_back(oinfo);    

        // yolo: %d %f (%f %f %f %f) (%f %f %f %f)\n", oinfo.classId, oinfo.detectionConfidence,  oinfo.left, oinfo.top, oinfo.width, oinfo.height, oinfo.points[0], oinfo.points[1], oinfo.points[2], oinfo.points[3]);

        // NvDsInferParseObjectInfo oinfo1;        
	    // oinfo1.classId = classes[i];
	    // oinfo1.left    = static_cast<unsigned int>(points[4*i+0]);
	    // oinfo1.top     = static_cast<unsigned int>(points[4*i+1]);
	    // oinfo1.width   = static_cast<unsigned int>((boxes[4*i+2]-boxes[4*i+0])/2.0);
	    // oinfo1.height  = static_cast<unsigned int>((boxes[4*i+3]-boxes[4*i+1])/2.0);
	    // oinfo1.detectionConfidence = scores[i];
        // objectList.push_back(oinfo1);  

        // NvDsInferParseObjectInfo oinfo2;        
	    // oinfo2.classId = classes[i];
	    // oinfo2.left    = static_cast<unsigned int>(points[4*i+2]);
	    // oinfo2.top     = static_cast<unsigned int>(points[4*i+3]);
	    // oinfo2.width   = static_cast<unsigned int>((boxes[4*i+2]-boxes[4*i+0])/2.0);
	    // oinfo2.height  = static_cast<unsigned int>((boxes[4*i+3]-boxes[4*i+1])/2.0);
	    // oinfo2.detectionConfidence = 0.1;
        // objectList.push_back(oinfo2);  
    }
    
    return true;
}

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    clock_t st_time, ed_time;
    st_time = clock();

    return NvDsInferParseYoloV5(
        outputLayersInfo, networkInfo, detectionParams, objectList);

    ed_time = clock();
    printf("NvDsInferParseCustomYoloV5 %f s \n", (double )(ed_time - st_time)/CLOCKS_PER_SEC );
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);
