//  JumpML Rocketship - Neural Network Inference with Audio Processing
// 
//  Copyright 2020-2024 JUMPML
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// 
//  nn_layers_tools.h
//
//  Created by Raghavendra Prabhu on 10/12/21.
//
#ifndef NN_LAYERS_TOOLS_H_
#define NN_LAYERS_TOOLS_H_

#include "signalsifter_config.h"
#include "dsplib.h"

typedef struct {
    unsigned int m;
    unsigned int n;
    unsigned int numBits;
    float maxv;
    float minv;
    float scale;
    float scaleInt;
    int enableRounding;
} LayerQParams;

typedef struct {
    float min;
    float max;
    float mean;
    float std;
    int numZeros;
} LayerStats;

typedef struct {
    int16_t min;
    int16_t max;
    int16_t mean;
    int16_t std;
    int numZeros;
} LayerStats_S16;

void initLayerStats(LayerStats *ls);
void initLayerQParams(LayerQParams *lqp, int numFracBits, int numBits, int roundingEnable);
void applyLayerQuantization(LayerQParams *lqp, float *A, JMPDSP_Stride iA, JMPDSP_Length N);
void computeLayerStats(LayerStats *ls, const float *A, JMPDSP_Stride iA, JMPDSP_Length N);

#endif /* NN_LAYERS_TOOLS_H_ */
