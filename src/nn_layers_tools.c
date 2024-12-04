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
//  nn_layers_tools.c
//
//  Created by Raghavendra Prabhu on 10/12/21.
//  

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>

#include "nn_layers_tools.h"

void applyLayerQuantization(LayerQParams *lqp, float *A, JMPDSP_Stride iA, JMPDSP_Length N)
{
    int tempInt[MAX_NEURONS];
    JMPDSP_vclip(A, iA, &lqp->minv, &lqp->maxv, A, iA, N);
    JMPDSP_vsmul(A, iA, &lqp->scaleInt, A, iA, N);
    switch (lqp->numBits)
    {
        case 8:
            if (lqp->enableRounding == 1)
                JMPDSP_vfixr8(A, iA, (int8_t *) tempInt, 1, N);
            else
                JMPDSP_vfix8(A, iA, (int8_t *) tempInt, 1, N);
            
            JMPDSP_vflt8((int8_t *) tempInt, 1, A, iA, N);
            break;
            
        case 16:
            if (lqp->enableRounding == 1)
                JMPDSP_vfixr16(A, iA, (short *) tempInt, 1, N);
            else
                JMPDSP_vfix16(A, iA, (short *) tempInt, 1, N);
            
            JMPDSP_vflt16((short *) tempInt, 1, A, iA, N);
            break;
            
        case 32:
            if (lqp->enableRounding == 1)
                JMPDSP_vfixr32(A, iA, (int *) tempInt, 1, N);
            else
                JMPDSP_vfix32(A, iA, (int *) tempInt, 1, N);
            
            JMPDSP_vflt32((int *) tempInt, 1, A, iA, N);
            break;
            
        default:
            printf("VALUE ERROR numBits=%d\n",lqp->numBits);
            break;
    }
    JMPDSP_vsmul(A, iA, &lqp->scale, A, iA, N);
}

void initLayerStats(LayerStats *ls)
{
    ls->max = 0.0f;
    ls->min = FLT_MAX;
    ls->mean = 0.0f;
    ls->std = 0.0f;
    ls->numZeros = 0;
}
void initLayerQParams(LayerQParams *lqp, int numFracBits, int numBits, int roundingEnable)
{
    float maxVal;
    int m;
    m = numBits - numFracBits - 1; // 1 for sign bit
    assert(m >= -1);
    maxVal = powf(2, m);
    lqp->m = m;
    lqp->n = numFracBits;
    lqp->scale = powf(2, -numFracBits);
    lqp->scaleInt = powf(2, numFracBits);
    lqp->numBits = numBits;
    if (m == -1)
        lqp->minv = 0.0f;
    else
        lqp->minv = -maxVal;
    
    lqp->maxv =  maxVal - lqp->scale;
    lqp->enableRounding = roundingEnable;
}

void computeLayerStats(LayerStats *ls, const float *A, JMPDSP_Stride iA, JMPDSP_Length N)
{
    float mean, val;
    
    JMPDSP_meanv(A, iA, &mean, N);
    ls->mean = 0.1 * mean  + 0.9 * ls->mean;
    
    JMPDSP_maxv(A, iA, &val, N);
    ls->max = MAX(val, ls->max);
    
    JMPDSP_minv(A, iA, &val, N);
    ls->min = MIN(val, ls->min);
    
    JMPDSP_measqv(A, iA, &val, N);
    val = sqrtf(val - mean * mean);
    ls->std = 0.1f * val  + 0.9f * ls->std;
}
