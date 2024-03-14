//  JumpML Rocketship - Neural Network Inference with Audio Processing
// 
//  Copyright 2020-2024 JUMPML LLC
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
//  nnlib_float.h
//
//  Created by Raghavendra Prabhu on 2/10/22.
//

#ifndef NNLIB_FLOAT_H
#define NNLIB_FLOAT_H

#include "nnlib_types.h"
#include "tanh_table.h"
#include <math.h>
#include "common_def.h"

static inline float tanh_approx(float x)
{
    int i;
    float y, dy;
    float sign=1;
    if (x<0)
    {
        x=-x;
        sign=-1;
    }
    i = (int)floorf(0.5f+TANH_SCALEFAC*x);
    i = MAX(0, MIN(TANH_TABLE_MAXINDEX,i));
    x -= TANH_DELTAX*i;
    y = tanh_table[i];
    dy = 1-y*y;
    float temp = x*dy;
    float one_xy = 1 - y*x;
    temp = temp * one_xy;
    y = y + x*dy*(1 - y*x);
    return sign*y;
}

static inline float sigmoid_approx(float x)
{
    return 0.5f + 0.5f * tanh_approx(0.5f*x);
}

static inline float relu(float x)
{
    return x < 0 ? 0 : x;
}

static inline void vec_tanh_F32(float *output, const float *input, JMPDSP_Length N)
{
    int i;
    for (i=0;i<N;i++)
    {
        output[i] = tanh_approx(input[i]);
    }
}

static inline void vec_sigmoid_F32(float *output, const float *input, JMPDSP_Length N)
{
    int i;
    for (i=0;i<N;i++)
    {
        output[i] = sigmoid_approx(input[i]);
    }
}

static inline void vec_relu_F32(float *output, const float *input, JMPDSP_Length N)
{
    int i;
    for (i=0;i<N;i++)
    {
        output[i] = relu(input[i]);
    }
}

// FLOATING-POINT ARITHMETIC (FIXED-POINT PARAMETERS)
void JMPNN_linear_matXvec_S8xF32_F32(const int8_t *W, const float *input, const int8_t *bias,
                                     float *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                     float weightScale, float biasScale);


void JMPNN_gru_matXvec_S8xF32_F32_act(const int8_t *Wi, const float *input, const int8_t *Bi,
                                      const int8_t *Wh, const float *prev_state, const int8_t *Bh,
                                      float *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      float weightScale, float biasScale, JMPNN_ActivationType actType); // used for reset/update gates

void JMPNN_gru_newGate_S8xF32_F32_act(const int8_t *Wi, const float *input, const int8_t *Bi,
                                      const int8_t *Wh, const float *prev_state, const int8_t *Bh, const float *r,
                                      float *output, JMPDSP_Length input_len, JMPDSP_Length output_len, 
                                      float weightScale, float biasScale, JMPNN_ActivationType actType);

void JMPNN_apply_activation_F32(float *data, JMPDSP_Length N, JMPNN_ActivationType actType);
void JMPNN_vec_interpolation_F32(float *output, const float *interp_vec,
                                 const float *input1, const float *input2,
                                 JMPDSP_Length N);


#endif /* NNLIB_FLOAT_H */
