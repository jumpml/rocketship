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
//  nnlib_float.c
//
//  Created by Raghavendra Prabhu on 2/10/22.
//

#include "nnlib_float.h"

#ifndef USE_NEON
void JMPNN_linear_matXvec_S8xF32_F32(const int8_t * __restrict__ W, const float * __restrict__ input, const int8_t * __restrict__ bias,
                                  float * __restrict__ output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                  float weightScale, float biasScale)
{
    int i, j;
    for (i=0;i<output_len;i++)
    {
        float sum = 0.0f;
        for (j=0;j<input_len;j++)
            sum += W[i*input_len + j]*input[j];
        output[i] = weightScale*sum + biasScale*bias[i];
    }
    
}
void JMPNN_gru_matXvec_S8xF32_F32_act(const int8_t * __restrict__ Wi, const float * __restrict__ input, const int8_t * __restrict__ Bi,
                               const int8_t * __restrict__ Wh, const float * __restrict__ prev_state, const int8_t * __restrict__ Bh,
                               float * __restrict__ output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                               float weightScale, float biasScale, JMPNN_ActivationType actType)
{
    int i, j;
    for (i=0;i<output_len;i++)
    {
        float sum = 0.0f;
        for (j=0;j<input_len;j++)
            sum += Wi[i*input_len + j]*input[j];
        for (j=0;j<output_len;j++)
            sum += Wh[i*output_len + j]*prev_state[j];
        output[i] = weightScale*sum + biasScale*(Bi[i] + Bh[i]);
    }
    JMPNN_apply_activation_F32(output, output_len, actType);
}

void JMPNN_gru_newGate_S8xF32_F32_act(const int8_t * __restrict__ Wi, const float * __restrict__ input, const int8_t * __restrict__ Bi,
                                      const int8_t * __restrict__ Wh, const float * __restrict__ prev_state, const int8_t * __restrict__ Bh,
                                      const float * __restrict__ r, float * __restrict__ output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      float weightScale, float biasScale, JMPNN_ActivationType actType)
{
    int i, j;
    for (i=0;i<output_len;i++)
    {
        float sum = 0.0f;
        float rec_sum = 0.0f;
        for (j=0;j<input_len;j++)
            sum += Wi[i*input_len + j]*input[j];
        for (j=0;j<output_len;j++)
            rec_sum += Wh[i*output_len + j]*prev_state[j];
        rec_sum = weightScale*rec_sum + biasScale*Bh[i];
        sum = weightScale*sum + biasScale*Bi[i];
        output[i] = sum + rec_sum*r[i];
    }
    JMPNN_apply_activation_F32(output, output_len, actType);
}
#endif  //#ifndef USE_NEON

void JMPNN_apply_activation_F32(float *data, JMPDSP_Length N, JMPNN_ActivationType actType)
{
    if (actType == ACTIVATION_SIGMOID)
    {
        vec_sigmoid_F32(data, data, N);
    }
    else if (actType == ACTIVATION_TANH)
    {
        vec_tanh_F32(data, data, N);
    }
    else if (actType == ACTIVATION_RELU)
    {
        vec_relu_F32(data, data, N);
    }
    else
    {
        // ERROR
    }
}

void JMPNN_vec_interpolation_F32(float *output, const float *interp_vec,
                                 const float *input1, const float *input2,
                                 JMPDSP_Length N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        output[i] = interp_vec[i] * input1[i] + (1.0f - interp_vec[i]) * input2[i];
    }
}