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
//  nnlib_fixedpt_generic.c
//
//  Created by Raghavendra Prabhu on 2/10/22.
//

#include "nnlib_fixedpt.h"

void JMPNN_linear_matXvec_S8xS16_S32_generic(const int8_t * __restrict__ W, const int16_t * __restrict__ input, const int8_t * __restrict__ bias,
                                     int32_t * __restrict__ output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                     uint16_t bias_left_shift, uint16_t output_right_shift)
{
    int i, j;
    for (i=0;i<output_len;i++)
    {
        int acc = ((int32_t)(bias[i]) << bias_left_shift);
        for (j=0;j<input_len;j++)
            acc += W[i*input_len + j]*input[j]; //acc: Q7*Q15 = Q22
        output[i] = SLIMIT(acc >> output_right_shift, 15+4); //15 fractional bits, 3 integer bits
    }
}

void JMPNN_gru_matXvec_S8xS16_S16_act_generic(const int8_t * __restrict__ Wi, const int16_t * __restrict__ input, const int8_t * __restrict__ Bi,
                                      const int8_t * __restrict__ Wh, const int16_t * __restrict__ prev_state, const int8_t * __restrict__ Bh,
                                      int16_t * __restrict__ output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      uint16_t Bi_left_shift, uint16_t outi_right_shift, uint16_t Bh_left_shift, uint16_t outh_right_shift,
                                      JMPNN_ActivationType actType)
{
    int i, j;
    int32_t out_S32[MAX_NEURONS];
    
    for (i=0;i<output_len;i++)
    {
        int acci = ((int32_t)(Bi[i]) << Bi_left_shift);
        int acch = ((int32_t)(Bh[i]) << Bh_left_shift);
        for (j=0;j<input_len;j++)
            acci += Wi[i*input_len + j]*input[j]; //acc: Q7*Q15 = Q22
        for (j=0;j<output_len;j++)
            acch += Wh[i*output_len + j]*prev_state[j]; //acc: Q7*Q15 = Q22

        out_S32[i] = SLIMIT((acci >> outi_right_shift) + (acch >> outh_right_shift), 15+4);;
    }
    JMPNN_apply_activation_S16_generic(output, out_S32, output_len, actType);
}

void JMPNN_gru_newGate_S8xS16_S16_act_generic(const int8_t * __restrict__ Wi, const int16_t * __restrict__ input, const int8_t * __restrict__ Bi,
                                      const int8_t * __restrict__ Wh, const int16_t * __restrict__ prev_state, const int8_t * __restrict__ Bh, const int16_t * __restrict__ r,
                                      int16_t * __restrict__ output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      uint16_t Bi_left_shift, uint16_t outi_right_shift, uint16_t Bh_left_shift, uint16_t outh_right_shift,
                                      JMPNN_ActivationType actType)
{
    int32_t out_S32[MAX_NEURONS];
    int i, j;
    for (i=0;i<output_len;i++)
    {
        int acci = ((int32_t)(Bi[i]) << Bi_left_shift);
        int acch = ((int32_t)(Bh[i]) << Bh_left_shift);
        for (j=0;j<input_len;j++)
            acci += Wi[i*input_len + j]*input[j];
        for (j=0;j<output_len;j++)
            acch += Wh[i*output_len + j]*prev_state[j];
        acch =  FMUL32x16(acch, r[i]); // Q22*Q15=Q22
        out_S32[i] = SLIMIT((acci >> outi_right_shift) + (acch >> outh_right_shift), 15+4); //Q3.15
    }
    JMPNN_apply_activation_S16_generic(output, out_S32, output_len, actType);
}


void JMPNN_apply_activation_S16_generic(int16_t * __restrict__ output, int32_t * __restrict__ input, JMPDSP_Length N, JMPNN_ActivationType actType)
{
    if (actType == ACTIVATION_SIGMOID)
    {
        vec_sigmoid_S16(output, input, N);
    }
    else if (actType == ACTIVATION_TANH)
    {
        vec_tanh_S16(output, input, N);
    }
    else if (actType == ACTIVATION_RELU)
    {
        vec_relu_S16(output, input, N);
    }
    else
    {
        // ERROR
    }
    
}

void JMPNN_vec_interpolation_S16_generic(int16_t *output, const int16_t *interp_vec,
                                 const int16_t *input1, const int16_t *input2,
                                 JMPDSP_Length N)
{
    int sum;
    int i;
    for (i=0;i<N;i++)
    {
        sum = interp_vec[i]*input1[i] + (0x7FFF-interp_vec[i])*input2[i];
        //output[i] = SLIMIT(sum >> 15, 16); // For the convex combination, we don't need this.
        output[i] = sum >> 15;
    }
}
