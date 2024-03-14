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
//  nnlib_fixedpt.h
//
//  Created by Raghavendra Prabhu on 2/22/24.
//

#ifndef NNLIB_FIXEDPT_H
#define NNLIB_FIXEDPT_H

#include "nnlib_types.h"
#include "tanh_table_S16.h"
#include <math.h>
#include "common_def.h"
#include "fixed_point_math.h"
#include "signalsifter_config.h"


#if XCHAL_HAVE_HIFI5 
#define JMPNN_linear_matXvec_S8xS16_S32  JMPNN_linear_matXvec_S8xS16_S32_hifi5
#define JMPNN_gru_matXvec_S8xS16_S16_act JMPNN_gru_matXvec_S8xS16_S16_act_hifi5
#define JMPNN_gru_newGate_S8xS16_S16_act JMPNN_gru_newGate_S8xS16_S16_act_hifi5
#define JMPNN_apply_activation_S16       JMPNN_apply_activation_S16_generic
#define JMPNN_vec_interpolation_S16      JMPNN_vec_interpolation_S16_generic
#else
#define JMPNN_linear_matXvec_S8xS16_S32  JMPNN_linear_matXvec_S8xS16_S32_generic
#define JMPNN_gru_matXvec_S8xS16_S16_act JMPNN_gru_matXvec_S8xS16_S16_act_generic
#define JMPNN_gru_newGate_S8xS16_S16_act JMPNN_gru_newGate_S8xS16_S16_act_generic
#define JMPNN_apply_activation_S16       JMPNN_apply_activation_S16_generic
#define JMPNN_vec_interpolation_S16      JMPNN_vec_interpolation_S16_generic
#endif

static inline int16_t tanh_approx_S16(int32_t x_Q16_15)
{
    int i;
    int32_t x = x_Q16_15; // Actually Q3.15
    int32_t y, dy, one_xy;
    int32_t one_Q31 = (unsigned int)(1<<31) - 1;
    int16_t half_Q15 = (1<<14);
    int32_t temp;
    int sign=1;
    if (x<0)
    {
        x=-x;
        sign=-1;
    }
    i = (int)(((int32_t)half_Q15 + TANH_SCALEFAC_S16*x)>>15); // Q15 + INT * Q3.15 = Q3+.15. Retain integer part.
    i = MAX(0, MIN(TANH_TABLE_MAXINDEX_S16,i));
    temp = IMUL(TANH_DELTAX_S16,i);
    x = x - IMUL(TANH_DELTAX_S16,i );  // Q3.15 - Q15 * INT = Q15 (should have no integer part)
    y = tanh_table_S16[i];            // Q15
    temp = IMUL(y, y) << 1;           // Q31
    dy = one_Q31 - temp;              // Q31
    one_xy = IMUL(y,x);               // Q1.30
    one_xy = (one_Q31>>1) - one_xy;   // Q1.30
    temp = FMUL32x16(x, dy);          // Q31
    one_xy = FMUL32x32(one_xy, temp);
    y = y  + (one_xy >> 15);          //Q15

    return sign * y;
}

static inline int16_t sigmoid_approx_S16(int32_t x_Q15_16)
{
    return (0x4000 + (tanh_approx_S16(x_Q15_16>>1) >> 1));
}

static inline int16_t relu_S16(int32_t x)
{
    return x < 0 ? 0 : x;
}

static inline void vec_tanh_S16(int16_t *output, const int32_t *input, JMPDSP_Length N)
{
    int i;
    for (i=0;i<N;i++)
    {
        output[i] = tanh_approx_S16(input[i]);
    }
}


static inline void vec_sigmoid_S16(int16_t *output, const int32_t *input, JMPDSP_Length N)
{
    int i;
    for (i=0;i<N;i++)
    {
        output[i] = sigmoid_approx_S16(input[i]);
    }
}

static inline void vec_relu_S16(int16_t *output, const int32_t *input, JMPDSP_Length N)
{
    int i;
    for (i=0;i<N;i++)
    {
        output[i] = relu_S16(input[i]);
    }
}

void JMPNN_linear_matXvec_S8xS16_S32_generic(const int8_t *W, const int16_t *input, const int8_t *bias,
                                     int32_t *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                     uint16_t bias_left_shift, uint16_t output_right_shift);


void JMPNN_gru_matXvec_S8xS16_S16_act_generic(const int8_t *Wi, const int16_t *input, const int8_t *Bi,
                                      const int8_t *Wh, const int16_t *prev_state, const int8_t *Bh,
                                      int16_t *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      uint16_t Bi_left_shift, uint16_t outi_right_shift, uint16_t Bh_left_shift, uint16_t outh_right_shift,
                                      JMPNN_ActivationType actType); // used for reset/update gates

void JMPNN_gru_newGate_S8xS16_S16_act_generic(const int8_t *Wi, const int16_t *input, const int8_t *Bi,
                                      const int8_t *Wh, const int16_t *prev_state, const int8_t *Bh, const int16_t *r,
                                      int16_t *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      uint16_t Bi_left_shift, uint16_t outi_right_shift, uint16_t Bh_left_shift, uint16_t outh_right_shift,
                                      JMPNN_ActivationType actType);



void JMPNN_apply_activation_S16_generic(int16_t *output, int32_t *input, JMPDSP_Length N, JMPNN_ActivationType actType);

void JMPNN_vec_interpolation_S16_generic(int16_t *output, const int16_t *interp_vec,
                                 const int16_t *input1, const int16_t *input2,
                                 JMPDSP_Length len);

#if XCHAL_HAVE_HIFI5 
void JMPNN_linear_matXvec_S8xS16_S32_hifi5(const int8_t *W, const int16_t *input, const int8_t *bias,
                                     int32_t *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                     uint16_t bias_left_shift, uint16_t output_right_shift);


void JMPNN_gru_matXvec_S8xS16_S16_act_hifi5(const int8_t *Wi, const int16_t *input, const int8_t *Bi,
                                      const int8_t *Wh, const int16_t *prev_state, const int8_t *Bh,
                                      int16_t *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      uint16_t Bi_left_shift, uint16_t outi_right_shift, uint16_t Bh_left_shift, uint16_t outh_right_shift,
                                      JMPNN_ActivationType actType); // used for reset/update gates

void JMPNN_gru_newGate_S8xS16_S16_act_hifi5(const int8_t *Wi, const int16_t *input, const int8_t *Bi,
                                      const int8_t *Wh, const int16_t *prev_state, const int8_t *Bh, const int16_t *r,
                                      int16_t *output, JMPDSP_Length input_len, JMPDSP_Length output_len,
                                      uint16_t Bi_left_shift, uint16_t outi_right_shift, uint16_t Bh_left_shift, uint16_t outh_right_shift,
                                      JMPNN_ActivationType actType);
#endif

#endif /* NNLIB_FIXEDPT_H */
