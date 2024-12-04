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
//  nn_layers_fixedpt.c
//
//  Created by Raghavendra Prabhu on 10/12/21.
//  

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>

#include "nn_layers.h"
#include "nnlib_fixedpt.h"
#include "signalsifter_config.h"

void computeLinearLayer_S16(const LinearLayer *layer, int16_t *output, const int16_t *input,
                            uint16_t bias_left_shift, uint16_t output_right_shift)
{
    int32_t out_S32[MAX_NEURONS];
    
    JMPNN_linear_matXvec_S8xS16_S32(layer->weights, input, layer->bias, out_S32,
                                    layer->input_size, layer->hidden_size,
                                    bias_left_shift, output_right_shift);
    
    JMPNN_apply_activation_S16(output, out_S32, layer->hidden_size, layer->activation);
}

void computeGRULayer_S16(const GRULayer *gru, int16_t *state, const int16_t *input,
                         uint16_t Bi_left_shift, uint16_t outi_right_shift,
                         uint16_t Bh_left_shift, uint16_t outh_right_shift)
{

    int Ni = gru->hidden_size * gru->input_size;
    int Nh = gru->hidden_size * gru->hidden_size;
    int N = gru->hidden_size;
    int16_t z[MAX_NEURONS]; int16_t r[MAX_NEURONS]; int16_t h[MAX_NEURONS];
    
    JMPNN_gru_matXvec_S8xS16_S16_act(gru->input_weights, input, gru->bias,
                                     gru->recurrent_weights, state, gru->recurrent_bias,
                                     r, gru->input_size, gru->hidden_size,
                                     Bi_left_shift, outi_right_shift, Bh_left_shift, outh_right_shift,
                                     ACTIVATION_SIGMOID);
    
    JMPNN_gru_matXvec_S8xS16_S16_act(&gru->input_weights[Ni], input, &gru->bias[N],
                                     &gru->recurrent_weights[Nh], state, &gru->recurrent_bias[N],
                                     z, gru->input_size, gru->hidden_size,
                                     Bi_left_shift, outi_right_shift, Bh_left_shift, outh_right_shift,
                                     ACTIVATION_SIGMOID);
    
    JMPNN_gru_newGate_S8xS16_S16_act(&gru->input_weights[2*Ni], input, &gru->bias[2*N],
                                     &gru->recurrent_weights[2*Nh], state, &gru->recurrent_bias[2*N], r,
                                     h, gru->input_size, gru->hidden_size,
                                     Bi_left_shift, outi_right_shift, Bh_left_shift, outh_right_shift,
                                     gru->activation);
    
    JMPNN_vec_interpolation_S16(state, z, state, h, gru->hidden_size);
 }
