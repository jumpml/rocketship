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
//  nn_layers_float.c
//
//  Created by Raghavendra Prabhu on 10/12/21.
//  

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>

#include "nn_layers.h"
#include "nnlib_float.h"

void computeLinearLayer(const LinearLayer *layer, float *output, const float *input)
{
    JMPNN_linear_matXvec_S8xF32_F32(layer->weights, input, layer->bias, output,
                                    layer->input_size, layer->hidden_size,
                                    WEIGHTS_SCALE, BIAS_SCALE);
    JMPNN_apply_activation_F32(output, layer->hidden_size, layer->activation);
}


void computeGRULayer(const GRULayer *gru, float *state, const float *input)
{
    float z[MAX_NEURONS]; float r[MAX_NEURONS]; float h[MAX_NEURONS];
    
    int Ni = gru->hidden_size * gru->input_size;
    int Nh = gru->hidden_size * gru->hidden_size;
    int N = gru->hidden_size;

    JMPNN_gru_matXvec_S8xF32_F32_act(gru->input_weights, input, gru->bias,
                                     gru->recurrent_weights, state, gru->recurrent_bias,
                                     r, gru->input_size, gru->hidden_size,
                                     WEIGHTS_SCALE, BIAS_SCALE, ACTIVATION_SIGMOID);
    
    JMPNN_gru_matXvec_S8xF32_F32_act(&gru->input_weights[Ni], input, &gru->bias[N],
                                     &gru->recurrent_weights[Nh], state, &gru->recurrent_bias[N],
                                     z, gru->input_size, gru->hidden_size,
                                     WEIGHTS_SCALE, BIAS_SCALE, ACTIVATION_SIGMOID);

    JMPNN_gru_newGate_S8xF32_F32_act(&gru->input_weights[2*Ni], input, &gru->bias[2*N],
                                     &gru->recurrent_weights[2*Nh], state, &gru->recurrent_bias[2*N], r,
                                     h, gru->input_size, gru->hidden_size,
                                     WEIGHTS_SCALE, BIAS_SCALE, gru->activation);
    JMPNN_vec_interpolation_F32(state, z, state, h, gru->hidden_size);
                                       

}
