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
//  nn_layers.h
//
//  Created by Raghavendra Prabhu on 10/12/21.
//
#ifndef NN_LAYERS_H_
#define NN_LAYERS_H_

#include "signalsifter_config.h"
#include "dsplib.h"

typedef struct {
    const nnBias *bias;
    const nnWeight *weights;
    int input_size;
    int hidden_size;
    int activation;
} LinearLayer;

typedef struct {
    const nnBias *bias;
    const nnBias *recurrent_bias;
    const nnWeight *input_weights;
    const nnWeight *recurrent_weights;
    int input_size;
    int hidden_size;
    int activation;
} GRULayer;

void computeLinearLayer(const LinearLayer *layer, float *output, const float *input);
void computeLinearLayer_S16(const LinearLayer *layer, int16_t *output, const int16_t *input,
                            uint16_t bias_left_shift, uint16_t output_right_shift);
void computeGRULayer(const GRULayer *gru, float *state, const float *input);
void computeGRULayer_S16(const GRULayer *gru, int16_t *state, const int16_t *input,
                         uint16_t Bi_left_shift, uint16_t outi_right_shift,
                         uint16_t Bh_left_shift, uint16_t outh_right_shift);
#endif /* NN_LAYERS_H_ */
