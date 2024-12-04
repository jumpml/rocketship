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
//  resample.c
//
#include "resample.h"

void upsample_S16(const int16_t* input, int16_t* output, unsigned int input_size, int16_t* last_sample) {
    int i;
    int32_t temp;

    // First sample
    temp = (int32_t)(*last_sample) + (int32_t)input[0];
    output[0] = (int16_t)((temp + 1) >> 1);  // Round to nearest
    output[1] = input[0];

    // Process the rest
    for (i = 1; i < input_size; i++) {
        temp = (int32_t)input[i-1] + (int32_t)input[i];
        output[i*2] = (int16_t)((temp + 1) >> 1);  // Round to nearest
        output[i*2 + 1] = input[i];  // Round to nearest
    }

    // Store the last sample for the next frame
    *last_sample = input[input_size - 1];
}

void upsample_F32(const float* input, float* output, unsigned int input_size, float* last_sample) {
    int i;
    
    // First sample (interpolate between last_sample and input[0])
    output[0] = (*last_sample + input[0]) * 0.5f;
    output[1] = input[0];
    
    // Process the rest
    for (i = 1; i < input_size; i++) {
        output[i*2]     = (input[i-1] + input[i]) * 0.5f;
        output[i*2 + 1] = input[i];
    }
    
    // Store the last sample for the next frame
    *last_sample = input[input_size - 1];
}

void downsample_S16(const int16_t* input, int16_t* output, unsigned int output_size) {
    int i;
    int32_t temp;

    for (i = 0; i < output_size; i++) {
        temp = (int32_t)input[i*2] + (int32_t)input[i*2 + 1];
        output[i] = (int16_t)((temp + 1) >> 1);  // Round to nearest
    }
}

void downsample_F32(const float* input, float* output, unsigned int output_size) {
    int i;
    
    for (i = 0; i < output_size; i++) {
        // Simple averaging
        output[i] = (input[i*2] + input[i*2 + 1]) * 0.5f;
    }
}


