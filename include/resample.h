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
//  resample.h
//
#ifndef RESAMPLE_H
#define RESAMPLE_H

#include <stdint.h>

void upsample_S16(const int16_t* input, int16_t* output, unsigned int input_size, int16_t* last_sample);
void downsample_S16(const int16_t* input, int16_t* output, unsigned int output_size);

void upsample_F32(const float* input, float* output, unsigned int input_size, float* last_sample);
void downsample_F32(const float* input, float* output, unsigned int output_size);

#endif