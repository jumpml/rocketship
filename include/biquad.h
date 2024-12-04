
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
//  biquad.h
//
//  Created 8/20/24.
//
#ifndef _BIQUAD_H_
#define _BIQUAD_H_

typedef enum {
    LOWPASS,
    HIGHPASS,
    BANDPASS,
    NOTCH,
    LOWSHELF,
    HIGHSHELF,
    PEAKINGEQ
} FilterType;

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float z1, z2;
} BiquadFilter;

typedef struct {
    float fc;       // Cutoff/center frequency in Hz
    float fs;       // Sampling frequency in Hz
    float gain_db;  // Gain in dB (only used for shelving/peaking filters)
    float Q;        // Quality factor
    FilterType type; // Type of filter
} BiquadParams;

#define MAX_BIQUADS 8  // Adjust this value as needed

typedef struct {
    BiquadFilter biquads[MAX_BIQUADS];
    int num_biquads;  // Number of biquads in the cascade
} BiquadCascade;



void biquad_init(BiquadFilter *filter, BiquadParams *params);
void biquad_process(BiquadFilter *filter, float *input, float *output, int frame_size);

#endif /* _BIQUAD_H_ */