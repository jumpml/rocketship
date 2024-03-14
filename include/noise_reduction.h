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
//  noise_reduction.h
//
//  Created by Raghavendra Prabhu on 10/12/21.
//
#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H

#include "signalsifter_config.h"
#include "dsp_processing.h"
#include "signalsifter.h"

// #define USE_FLOAT32_SIGNALSIFTER

#define NR_ALPHA_REVERB 0.0f
#define NR_ALPHA_LOWLEVEL 0.75f
#define NR_MIN_GAIN 0.0001f
#define NR_GAIN_BOOST 1.0f
#define NR_SPEECH_GAIN 20.0f
#define SPEECH_BAND_START 300
#define SPEECH_BAND_END 3800
#define ENABLE_SPEECH_BOOST 0

#define NR_STATE_SIZE_BYTES sizeof(struct NoiseReduction)

#ifdef USE_FLOAT32_SIGNALSIFTER
#define NRSignalSifterState SignalSifterState
#else
#define NRSignalSifterState SignalSifterState_S16
#endif

struct NoiseReduction {
    // Tuning parameters
    float alphaReverb;
    float alphaLowLev;
    float minGain;
    float gainBoost;
    int speechBandStartBin;
    int speechBandEndBin;
    STFTStruct STFT;
    NRSignalSifterState SS;
    float gains[IO_SIZE] __attribute__((aligned(16)));
};

typedef struct NoiseReduction NoiseReductionState;
typedef struct NoiseReduction* NoiseReductionStatePtr;

void create_noise_reduction(NoiseReductionState *nr, float naturalness, float min_gain);
void destroy_noise_reduction(NoiseReductionState *nr);
void postprocess_gains(float *gainsIn, float *gainsState, int numBins, NoiseReductionState *nr);
void noise_reduction_process(NoiseReductionState *nr, const float *input, float *output, unsigned int R);
void noise_reduction_monitor(NoiseReductionState *nr);

#endif /* NOISE_REDUCTION_H */
