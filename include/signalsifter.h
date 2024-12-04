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
//  signalsifter.h
//
//  Created by Raghavendra Prabhu on 10/12/21.
//
#ifndef SIGNALSIFTER_H
#define SIGNALSIFTER_H

#include "nn_layers.h"
#include "signalsifter_config.h"
#include "nnlib_types.h"
#include "signalsifter_monitor.h"

#define ENABLE_SS_MONITOR 0           //Set to 1 to enable Layer Statistics Monitoring
#define ENABLE_QUANTIZATION_SIM 0     //Set to 1 to enabled simulated quantization on F32 inference

#define INPUT_ENABLE_ROUNDING 1
#define GRU_ENABLE_ROUNDING 0
#define LIN_ENABLE_ROUNDING 1

struct SignalSifterModel {
    int gru1_hidden_size;
    const GRULayer *gru1_gru;
    
    int gru2_hidden_size;
    const GRULayer *gru2_gru;
    
    int gru3_hidden_size;
    const GRULayer *gru3_gru;
    
    int linear1_hidden_size;
    const LinearLayer *linear1_linear;
};
typedef struct SignalSifterModel SignalSifterModel;

struct SignalSifterState {
    const SignalSifterModel *model;
    float gru1_gru_state[GRU_STATE_SIZE];
    float gru2_gru_state[GRU_STATE_SIZE];
    float gru3_gru_state[GRU_STATE_SIZE];
#if ENABLE_SS_MONITOR
    SignalSifterMonitor *monitor;
#endif
    SignalSifterQParams *SS_qparams;
};
typedef struct SignalSifterState SignalSifterState;

struct SignalSifterState_S16 {
    const SignalSifterModel *model;
    int16_t gru1_gru_state[GRU_STATE_SIZE] __attribute__((aligned(16)));
    int16_t gru2_gru_state[GRU_STATE_SIZE] __attribute__((aligned(16)));
    int16_t gru3_gru_state[GRU_STATE_SIZE] __attribute__((aligned(16)));
#if ENABLE_SS_MONITOR
    SignalSifterMonitor_S16 *monitor;
#endif
};
typedef struct SignalSifterState_S16 SignalSifterState_S16;

void createSignalSifterModel(SignalSifterState *ss);
void destroySignalSifterModel(SignalSifterState *ss);
void computeSignalSifterModel(SignalSifterState *ss, float *gains, const float *input);

//QUANTIZED INFERENCE
void computeSignalSifterModel_simS16(SignalSifterState *ss, float *gains, const float *input);

void createSignalSifterModel_S16(SignalSifterState_S16 *ss);
void destroySignalSifterModel_S16(SignalSifterState_S16 *ss);
void computeSignalSifterModel_S16(SignalSifterState_S16 *ss, int16_t *gains, const int16_t *input);

#if ENABLE_SS_MONITOR
void printSignalSifterStats(SignalSifterMonitor *sm);
void printSignalSifterStats_S16(SignalSifterMonitor_S16 *sm);
#endif
#endif /* SIGNALSIFTER_H */
