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
//  signalsifter.c
//
//  Created by Raghavendra Prabhu on 10/12/21.
//

#include "signalsifter.h"
#include <stdlib.h>
#include "utils.h"
#include <stdio.h>
#include <float.h>
#include <string.h>
extern const SignalSifterModel ss_model;

void createSignalSifterModel(SignalSifterState *ss)
{
    ss->model = &ss_model;
    JMPDSP_vclr(ss->gru1_gru_state, 1, ss->model->gru1_hidden_size);
    JMPDSP_vclr(ss->gru2_gru_state, 1, ss->model->gru2_hidden_size);
    JMPDSP_vclr(ss->gru3_gru_state, 1, ss->model->gru3_hidden_size);
#if ENABLE_SS_MONITOR
    ss->monitor = calloc(1, sizeof(SignalSifterMonitor));
    initLayerStats(&ss->monitor->input_stats);
    initLayerStats(&ss->monitor->gru1_stats);
    initLayerStats(&ss->monitor->gru2_stats);
    initLayerStats(&ss->monitor->gru3_stats);
    initLayerStats(&ss->monitor->linear1_stats);
#endif
#if ENABLE_QUANTIZATION_SIM
    ss->SS_qparams = calloc(1, sizeof(SignalSifterQParams));
    initLayerQParams(&ss->SS_qparams->input_qparams, INPUT_NUM_FRAC_BITS, INPUT_NUMBITS, INPUT_ENABLE_ROUNDING);
    initLayerQParams(&ss->SS_qparams->gru1_qparams, GRU_NUM_FRAC_BITS, GRU_NUMBITS,GRU_ENABLE_ROUNDING);
    initLayerQParams(&ss->SS_qparams->gru2_qparams, GRU_NUM_FRAC_BITS, GRU_NUMBITS,GRU_ENABLE_ROUNDING);
    initLayerQParams(&ss->SS_qparams->gru3_qparams, GRU_NUM_FRAC_BITS, GRU_NUMBITS,GRU_ENABLE_ROUNDING);
    initLayerQParams(&ss->SS_qparams->linear1_qparams, LIN_NUM_FRAC_BITS, LIN_NUMBITS,LIN_ENABLE_ROUNDING);
#endif
}

void createSignalSifterModel_S16(SignalSifterState_S16 *ss)
{
    ss->model = &ss_model;
    JMPDSP_vclr_S16(ss->gru1_gru_state, 1, ss->model->gru1_hidden_size);
    JMPDSP_vclr_S16(ss->gru2_gru_state, 1, ss->model->gru2_hidden_size);
    JMPDSP_vclr_S16(ss->gru3_gru_state, 1, ss->model->gru3_hidden_size);
}

void destroySignalSifterModel(SignalSifterState *ss)
{
#if ENABLE_SS_MONITOR
    free(ss->monitor);
#endif
#if ENABLE_QUANTIZATION_SIM
    free(ss->SS_qparams);
#endif
}

void destroySignalSifterModel_S16(SignalSifterState_S16 *ss)
{

}

void computeSignalSifterModel_simS16(SignalSifterState *ss, float *gains, const float *input)
{
    float quantized_input[MAX_NEURONS];
    memcpy(quantized_input, input, ss->model->gru1_gru->input_size * sizeof(float));
    applyLayerQuantization(&ss->SS_qparams->input_qparams, quantized_input, 1, ss->model->gru1_gru->input_size);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->input_stats, quantized_input, 1, ss->model->gru1_gru->input_size);
#endif
    computeGRULayer(ss->model->gru1_gru, ss->gru1_gru_state, quantized_input);
    applyLayerQuantization(&ss->SS_qparams->gru1_qparams, ss->gru1_gru_state, 1, ss->model->gru1_gru->hidden_size);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->gru1_stats, ss->gru1_gru_state, 1, ss->model->gru1_gru->hidden_size);
#endif
    computeGRULayer(ss->model->gru2_gru, ss->gru2_gru_state, ss->gru1_gru_state);
    applyLayerQuantization(&ss->SS_qparams->gru2_qparams, ss->gru2_gru_state, 1, ss->model->gru2_gru->hidden_size);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->gru2_stats, ss->gru2_gru_state, 1, ss->model->gru2_gru->hidden_size);
#endif
    computeGRULayer(ss->model->gru3_gru, ss->gru3_gru_state, ss->gru2_gru_state);
    applyLayerQuantization(&ss->SS_qparams->gru3_qparams, ss->gru3_gru_state, 1, ss->model->gru3_gru->hidden_size);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->gru3_stats, ss->gru3_gru_state, 1, ss->model->gru3_gru->hidden_size);
#endif
    computeLinearLayer(ss->model->linear1_linear, gains, ss->gru3_gru_state);
    applyLayerQuantization(&ss->SS_qparams->linear1_qparams, gains, 1,  ss->model->linear1_hidden_size);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->linear1_stats, gains, 1, ss->model->linear1_hidden_size);
#endif
}

void computeSignalSifterModel(SignalSifterState *ss, float *gains, const float *input)
{
#if ENABLE_QUANTIZATION_SIM
    computeSignalSifterModel_simS16(ss, gains, input);
#else
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->input_stats, input, 1, ss->model->gru1_gru->input_size);
#endif
    computeGRULayer(ss->model->gru1_gru, ss->gru1_gru_state, input);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->gru1_stats, ss->gru1_gru_state, 1, ss->model->gru1_gru->hidden_size);
#endif
    computeGRULayer(ss->model->gru2_gru, ss->gru2_gru_state, ss->gru1_gru_state);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->gru2_stats, ss->gru2_gru_state, 1, ss->model->gru2_gru->hidden_size);
#endif
    computeGRULayer(ss->model->gru3_gru, ss->gru3_gru_state, ss->gru2_gru_state);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->gru3_stats, ss->gru3_gru_state, 1, ss->model->gru3_gru->hidden_size);
#endif
    computeLinearLayer(ss->model->linear1_linear, gains, ss->gru3_gru_state);
#if ENABLE_SS_MONITOR
    computeLayerStats(&ss->monitor->linear1_stats, gains, 1, ss->model->linear1_hidden_size);
#endif
#endif 
}

void computeSignalSifterModel_S16(SignalSifterState_S16 *ss, int16_t *gains, const int16_t *input)
{
    computeGRULayer_S16(ss->model->gru1_gru, ss->gru1_gru_state, input,
                        INPUT_NUM_FRAC_BITS, INPUTLAYER_SHIFT_RIGHT, GRU_NUM_FRAC_BITS, WAB_FRAC_BITS);
    computeGRULayer_S16(ss->model->gru2_gru, ss->gru2_gru_state, ss->gru1_gru_state,
                        GRU_NUM_FRAC_BITS, WAB_FRAC_BITS, GRU_NUM_FRAC_BITS, WAB_FRAC_BITS);
    computeGRULayer_S16(ss->model->gru3_gru, ss->gru3_gru_state, ss->gru2_gru_state,
                        GRU_NUM_FRAC_BITS, WAB_FRAC_BITS, GRU_NUM_FRAC_BITS, WAB_FRAC_BITS);
    computeLinearLayer_S16(ss->model->linear1_linear, gains, ss->gru3_gru_state,
                           LIN_NUM_FRAC_BITS, WAB_FRAC_BITS);
}


#if ENABLE_SS_MONITOR
void printSignalSifterStats(SignalSifterMonitor *sm)
{
    print_layer_stats("Input", sm->input_stats.min, sm->input_stats.max, sm->input_stats.mean, sm->input_stats.std);
    print_layer_stats("GRU1", sm->gru1_stats.min, sm->gru1_stats.max, sm->gru1_stats.mean, sm->gru1_stats.std);
    print_layer_stats("GRU2", sm->gru2_stats.min, sm->gru2_stats.max, sm->gru2_stats.mean, sm->gru2_stats.std);
    print_layer_stats("GRU3", sm->gru3_stats.min, sm->gru3_stats.max, sm->gru3_stats.mean, sm->gru3_stats.std);
    print_layer_stats("Output", sm->linear1_stats.min, sm->linear1_stats.max, sm->linear1_stats.mean, sm->linear1_stats.std);
}

void printSignalSifterStats_S16(SignalSifterMonitor_S16 *sm)
{
    print_layer_stats_S16("Input", sm->input_stats.min, sm->input_stats.max, sm->input_stats.mean, sm->input_stats.std);
    print_layer_stats_S16("GRU1", sm->gru1_stats.min, sm->gru1_stats.max, sm->gru1_stats.mean, sm->gru1_stats.std);
    print_layer_stats_S16("GRU2", sm->gru2_stats.min, sm->gru2_stats.max, sm->gru2_stats.mean, sm->gru2_stats.std);
    print_layer_stats_S16("GRU3", sm->gru3_stats.min, sm->gru3_stats.max, sm->gru3_stats.mean, sm->gru3_stats.std);
    print_layer_stats_S16("Output", sm->linear1_stats.min, sm->linear1_stats.max, sm->linear1_stats.mean, sm->linear1_stats.std);
}
#endif

