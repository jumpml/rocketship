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
//  signalsifter_monitor.h
//
//  Created by Raghavendra Prabhu on 10/12/21.
//
#ifndef SIGNALSIFTER_MONITOR_H
#define SIGNALSIFTER_MONITOR_H

#include "nn_layers_tools.h"

struct SignalSifterMonitor {
    LayerStats input_stats;
    LayerStats gru1_stats;
    LayerStats gru2_stats;
    LayerStats gru3_stats;
    LayerStats linear1_stats;
};
typedef struct SignalSifterMonitor SignalSifterMonitor;

struct SignalSifterMonitor_S16 {
    LayerStats_S16 input_stats;
    LayerStats_S16 gru1_stats;
    LayerStats_S16 gru2_stats;
    LayerStats_S16 gru3_stats;
    LayerStats_S16 linear1_stats;
};
typedef struct SignalSifterMonitor_S16 SignalSifterMonitor_S16;

struct SignalSifterQParams {
    LayerQParams input_qparams;
    LayerQParams gru1_qparams;
    LayerQParams gru2_qparams;
    LayerQParams gru3_qparams;
    LayerQParams linear1_qparams;
};
typedef struct SignalSifterQParams SignalSifterQParams;

#endif /* SIGNALSIFTER_MONITOR_H */
