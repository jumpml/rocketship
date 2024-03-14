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
//  jumpml_nr.h
//

#ifndef _JUMPML_NR_H_
#define _JUMPML_NR_H_

#include "noise_reduction.h"
#include <stdint.h>
#include "signalsifter_config.h"

#define JUMPML_NR_FRAME_SIZE HOP_LENGTH   // Do not change.

typedef struct stru_dsp_jmpnr_st
{
    int NRState_buf[((NR_STATE_SIZE_BYTES)>>2)] __attribute__((aligned(16)));
    uint32_t frameCount;
    NoiseReductionStatePtr NR_Ptr;
} DSP_JMPNR_ST_STRU;

uint32_t jumpml_nr_init(void *jmpnr_st_ptr, float naturalness, float min_gain);
uint32_t jumpml_nr_proc(int16_t *output, int16_t *input, int32_t N, void *jmpnr_st_ptr);
void run_jumpml_nr_prediction(int16_t *output, int16_t *input, NoiseReductionStatePtr NRst_Ptr);

#endif /* _JUMPML_NR_H_ */

