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
//  jumpml_nr.c
//
//  Created by Raghavendra Prabhu on 5/2/22.
//
#include "jumpml_nr.h"
#include "jumpml_nr_tuning.h"
#include "common_def.h"
#include "resample.h"
#include "biquad.h"

uint32_t jumpml_nr_init(void *jmpnr_st_ptr, float naturalness, float min_gain)
{
    DSP_JMPNR_ST_STRU *NRst = (DSP_JMPNR_ST_STRU *) jmpnr_st_ptr;
    NRst->frameCount = 0;
    NRst->last_sample = 0.0f;
    NRst->NR_Ptr = (NoiseReductionStatePtr) NRst->NRState_buf;

    NRst->hsparams.fc = JUMPML_NR_OUTPUT_HIGHSHELF_FC;
    NRst->hsparams.fs = JUMPML_NR_OUTPUT_HIGHSHELF_FS;  
    NRst->hsparams.gain_db = JUMPML_NR_OUTPUT_HIGHSHELF_GAIN;
    NRst->hsparams.Q = JUMPML_NR_OUTPUT_HIGHSHELF_Q;
    NRst->hsparams.type = HIGHSHELF;
    biquad_init(&NRst->hsfilter, &NRst->hsparams);

    create_noise_reduction(NRst->NR_Ptr, naturalness, min_gain);
    return 0;
}

void run_jumpml_nr_prediction(int16_t *output, int16_t *input, NoiseReductionStatePtr NRst_Ptr, BiquadFilter* hsf)
{
    float input_frame[JUMPML_NR_FRAME_SIZE] __attribute__((aligned(16)));
    float output_frame[JUMPML_NR_FRAME_SIZE] __attribute__((aligned(16)));
    int i;
    for (i=0;i<JUMPML_NR_FRAME_SIZE;i++)
    {
        input_frame[i] = ((float) input[i]) / 32768.0f;
#if JUMPML_NR_APPLY_INPUT_GAIN
        input_frame[i] = MAX(MIN(input_frame[i] * JUMPML_NR_INPUT_MIC_GAIN, 0.9999),-1.0);
#endif
    }
    noise_reduction_process(NRst_Ptr, input_frame, output_frame, JUMPML_NR_FRAME_SIZE);
#if JUMPML_NR_APPLY_HIGHSHELF
    biquad_process(hsf, output_frame, output_frame, JUMPML_NR_FRAME_SIZE);
#endif
    for (i=0;i<JUMPML_NR_FRAME_SIZE;i++)
    {
#if JUMPML_NR_APPLY_OUTPUT_GAIN
        output_frame[i] = MAX(MIN(output_frame[i] * JUMPML_NR_OUTPUT_GAIN, 0.9999),-1.0);
#endif
        output[i] = ((short)(output_frame[i] * INT16_MAX));
    }
    
}

uint32_t jumpml_nr_proc(int16_t *output, int16_t *input, void *jmpnr_st_ptr, int sr)
{
    DSP_JMPNR_ST_STRU *NRst = (DSP_JMPNR_ST_STRU *) jmpnr_st_ptr;
    int16_t resampled_input[JUMPML_NR_FRAME_SIZE*2];
    int16_t resampled_output[JUMPML_NR_FRAME_SIZE*2];

    if (sr == 8000){
        upsample_S16(input, resampled_input, JUMPML_NR_FRAME_SIZE, &(NRst->last_sample));
        run_jumpml_nr_prediction(&resampled_output[0], &resampled_input[0], NRst->NR_Ptr, &NRst->hsfilter);
        run_jumpml_nr_prediction(&resampled_output[JUMPML_NR_FRAME_SIZE], &resampled_input[JUMPML_NR_FRAME_SIZE], NRst->NR_Ptr, &NRst->hsfilter);
        downsample_S16(resampled_output, output, JUMPML_NR_FRAME_SIZE);
    }
    else{
        run_jumpml_nr_prediction(output, input, NRst->NR_Ptr, &NRst->hsfilter);
    }
    return 0;
}
