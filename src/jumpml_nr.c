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
//  jumpml_nr.c
//
//  Created by Raghavendra Prabhu on 5/2/22.
//
#include "jumpml_nr.h"
#include "jumpml_nr_tuning.h"
#include "common_def.h"

uint32_t jumpml_nr_init(void *jmpnr_st_ptr, float naturalness, float min_gain)
{
    DSP_JMPNR_ST_STRU *NRst = (DSP_JMPNR_ST_STRU *) jmpnr_st_ptr;
    NRst->frameCount = 0;
    NRst->NR_Ptr = (NoiseReductionStatePtr) NRst->NRState_buf;
    create_noise_reduction(NRst->NR_Ptr, naturalness, min_gain);
    return 0;
}

void run_jumpml_nr_prediction(int16_t *output, int16_t *input, NoiseReductionStatePtr NRst_Ptr)
{
    float input_frame[JUMPML_NR_FRAME_SIZE] __attribute__((aligned(16)));
    float output_frame[JUMPML_NR_FRAME_SIZE] __attribute__((aligned(16)));
    int i;
    for (i=0;i<JUMPML_NR_FRAME_SIZE;i++)
    {
        input_frame[i] = ((float) input[i]) / 32768.0f;
#if XCHAL_HAVE_HIFI5
        input_frame[i] = MAX(MIN(input_frame[i] * JUMPML_NR_INPUT_MIC_GAIN, 0.9999),-1.0);
#endif
    }
    noise_reduction_process(NRst_Ptr, input_frame, output_frame, JUMPML_NR_FRAME_SIZE);
    for (i=0;i<JUMPML_NR_FRAME_SIZE;i++)
    {
#if XCHAL_HAVE_HIFI5
        output_frame[i] = MAX(MIN(output_frame[i] * JUMPML_NR_OUTPUT_GAIN, 0.9999),-1.0);
#endif
        output[i] = ((short)(output_frame[i] * INT16_MAX));
    }
    
}

uint32_t jumpml_nr_proc(int16_t *output, int16_t *input, int32_t N, void *jmpnr_st_ptr)
{
    DSP_JMPNR_ST_STRU *NRst = (DSP_JMPNR_ST_STRU *) jmpnr_st_ptr;
    int i;
    
    switch (N)
    {
        case JUMPML_NR_FRAME_SIZE:  // 120
            run_jumpml_nr_prediction(output, input, NRst->NR_Ptr);
            break;
            
        case 2*JUMPML_NR_FRAME_SIZE:  //240
        case 480:                     //480 where it is really 240, with 240 samples ignored.
            run_jumpml_nr_prediction(&output[0], &input[0], NRst->NR_Ptr);
            run_jumpml_nr_prediction(&output[JUMPML_NR_FRAME_SIZE], &input[JUMPML_NR_FRAME_SIZE], NRst->NR_Ptr);
            break;
            
        default:
            for (i=0; i<N; i++)
            {
                output[i] = input[i];
            }
            break;
    }
    
    return 0;
}
