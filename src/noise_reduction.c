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
//  noise_reduction.c
//
//  Created by Raghavendra Prabhu on 10/12/21.
//

#include "noise_reduction.h"
#include "utils.h"
#include <assert.h>
#include <time.h>
#include <syslog.h>

void create_noise_reduction(NoiseReductionState *nr, float naturalness, float min_gain)
{
// #if PRINT_JUMPML_NR_MEMORY_STATS 
//     printf("NR State Memory Requirements = %lu bytes\n", NR_STATE_SIZE);
// #endif
    create_stft(&nr->STFT);
#ifdef USE_FLOAT32_SIGNALSIFTER
    createSignalSifterModel(&nr->SS);
#else
    createSignalSifterModel_S16(&nr->SS);
#endif
//    for (int i=0; i < nr->STFT.numBins; i++)
//        nr->gains[i] = 1.0f;
    // Tuning parameter
    nr->alphaReverb = NR_ALPHA_REVERB;
    nr->alphaLowLev = naturalness;
    nr->minGain = min_gain;
    nr->gainBoost = NR_GAIN_BOOST;
    nr->speechBandStartBin =  SPEECH_BAND_START / FREQ_RESOLUTION;
    nr->speechBandEndBin = SPEECH_BAND_END / FREQ_RESOLUTION;
}

void destroy_noise_reduction(NoiseReductionState *nr)
{
    destroy_stft(&nr->STFT);
#ifdef USE_FLOAT32_SIGNALSIFTER
    destroySignalSifterModel(&nr->SS);
#else
    destroySignalSifterModel_S16(&nr->SS);
#endif
}

/* Computes a more aggressive gain function and then conditionally adds
   reverb or restores lower level gain to preserve high-frequency speech.
 gainsIn: raw gains from SignalSifter NN model
 gainsState: contains gains applied in prior frame
 alpha: tuning parameter than controls the reverb (and lower level gains)
 */
void postprocess_gains(float *gainsIn, float *gainsState, int numBins, NoiseReductionState *nr)
{
    float gain, gain_reverb, gain_lower_level;
    int i;

    for (i=0; i<numBins; i++)
    {
        gain = gainsIn[i] * sinf(M_PI_2 * gainsIn[i]);
        
        if (ENABLE_SPEECH_BOOST)
        {
            if (nr->speechBandStartBin <= i <= nr->speechBandEndBin)
                gain = gain * (gain > 0.25f ? NR_SPEECH_GAIN:1.0f);
        }
            
        gain_reverb = nr->alphaReverb * gainsState[i];
        gain = fmaxf(gain, gain_reverb);
        
        gain_lower_level = nr->alphaLowLev * gainsIn[i];
        gain = fmaxf(gain, gain_lower_level);
        
        gainsState[i] = fmaxf(fminf(gain * nr->gainBoost, 1.0f), nr->minGain);
    }
}

void noise_reduction_process(NoiseReductionState *nr, const float *input, float *output, unsigned int R)
{
    float gains[NUM_BINS] __attribute__((aligned(16)));

#if defined(ENABLE_PROFILING)
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif 

#ifndef USE_FLOAT32_SIGNALSIFTER
    int16_t gains_S16[NUM_BINS], Xmag_S16[NUM_BINS];
    JMPDSP_vclr_S16(gains_S16, 1, NUM_BINS);
#endif
    assert(NUM_BINS == nr->STFT.numBins);
    JMPDSP_vclr(gains, 1, NUM_BINS);
    
    stft_process(&nr->STFT, input, R);
#ifdef USE_FLOAT32_SIGNALSIFTER
    computeSignalSifterModel(&nr->SS, gains, nr->STFT.Xmag);
#else
    convert_F32toS16(nr->STFT.Xmag, Xmag_S16, NUM_BINS, INPUT_NUM_FRAC_BITS);
    computeSignalSifterModel_S16(&nr->SS, gains_S16, Xmag_S16);
    convert_S16toF32(gains_S16, gains, NUM_BINS, GRU_NUM_FRAC_BITS);
#endif
    postprocess_gains(gains, nr->gains, nr->STFT.numBins, nr);
//    print_vector(nr->gains, nr->STFT.numBins, "NR Gains");
    mask_process(&nr->STFT, nr->gains);
    istft_process(&nr->STFT, output, R);
    
#if defined(ENABLE_PROFILING)
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    syslog(LOG_WARNING, "noise reduction (took %f ms) \n",time_elapsed_s * 1000);
#endif

}


void noise_reduction_monitor(NoiseReductionState *nr)
{
#ifdef USE_FLOAT32_SIGNALSIFTER
#if ENABLE_SS_MONITOR
    printSignalSifterStats(nr->SS.monitor);
#endif
#endif
}


