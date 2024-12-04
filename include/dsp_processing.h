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
//  dsp_processing.h
//
//  Created by Raghavendra Prabhu on 8/25/21.
//
#ifndef dsp_processing_h
#define dsp_processing_h

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "kiss_fftr.h"
#include "signalsifter_config.h"

#define KISSFFT_SUBSIZE   (272 + sizeof(kiss_fft_cpx)*((FFT_SIZE>>1) - 1))
#define KISSFFT_MEMNEEDED KISSFFT_SUBSIZE + 24 + sizeof(kiss_fft_cpx)*((FFT_SIZE>>1) * 3/2)


typedef struct STFT {
    unsigned int NFFT;
    unsigned int NFFT_by_2;
    unsigned int numBins;
//    unsigned int R;    // Hop-length
    float FFTscale;
    float IFFTscale;
    //KISS_FFT
    kiss_fftr_cfg kissFFT_cfg;
    kiss_fftr_cfg kissIFFT_cfg;
    kiss_fft_cpx Xk[NUM_BINS] __attribute__((aligned(16)));
    kiss_fft_cpx Yk[NUM_BINS] __attribute__((aligned(16)));

    float window[FFT_SIZE] __attribute__((aligned(16)));
    float inputAux[FFT_SIZE] __attribute__((aligned(16)));
    float outputAux[FFT_SIZE] __attribute__((aligned(16)));
    float Xmag[NUM_BINS] __attribute__((aligned(16)));

    char kissFFTmem[KISSFFT_MEMNEEDED] __attribute__((aligned(16)));
    char kissIFFTmem[KISSFFT_MEMNEEDED] __attribute__((aligned(16)));

  } STFTStruct;

void create_stft(STFTStruct *stft);
void destroy_stft(STFTStruct *stft);
void stft_process(STFTStruct *stft, const float *input, unsigned int R);
void mask_process(STFTStruct *stft, float *mask);
void istft_process(STFTStruct *stft, float *output, unsigned int R);

/*
 Perform windowing and FFT on input buffer. (STFT)
 
 The input array is of length stft->R samples (aka frame size). These
 new samples are used to update the buffer of size stft->NFFT. The
 buffer is then multiplied by a window function, before a real
 FFT is performed on it. The NFFT/2 + 1 complex FFT results are
 appropriately scaled and available in stft->X (or) in stft->Xk
 if using KISS_FFT.
 */
void perform_windowed_FFT(STFTStruct *stft, const float *inBuf, unsigned int R);
void perform_OLA_IFFT(STFTStruct *stft, float *out, unsigned int R);
// GENERAL UTILITY FUNCTIONS
void update_buffer(float *buffer, const float *new_input, unsigned int N, unsigned int R);
float calculate_mean(const float *x, unsigned int N);
void compute_magSquared(kiss_fft_cpx *X, float *Xmag, unsigned int num_bins);
void compute_logMag(float *Xmag, float *logMag, unsigned int num_bins, float scaleFactor);
void apply_spectrum_mask(kiss_fft_cpx *Xk, float *mask, kiss_fft_cpx *Yk, unsigned int num_bins);

#endif /* dsp_processing_h */
