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
//  dsp_processing.c
//
//  Created by Raghavendra Prabhu on 8/25/21.
//

#include "dsp_processing.h"
#include "fixed_point_math.h"
#include "common_def.h"
#include "dsplib.h"
//#include "utils.h"

#define IFFT_SCALE_FACTOR 1.0f/FFT_SIZE
#define TWO_PI_BY_N 2.0f * M_PI/FFT_SIZE

void create_stft(STFTStruct *stft)
{
    int i;
    size_t memneeded;
    stft->NFFT = FFT_SIZE;
//    stft->R = R;
    stft->NFFT_by_2 = FFT_SIZE >> 1;
    stft->numBins = stft->NFFT_by_2 + 1;
    
    stft->FFTscale = 0.5f;
    stft->IFFTscale = IFFT_SCALE_FACTOR;
    //KISS_FFT
    memneeded = KISSFFT_MEMNEEDED;
    stft->kissFFT_cfg = kiss_fftr_alloc(FFT_SIZE, 0, stft->kissFFTmem, &memneeded);
    memneeded = KISSFFT_MEMNEEDED;
    stft->kissIFFT_cfg = kiss_fftr_alloc(FFT_SIZE, 1, stft->kissIFFTmem, &memneeded);
    for (i=0; i<FFT_SIZE; i++)
    {
#if XCHAL_HAVE_HIFI5
        stft->window[i] = SQRT_S(0.5f - 0.5f * cosf(TWO_PI_BY_N * i));
#else
        stft->window[i] = sqrtf(0.5f - 0.5f * cosf(TWO_PI_BY_N * i));
#endif
    }
    //vDSP_hamm_window(stft->window, NFFT, 0);
    //vDSP_vfill(&temp, stft->window, 1, NFFT);
    JMPDSP_vclr(stft->inputAux, 1, FFT_SIZE);
    JMPDSP_vclr(stft->outputAux, 1, FFT_SIZE);
    JMPDSP_vclr(stft->Xmag, 1, stft->numBins);
}

void destroy_stft(STFTStruct *stft)
{

}

void stft_process(STFTStruct *stft, const float *input, unsigned int R)
{
    
    perform_windowed_FFT(stft, input, R);
    compute_magSquared(stft->Xk, stft->Xmag, stft->numBins);
    compute_logMag(stft->Xmag, stft->Xmag, stft->numBins, 10.0f);
    
}

void istft_process(STFTStruct *stft, float *output, unsigned int R)
{
    perform_OLA_IFFT(stft, output, R);
}

void mask_process(STFTStruct *stft, float *mask)
{
    apply_spectrum_mask(stft->Xk, mask, stft->Yk, stft->numBins);
}

void perform_windowed_FFT(STFTStruct *stft, const float *input, unsigned int R)
{
    float windowedInput[FFT_SIZE];
    update_buffer(stft->inputAux, input, stft->NFFT, R);
    JMPDSP_vmul(stft->inputAux, 1, stft->window, 1, windowedInput, 1, stft->NFFT);
    kiss_fftr(stft->kissFFT_cfg, windowedInput, stft->Xk); //KISS_FFT
}

void perform_OLA_IFFT(STFTStruct *stft, float *out, unsigned int R)
{
    float windowedOutput[FFT_SIZE];
    JMPDSP_vclr(windowedOutput, 1, stft->NFFT);
    update_buffer(stft->outputAux, windowedOutput, stft->NFFT, R);
    kiss_fftri(stft->kissIFFT_cfg, stft->Yk, windowedOutput);  //KISS_FFT
    JMPDSP_vmul(windowedOutput, 1, stft->window, 1, windowedOutput, 1, stft->NFFT); //Since we are not using sqrt(hann), we don't need to apply window in synthesis
    JMPDSP_vsmul(windowedOutput, 1, &stft->IFFTscale, windowedOutput, 1, stft->NFFT);
    JMPDSP_vadd(stft->outputAux, 1, windowedOutput, 1, stft->outputAux, 1, stft->NFFT);
    memcpy(out, stft->outputAux, R * sizeof(float));
}

// GENERAL (and KISSFFT) UTILITY FUNCTIONS
void update_buffer(float *buffer, const float *new_input, unsigned int N, unsigned int R)
{
    memcpy(&buffer[0], &buffer[R], (N-R) * sizeof(float));
    memcpy(&buffer[N-R], new_input, R * sizeof(float));
}

float calculate_mean(const float *x, unsigned int N)
{
    float result = 0.0f;
    JMPDSP_meanv(x, 1, &result, N);
    return(result);
}

void compute_magSquared(kiss_fft_cpx *Xk, float *Xmag, unsigned int num_bins)
{
    int i;
    for (i=0; i<num_bins; i++)
    {
        Xmag[i] = Xk[i].r*Xk[i].r + Xk[i].i*Xk[i].i;
        if ( (i > 0) && (i < num_bins-1) )
            Xmag[i] *= 2;
    }
    
}

void compute_logMag(float *Xmag, float *logMag, unsigned int num_bins, float scaleFactor)
{
    int i;
    for (i=0; i<num_bins; i++)
        logMag[i] = scaleFactor * log10f(Xmag[i] + LOGMAG_EPSILON);
}

void apply_spectrum_mask(kiss_fft_cpx *Xk, float *mask, kiss_fft_cpx *Yk, unsigned int num_bins)
{
    int i;
    for (i=0; i<num_bins; i++)
    {
        Yk[i].r = Xk[i].r * mask[i];
        Yk[i].i = Xk[i].i * mask[i];
    }
}



