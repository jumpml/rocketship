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
//  utils.c
//
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "dsplib.h"
#include "utils.h"
#include "fixed_point_math.h"
void gen_linspace(float *vec, float L, float R, int N)
{
    float step = (R - L) / (N-1);
    int i;
    for (i=0; i<N; i++)
        vec[i] = L + i * step;
}

void gen_randvec(float *vec, unsigned int N, int numFracBits)
{
    float num;
    int i;
    float scale = powf(2, numFracBits);
    
    for (i=0; i<N; i++)
    {
        num = ((float)rand())/RAND_MAX;
        vec[i] =  floorf((2.0*num - 1.0f) * scale) / scale ;
    }
}

void gen_cpxvec(kiss_fft_cpx *vec, unsigned int N, int numFracBits)
{
    float num_re, num_im;
    float scale = powf(2, numFracBits);
    int i;
    for (i=0; i<N; i++)
    {
        num_re = ((float)rand())/RAND_MAX;
        num_im = ((float)rand())/RAND_MAX;
        vec[i].r =  floorf((2.0*num_re - 1.0f) * scale) / scale;
        vec[i].i =  floorf((2.0*num_im - 1.0f) * scale) / scale;
    }
}

void convert_F32toS16(const float *A, int16_t *B, unsigned int N, int numFracBits)
{
    uint16_t scaleInt = 1 << numFracBits;
    int32_t val;
    int i;
    
    for (i=0; i<N; i++)
    {
        val = A[i] * scaleInt;
        B[i] = MIN(MAX(val, -32768), 32767);
    }
    
//    JMPDSP_vclip(A, 1, &minv, &maxv, temp, 1, N);
//    JMPDSP_vsmul(temp, 1, &scaleInt, temp, 1, N);
//    JMPDSP_vfix16(temp, 1, B, 1, N);
}

void convert_S16toF32(const int16_t *A, float *B, unsigned int N, int numFracBits)
{
    float scale = powf(2, -numFracBits);
    int i;
    for (i=0; i<N; i++)
    {
        B[i] = A[i] * scale;
    }
}

void convert_C16toC32(const kiss_fft_cpx_S16 *A, kiss_fft_cpx_F32 *B, unsigned int N, int numFracBits)
{
    float scale = powf(2, -numFracBits);
    int i;
    for (i=0; i<N; i++)
    {
        B[i].r = A[i].r * scale;
        B[i].i = A[i].i * scale;
    }
}

void convert_C32toC16(const kiss_fft_cpx *A, kiss_fft_cpx_S16 *B, unsigned int N)
{
    int numFracBits = 15;
    float scale = powf(2, -numFracBits);
    float scaleInt = powf(2, numFracBits);
    float minv = -1;
    float maxv = 1-scale;
    int i;

    for (i=0; i<N; i++)
    {
        B[i].r = MAX(MIN(A[i].r, maxv),minv) * scaleInt;
        B[i].i = MAX(MIN(A[i].i, maxv),minv) * scaleInt;
    }

}

void convert_F32toS32(const float *A, int32_t *B, unsigned int N, int Qm, int Qn)
{
    float temp[N];
    int numFracBits = Qn;
    int numIntBits = Qm;
    float scale = powf(2, -numFracBits);
    float scaleInt = powf(2, numFracBits);
    float minv = -powf(2,numIntBits);
    float maxv = -minv-scale;
    JMPDSP_vclip(A, 1, &minv, &maxv, temp, 1, N);
    JMPDSP_vsmul(temp, 1, &scaleInt, temp, 1, N);
    JMPDSP_vfix32(temp, 1, B, 1, N);
}

void convert_F32toS8(const float *A, int8_t *B, unsigned int N)
{
    float temp[N];
    int numFracBits = 7;
    float scale = powf(2, -numFracBits);
    float scaleInt = powf(2, numFracBits);
    float minv = -1;
    float maxv = 1-scale;
    JMPDSP_vclip(A, 1, &minv, &maxv, temp, 1, N);
    JMPDSP_vsmul(temp, 1, &scaleInt, temp, 1, N);
    JMPDSP_vfix8(temp, 1, B, 1, N);
}

void print_vector(float *vector, unsigned int N, char *vec_name)
{
    int i;
    printf("%s=[", vec_name);
    for (i=0; i<N; i++)
    {
        printf("%f", vector[i]);
        if ((i+1) % 10 == 0)
            printf("\n");
        
        if (i < N-1)
            printf(", ");
        
    }
    printf("]\n");
}

void print_vector_S32(int32_t *vector, unsigned int N, char *vec_name, int Qn)
{
    int i;
    printf("%s=[", vec_name);
    float scale = powf(2, -Qn);
    for (i=0; i<N; i++)
    {
        printf("%f", vector[i] * scale);
        if ((i+1) % 10 == 0)
            printf("\n");
        
        if (i < N-1)
            printf(", ");
    }
    printf("]\n");
    
}

void print_vector_S16(int16_t *vector, unsigned int N, char *vec_name, int Qn)
{
    float scale = powf(2, -Qn);
    int i;
    printf("%s=[", vec_name);
    for (i=0; i<N; i++)
    {
        printf("%f", vector[i] * scale);
        if ((i+1) % 10 == 0)
            printf("\n");
        
        if (i < N-1)
            printf(", ");
        
    }
    printf("]\n");
}

void print_vector_S8(int8_t *vector, unsigned int N, char *vec_name)
{
    int i;
    printf("%s=[", vec_name);
    for (i=0; i<N; i++)
    {
        printf("%f", vector[i]/128.0f);
        if ((i+1) % 10 == 0)
            printf("\n");
        
        if (i < N-1)
            printf(", ");
        
    }
    printf("]\n");
}

void print_cpxvec(kiss_fft_cpx *vector, unsigned int N, char *vec_name)
{
    int i;
    printf("%s=[", vec_name);
    for (i=0; i<N; i++)
    {
        printf("%f + %fj", vector[i].r, vector[i].i);
        if ((i+1) % 5 == 0)
            printf("\n");

        if (i < N-1)
            printf(", ");

    }
    printf("]\n");
}
void print_cpxvec_S16(kiss_fft_cpx_S16 *vector, unsigned int N, char *vec_name, int Qn)
{
    int i;
    float scale = powf(2, -Qn);
    printf("%s=[", vec_name);
    for (i=0; i<N; i++)
    {
        printf("%f + %fj", vector[i].r * scale, vector[i].i * scale);
        if ((i+1) % 5 == 0)
            printf("\n");

        if (i < N-1)
            printf(", ");

    }
    printf("]\n");

}

#define MSE_THRESH 1e-4f
float check_vectors(float *x, float *ref, int N)
{
    int i;
    float mse = 0.0f;
    for (i=0; i<N; i++)
        mse += (x[i] - ref[i])*(x[i] - ref[i]);
    mse = mse / (float) N;
    return mse;
}

float check_vec_S16(int16_t *x, float *ref, int N, int Qn)
{
    int i;
    float mse = 0.0f;
    float scale = powf(2, -Qn);
    float x_val;
    for (i=0; i<N; i++)
    {
        x_val = (float) x[i] * scale;
        mse += (x_val - ref[i]) * (x_val - ref[i]);
    }
    mse = mse / (float) N;
    return mse;
}

float check_vec_S32(int32_t *x, float *ref, int N, int Qn)
{
    float mse = 0.0f;
    float scale = powf(2, -Qn);
    float x_val;
    int i;
    for (i=0; i<N; i++)
    {
        x_val = (float) x[i] * scale;
        mse += (x_val - ref[i]) * (x_val - ref[i]);
    }
    mse = mse / (float) N;
    return mse;
}


void print_layer_stats(char *layer_name, float min, float max, float mean, float std)
{
    printf("%s :[Min, Max, Mean, Std] = [%.4f,%.4f,%.4f,%.4f]\n\n", layer_name, min, max, mean, std);
}

void print_layer_stats_S16(char *layer_name, int16_t min, int16_t max, int16_t mean, int16_t std)
{
    printf("%s :[Min, Max, Mean, Std] = [%.4f,%.4f,%.4f,%.4f]\n\n", layer_name, min/32768.0f, max/32768.0f, mean/32768.0f, std/32768.0f);
}
