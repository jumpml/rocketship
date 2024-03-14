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
//  dsplib.c
//
//  Created by Raghavendra Prabhu on 12/2/21.
//

#include "dsplib.h"
#include <math.h>
#include "fixed_point_math.h"
#include "common_def.h"
//#include <stdio.h>

void JMPDSP_vclr(float *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = 0.0f;
    }
}
void JMPDSP_vclr_S16(int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = 0;
    }
}
void JMPDSP_vclr_S32(int32_t *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = 0;
    }
}

void JMPDSP_vmul(const float *A, JMPDSP_Stride iA, const float *B, JMPDSP_Stride iB, float *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = A[i*iA] * B[i*iB];
    }
}

void JMPDSP_vmul_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *B, JMPDSP_Stride iB, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = FMUL16x16(A[i*iA], B[i*iB]);
    }
    
}


void JMPDSP_vadd(const float *A, JMPDSP_Stride iA, const float *B, JMPDSP_Stride iB, float *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = A[i*iA] + B[i*iB];
    }
}

void JMPDSP_vadd_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *B, JMPDSP_Stride iB, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = SLIMIT(FADD16(A[i*iA], B[i*iB]),15);
    }
}

void JMPDSP_vsmul(const float *A, JMPDSP_Stride iA, const float *B, float *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = A[i*iA] * (*B);
    }
}

void JMPDSP_vsmul_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *B, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = FMUL16x16(A[i*iA], *B);
    }
}


void JMPDSP_meanv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    float sum = 0.0f;
    for (i=0; i<N; i++)
    {
        sum += A[i*iA];
    }
    *C = sum / ((float) N);
}

void JMPDSP_meanv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    int32_t sum = 0;
    for (i=0; i<N; i++)
    {
        sum += A[i*iA];
    }
    *C = sum / N;
}

void JMPDSP_maxv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    float max_val = A[0];
    for (i=1; i<N; i++)
    {
        if (A[i*iA] > max_val)
            max_val = A[i*iA];
    }
    *C = max_val;
}

void JMPDSP_maxv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    int16_t max_val = A[0];
    for (i=1; i<N; i++)
    {
        if (A[i*iA] > max_val)
            max_val = A[i*iA];
    }
    *C = max_val;
}

void JMPDSP_minv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    float min_val = A[0];
    for (i=1; i<N; i++)
    {
        if (A[i*iA] < min_val)
            min_val = A[i*iA];
    }
    *C = min_val;
}

void JMPDSP_minv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    int16_t min_val = A[0];
    for (i=1; i<N; i++)
    {
        if (A[i*iA] < min_val)
            min_val = A[i*iA];
    }
    *C = min_val;
}

void JMPDSP_measqv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    float sum = 0.0f;
    for (i=0; i<N; i++)
    {
        sum += A[i*iA] * A[i*iA];
    }
    *C = sum / ((float) N);
}

void JMPDSP_measqv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N)
{
    JMPDSP_Length i;
    int32_t sum = 0;
    for (i=0; i<N; i++)
    {
        sum += IMUL(A[i*iA], A[i*iA]) << 15;
    }
    *C = sum / N;
}

void JMPDSP_vclip(const float *A, JMPDSP_Stride iA, const float *L, const float *H, float *D, JMPDSP_Stride iD, JMPDSP_Length N)
{
    JMPDSP_Length i;
    
    for (i=0; i<N; i++)
    {
        if (A[i*iA] < *L)
            D[i*iD] = *L;
        else if (A[i*iA] > *H)
            D[i*iD] = *H;
        else
            D[i*iD] = A[i*iA];
    }
}

void JMPDSP_vclip_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *L, const int16_t *H, int16_t *D, JMPDSP_Stride iD, JMPDSP_Length N)
{
    JMPDSP_Length i;
    
    for (i=0; i<N; i++)
    {
        if (A[i*iA] < *L)
            D[i*iD] = *L;
        else if (A[i*iA] > *H)
            D[i*iD] = *H;
        else
            D[i*iD] = A[i*iA];
    }
}

void JMPDSP_vfix8(const float *A, JMPDSP_Stride iA, signed char *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (char) A[i*iA];
    }
}
void JMPDSP_vfixr8(const float *A, JMPDSP_Stride iA, signed char *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (char) roundf(A[i*iA]);
    }
}
void JMPDSP_vfix16(const float *A, JMPDSP_Stride iA, short *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (short) A[i*iA];
    }
}
void JMPDSP_vfixr16(const float *A, JMPDSP_Stride iA, short *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (short) roundf(A[i*iA]);
    }
}
void JMPDSP_vfix32(const float *A, JMPDSP_Stride iA, int *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (int) A[i*iA];
    }
}
void JMPDSP_vfixr32(const float *A, JMPDSP_Stride iA, int *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (int) roundf(A[i*iA]);
    }
}
void JMPDSP_vflt8(const signed char *A, JMPDSP_Stride iA, float *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (float) A[i*iA];
    }
}
void JMPDSP_vflt16(const short *A, JMPDSP_Stride iA, float *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (float) A[i*iA];
    }
}
void JMPDSP_vflt32(const int *A, JMPDSP_Stride iA, float *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    JMPDSP_Length i;
    for (i=0; i<N; i++)
    {
        C[i*iC] = (float) A[i*iA];
    }
}

void JMPDSP_dot_prod(const float * __restrict__ A, const float * __restrict__ B,  float * __restrict__ C, JMPDSP_Length N)
{
    float acc = *C;
    int i;
    for (i=0;i<N;i++)
    {
        acc += A[i]*B[i];
    }
    *C = acc;
}

void JMPDSP_dot_prod_S8F32_F32(const int8_t * __restrict__ W, const float * __restrict__ A, float * __restrict__ C, JMPDSP_Length N)
{
    float acc = *C;
    int i;
    for (i=0;i<N;i++)
    {
        acc += A[i]*W[i];
    }
    *C = acc;
    
}

void JMPDSP_dot_prod_S8S16_S16(const int8_t * __restrict__ W, const int16_t * __restrict__ A, int16_t * __restrict__ C, JMPDSP_Length N, uint32_t numIntBits)
{
    int32_t acc = *C;
    int i;
    for (i=0;i<N;i++)
    {
        acc += (int32_t)(A[i]*W[i]);
    }
    acc = SLIMIT(acc, numIntBits + 23);
    *C = (int16_t)(acc >> (numIntBits + 7));
}

void JMPDSP_dot_prod_S8S16_S32(const int8_t * __restrict__ W, const int16_t * __restrict__ A, int32_t * __restrict__ C, JMPDSP_Length N)
{
    int32_t acc = *C;
    int i;
    for (i=0;i<N;i++)
    {
        acc += (int32_t)(A[i] * W[i]);
    }
    *C = acc;
}


void JMPDSP_vadd_S16S8_S16(const int16_t *A, JMPDSP_Stride iA, const int8_t *B, JMPDSP_Stride iB, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N)
{
    int32_t acc=0;
    int i;
    for (i=0;i<N;i++)
    {
        acc = (int32_t)A[i*iA] + (((int32_t)B[i*iB])<<8);
        acc = acc > 0x00007FFF ? 0x00007FFF:acc;
        acc = acc < 0xFFFF8000 ? 0xFFFF8000:acc;
        C[i*iC] = (int16_t) acc;
    }
    
}
