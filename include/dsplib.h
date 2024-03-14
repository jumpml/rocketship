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
//  dsplib.h
//
//  Created by Raghavendra Prabhu on 12/2/21.
//

#ifndef dsplib_h
#define dsplib_h

#include <stdint.h>
#include "common_def.h"

void JMPDSP_vclr(float *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vmul(const float *A, JMPDSP_Stride iA, const float *B, JMPDSP_Stride iB, float *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vadd(const float *A, JMPDSP_Stride iA, const float *B, JMPDSP_Stride iB, float *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vsmul(const float *A, JMPDSP_Stride iA, const float *B, float *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_meanv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N);
void JMPDSP_maxv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N);
void JMPDSP_minv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N);
void JMPDSP_measqv(const float *A, JMPDSP_Stride iA, float *C, JMPDSP_Length N);
void JMPDSP_vclip(const float *A, JMPDSP_Stride iA, const float *L, const float *H, float *D, JMPDSP_Stride iD, JMPDSP_Length N);

void JMPDSP_vfix8(const float *A, JMPDSP_Stride iA, signed char *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vfixr8(const float *A, JMPDSP_Stride iA, signed char *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vfix16(const float *A, JMPDSP_Stride iA, short *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vfixr16(const float *A, JMPDSP_Stride iA, short *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vfix32(const float *A, JMPDSP_Stride iA, int *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vfixr32(const float *A, JMPDSP_Stride iA, int *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vflt8(const signed char *A, JMPDSP_Stride iA, float *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vflt16(const short *A, JMPDSP_Stride iA, float *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vflt32(const int *A, JMPDSP_Stride iA, float *C, JMPDSP_Stride iC, JMPDSP_Length N);

// No vDSP equivalents
void JMPDSP_dot_prod(const float *A, const float *B, float *C, JMPDSP_Length N);
void JMPDSP_dot_prod_S8F32_F32(const int8_t *W, const float *A, float *C, JMPDSP_Length N);
void JMPDSP_dot_prod_S8S16_S16(const int8_t *W, const int16_t *A, int16_t *C, JMPDSP_Length N, uint32_t numIntBits);
void JMPDSP_dot_prod_S8S16_S32(const int8_t *W, const int16_t *A, int32_t *C, JMPDSP_Length N);
void JMPDSP_vmul_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *B, JMPDSP_Stride iB, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vadd_S16S8_S16(const int16_t *A, JMPDSP_Stride iA, const int8_t *B, JMPDSP_Stride iB, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N);

// FIXED-POINT VERSIONS
void JMPDSP_vclr_S16(int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vclr_S32(int32_t *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vmul_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *B, JMPDSP_Stride iB, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vadd_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *B, JMPDSP_Stride iB, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_vsmul_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *B, int16_t *C, JMPDSP_Stride iC, JMPDSP_Length N);
void JMPDSP_meanv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N);
void JMPDSP_maxv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N);
void JMPDSP_minv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N);
void JMPDSP_measqv_S16(const int16_t *A, JMPDSP_Stride iA, int16_t *C, JMPDSP_Length N);
void JMPDSP_vclip_S16(const int16_t *A, JMPDSP_Stride iA, const int16_t *L, const int16_t *H, int16_t *D, JMPDSP_Stride iD, JMPDSP_Length N);

#endif /* dsplib_h */
