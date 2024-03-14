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
//  utils.h
//
#include "dsp_processing.h"

void gen_linspace(float *vec, float L, float R, int N);
void gen_randvec(float *vec, unsigned int N, int numFracBits);
void gen_cpxvec(kiss_fft_cpx *vec, unsigned int N, int numFracBits);

void convert_F32toS16(const float *A, int16_t *B, unsigned int N, int numFracBits);
void convert_F32toS32(const float *A, int32_t *B, unsigned int N, int Qm, int Qn);
void convert_F32toS8(const float *A, int8_t *B, unsigned int N);
void convert_S16toF32(const int16_t *A, float *B, unsigned int N, int numFracBits);
void convert_C32toC16(const kiss_fft_cpx *A, kiss_fft_cpx_S16 *B, unsigned int N);
void convert_C16toC32(const kiss_fft_cpx_S16 *A, kiss_fft_cpx_F32 *B, unsigned int N, int numFracBits);

void print_vector(float *vector, unsigned int N, char *vec_name);
void print_vector_S32(int32_t *vector, unsigned int N, char *vec_name, int Qn);
void print_vector_S16(int16_t *vector, unsigned int N, char *vec_name, int Qn);
void print_vector_S8(int8_t *vector, unsigned int N, char *vec_name);
void print_cpxvec(kiss_fft_cpx *vec, unsigned int N, char *vec_name);
void print_cpxvec_S16(kiss_fft_cpx_S16 *vec, unsigned int N, char *vec_name, int Qn);

float check_vectors(float *x, float *ref, int N);
float check_vec_S16(int16_t *x, float *ref, int N, int Qn);
float check_vec_S32(int32_t *x, float *ref, int N, int Qn);

void print_layer_stats(char *layer_name, float min, float max, float mean, float std);
void print_layer_stats_S16(char *layer_name, int16_t min, int16_t max, int16_t mean, int16_t std);
