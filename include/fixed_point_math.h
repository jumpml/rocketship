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
//  fixed_point_math.h
//
//  Created by Raghavendra Prabhu on 3/11/22.
//

#ifndef fixed_point_math_h
#define fixed_point_math_h
#include <stdint.h>
#include <math.h>
#include "common_def.h"

#define FMUL16x16(x, y) (int16_t)((int32_t)((int32_t)(x) * (int32_t)(y)) >> 15)
#define FMUL32x16(x, y) (int32_t) (((int64_t)(x) * (int64_t)(y)) >> 15)
#define IMUL(x, y) (int32_t)((int32_t)(x) * (int32_t)(y))
#define FMUL32x32(x, y) (int32_t)(((int64_t)(x) * (int64_t)(y)) >> 31)

#define FADD16(x,y) (int32_t)((int32_t)(x) + (int32_t)(y))

__STATIC_FORCEINLINE int32_t SLIMIT(int32_t val, uint32_t num_bits)
{
    int minval = -(1U << (num_bits - 1));
    int maxval = -minval - 1U;
    if (val > maxval)
    {
        return maxval;
    }
    else if (val < minval)
    {
        return minval;
    }
    return val;
}

// TODO : fix fake fixed-point math functions
__STATIC_INLINE int16_t sin_S16(int32_t x_S16)
{
    float x = x_S16 / 32768.0f;
    float y = sinf(x);
    int16_t y_S16 = MIN(y * 32768, 32767);
    return y_S16;
}

// TODO : fix fake fixed-point math functions
__STATIC_INLINE int16_t log10_S32(int32_t x_S32)
{
    // INPUT Q-format: 16.15
    float x = x_S32 / 32768.0f;
    float y = log10f(x);
    // OUTPUT Q-format: 16.15
    int32_t y_S32 = y * 32768;
    return y_S32;
}


#endif /* fixed_point_math_h */
