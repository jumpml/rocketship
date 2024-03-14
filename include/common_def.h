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
//  common_def.h
//
//  Created by Raghavendra Prabhu on 4/28/22.
//

#ifndef common_def_h
#define common_def_h

typedef int JMPDSP_Stride;
typedef unsigned int JMPDSP_Length;

#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif

#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE __attribute__((always_inline)) static inline
#endif

#define DSP_ALIGN16          __attribute__ ((__aligned__(16)))
#ifdef XCHAL_HAVE_HIFI5
#define DSP_RWDATA_IN_DRAM   __attribute__ ((__section__(".data")))
#else
#define DSP_RWDATA_IN_DRAM   
#endif



#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#endif /* common_def_h */
