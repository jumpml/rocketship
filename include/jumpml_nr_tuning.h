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
//  jumpml_nr_tuning.h
//

#ifndef _JUMPML_NR_TUNING_H_
#define _JUMPML_NR_TUNING_H_

#define JUMPML_NR_FIRMWARE_VERSION 0x0001 // VERSION NUMBER: MS byte Major and LS byte Minor.

#define JUMPML_NR_APPLY_INPUT_GAIN 0
#define JUMPML_NR_INPUT_MIC_GAIN 1.414f   // PRE NR GAIN

#define JUMPML_NR_APPLY_OUTPUT_GAIN 0
#define JUMPML_NR_OUTPUT_GAIN 1.414f      // POST NR GAIN

#define JUMPML_NR_APPLY_HIGHSHELF 0
#define JUMPML_NR_OUTPUT_HIGHSHELF_FC   1600  // Hz
#define JUMPML_NR_OUTPUT_HIGHSHELF_FS   16000 // Hz
#define JUMPML_NR_OUTPUT_HIGHSHELF_GAIN 0.0f  // dB
#define JUMPML_NR_OUTPUT_HIGHSHELF_Q    0.707f 

#define JUMPML_NR_NATURALNESS 1.0f       // optional float number between 0 (max suppression) and 1 (most natural). Default: 0.5\n"
#define JUMPML_NR_MIN_GAIN    -6.0f     // optional minimum suppression gain floor in dB in [-60, 0] dB. Default: -40 dB\n"

#endif /* _JUMPML_NR_TUNING_H_ */

