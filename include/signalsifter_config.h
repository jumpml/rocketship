//
//  ./include/signalsifter_config.h
//
//  Created by convert_model.py 
//  © 2024 JumpML
/*This file is automatically generated from a Pytorch model and config file*/
#ifndef signalsifter_config_h
#define signalsifter_config_h
#include <stdint.h>

#define WAB_FRAC_BITS  (7)
#define INPUT_NUM_FRAC_BITS  (10)
#define GRU_NUM_FRAC_BITS  (15)
#define LIN_NUM_FRAC_BITS  (15)
#define INPUTLAYER_SHIFT_RIGHT  ((INPUT_NUM_FRAC_BITS) + WAB_FRAC_BITS - (GRU_NUM_FRAC_BITS))
typedef int8_t nnWeight;
#define WEIGHTS_SCALE (1.f/128)
typedef int8_t nnBias;
#define BIAS_SCALE (1.f/128)

#define GRU_STATE_SIZE 352
#define MAX_NEURONS 352
#define IO_SIZE 128
#define FFT_SIZE 256
#define HOP_LENGTH 128
#define FRAME_SIZE 128
#define NUM_BINS 129

#define LOGMAG_EPSILON 0.001
#define LOGMAG_EPSILON_Q15 (int16_t)(LOGMAG_EPSILON * 32767.0f)

#endif
