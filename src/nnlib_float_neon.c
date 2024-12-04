#include "nnlib_float.h"

#ifdef USE_NEON  // NEON kernels are 4x faster than generic kernels on M1
#include <arm_neon.h>

void JMPNN_linear_matXvec_S8xF32_F32(const int8_t * restrict W, 
                                     const float * restrict input, 
                                     const int8_t * restrict bias,
                                     float * restrict output, 
                                     JMPDSP_Length input_len, 
                                     JMPDSP_Length output_len,
                                     float weightScale, 
                                     float biasScale)
{
    for (int i = 0; i < output_len; i++)
    {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        int j;
        
        for (j = 0; j <= input_len - 4; j += 4)
        {
            int8x8_t w_vec = vld1_s8(&W[i*input_len + j]);
            float32x4_t input_vec = vld1q_f32(&input[j]);
            int16x8_t w_vec16 = vmovl_s8(w_vec);
            int32x4_t w_vec32 = vmovl_s16(vget_low_s16(w_vec16));
            float32x4_t w_vecf = vcvtq_f32_s32(w_vec32);
            sum_vec = vmlaq_f32(sum_vec, w_vecf, input_vec);
        }

        float sum = vaddvq_f32(sum_vec);

        // Handle remaining elements
        for (; j < input_len; j++)
        {
            sum += W[i*input_len + j] * input[j];
        }

        output[i] = weightScale * sum + biasScale * bias[i];
    }
}

void JMPNN_gru_matXvec_S8xF32_F32_act(const int8_t * restrict Wi, 
                                      const float * restrict input, 
                                      const int8_t * restrict Bi,
                                      const int8_t * restrict Wh, 
                                      const float * restrict prev_state, 
                                      const int8_t * restrict Bh,
                                      float * restrict output, 
                                      JMPDSP_Length input_len, 
                                      JMPDSP_Length output_len,
                                      float weightScale, 
                                      float biasScale, 
                                      JMPNN_ActivationType actType)
{
    for (int i = 0; i < output_len; i++)
    {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        int j;

        // Process input
        for (j = 0; j <= input_len - 4; j += 4)
        {
            int8x8_t wi_vec = vld1_s8(&Wi[i*input_len + j]);
            float32x4_t input_vec = vld1q_f32(&input[j]);
            int16x8_t wi_vec16 = vmovl_s8(wi_vec);
            int32x4_t wi_vec32 = vmovl_s16(vget_low_s16(wi_vec16));
            float32x4_t wi_vecf = vcvtq_f32_s32(wi_vec32);
            sum_vec = vmlaq_f32(sum_vec, wi_vecf, input_vec);
        }

        // Handle remaining input elements
        for (; j < input_len; j++)
        {
            sum_vec = vaddq_f32(sum_vec, vdupq_n_f32(Wi[i*input_len + j] * input[j]));
        }

        // Process prev_state
        for (j = 0; j <= output_len - 4; j += 4)
        {
            int8x8_t wh_vec = vld1_s8(&Wh[i*output_len + j]);
            float32x4_t prev_state_vec = vld1q_f32(&prev_state[j]);
            int16x8_t wh_vec16 = vmovl_s8(wh_vec);
            int32x4_t wh_vec32 = vmovl_s16(vget_low_s16(wh_vec16));
            float32x4_t wh_vecf = vcvtq_f32_s32(wh_vec32);
            sum_vec = vmlaq_f32(sum_vec, wh_vecf, prev_state_vec);
        }

        // Handle remaining prev_state elements
        for (; j < output_len; j++)
        {
            sum_vec = vaddq_f32(sum_vec, vdupq_n_f32(Wh[i*output_len + j] * prev_state[j]));
        }

        float sum = vaddvq_f32(sum_vec);
        output[i] = weightScale * sum + biasScale * (Bi[i] + Bh[i]);
    }
    JMPNN_apply_activation_F32(output, output_len, actType);
}

void JMPNN_gru_newGate_S8xF32_F32_act(const int8_t * restrict Wi, 
                                      const float * restrict input, 
                                      const int8_t * restrict Bi,
                                      const int8_t * restrict Wh, 
                                      const float * restrict prev_state, 
                                      const int8_t * restrict Bh,
                                      const float * restrict r, 
                                      float * restrict output, 
                                      JMPDSP_Length input_len, 
                                      JMPDSP_Length output_len,
                                      float weightScale, 
                                      float biasScale, 
                                      JMPNN_ActivationType actType)
{
    for (int i = 0; i < output_len; i++)
    {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        float32x4_t rec_sum_vec = vdupq_n_f32(0.0f);
        int j;

        // Process input
        for (j = 0; j <= input_len - 4; j += 4)
        {
            int8x8_t wi_vec = vld1_s8(&Wi[i*input_len + j]);
            float32x4_t input_vec = vld1q_f32(&input[j]);
            int16x8_t wi_vec16 = vmovl_s8(wi_vec);
            int32x4_t wi_vec32 = vmovl_s16(vget_low_s16(wi_vec16));
            float32x4_t wi_vecf = vcvtq_f32_s32(wi_vec32);
            sum_vec = vmlaq_f32(sum_vec, wi_vecf, input_vec);
        }

        // Handle remaining input elements
        for (; j < input_len; j++)
        {
            sum_vec = vaddq_f32(sum_vec, vdupq_n_f32(Wi[i*input_len + j] * input[j]));
        }

        // Process prev_state
        for (j = 0; j <= output_len - 4; j += 4)
        {
            int8x8_t wh_vec = vld1_s8(&Wh[i*output_len + j]);
            float32x4_t prev_state_vec = vld1q_f32(&prev_state[j]);
            int16x8_t wh_vec16 = vmovl_s8(wh_vec);
            int32x4_t wh_vec32 = vmovl_s16(vget_low_s16(wh_vec16));
            float32x4_t wh_vecf = vcvtq_f32_s32(wh_vec32);
            rec_sum_vec = vmlaq_f32(rec_sum_vec, wh_vecf, prev_state_vec);
        }

        // Handle remaining prev_state elements
        for (; j < output_len; j++)
        {
            rec_sum_vec = vaddq_f32(rec_sum_vec, vdupq_n_f32(Wh[i*output_len + j] * prev_state[j]));
        }

        float sum = vaddvq_f32(sum_vec);
        float rec_sum = vaddvq_f32(rec_sum_vec);

        rec_sum = weightScale * rec_sum + biasScale * Bh[i];
        sum = weightScale * sum + biasScale * Bi[i];
        output[i] = sum + rec_sum * r[i];
    }
    JMPNN_apply_activation_F32(output, output_len, actType);
}
#endif 