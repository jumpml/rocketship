#include <math.h>
#include "biquad.h"
#include "common_def.h"
#include <stdio.h>


void biquad_init(BiquadFilter *filter, BiquadParams *params) {
    float omega = 2.0f * M_PI * params->fc / params->fs;
    float sin_omega = sinf(omega);
    float cos_omega = cosf(omega);
    float alpha = sin_omega / (2.0f * params->Q);
    float A = powf(10.0f, params->gain_db / 40.0f);

    float b0, b1, b2, a0, a1, a2;

    switch (params->type) {
        case LOWPASS:
            b0 = (1.0f - cos_omega) / 2.0f;
            b1 = 1.0f - cos_omega;
            b2 = (1.0f - cos_omega) / 2.0f;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case HIGHPASS:
            b0 = (1.0f + cos_omega) / 2.0f;
            b1 = -(1.0f + cos_omega);
            b2 = (1.0f + cos_omega) / 2.0f;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case BANDPASS:
            b0 = alpha;
            b1 = 0.0f;
            b2 = -alpha;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case NOTCH:
            b0 = 1.0f;
            b1 = -2.0f * cos_omega;
            b2 = 1.0f;
            a0 = 1.0f + alpha;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha;
            break;

        case LOWSHELF:
            b0 = A * ((A + 1.0f) - (A - 1.0f) * cos_omega + 2.0f * sqrtf(A) * alpha);
            b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cos_omega);
            b2 = A * ((A + 1.0f) - (A - 1.0f) * cos_omega - 2.0f * sqrtf(A) * alpha);
            a0 = (A + 1.0f) + (A - 1.0f) * cos_omega + 2.0f * sqrtf(A) * alpha;
            a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cos_omega);
            a2 = (A + 1.0f) + (A - 1.0f) * cos_omega - 2.0f * sqrtf(A) * alpha;
            break;

        case HIGHSHELF:
            b0 = A * ((A + 1.0f) + (A - 1.0f) * cos_omega + 2.0f * sqrtf(A) * alpha);
            b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cos_omega);
            b2 = A * ((A + 1.0f) + (A - 1.0f) * cos_omega - 2.0f * sqrtf(A) * alpha);
            a0 = (A + 1.0f) - (A - 1.0f) * cos_omega + 2.0f * sqrtf(A) * alpha;
            a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cos_omega);
            a2 = (A + 1.0f) - (A - 1.0f) * cos_omega - 2.0f * sqrtf(A) * alpha;
            break;

        case PEAKINGEQ:
            b0 = 1.0f + alpha * A;
            b1 = -2.0f * cos_omega;
            b2 = 1.0f - alpha * A;
            a0 = 1.0f + alpha / A;
            a1 = -2.0f * cos_omega;
            a2 = 1.0f - alpha / A;
            break;
    }

    // Normalize coefficients
    filter->b0 = b0 / a0;
    filter->b1 = b1 / a0;
    filter->b2 = b2 / a0;
    filter->a1 = a1 / a0;
    filter->a2 = a2 / a0;

    // Initialize delay line
    filter->z1 = 0.0f;
    filter->z2 = 0.0f;
}


void biquad_process(BiquadFilter *filter, float *input, float *output, int frame_size) 
{
    for (int i = 0; i < frame_size; i++) 
    {
        float new_out = filter->b0 * input[i] + filter->z1;
        new_out = MAX(MIN(new_out, 0.9999),-1.0);
        filter->z1 = filter->b1 * input[i] + filter->z2 - filter->a1 * new_out;
        filter->z2 = filter->b2 * input[i] - filter->a2 * new_out;
        output[i] = new_out;
    }
}

void biquad_cascade_init(BiquadCascade *cascade, BiquadParams *params, int num_biquads) {
    cascade->num_biquads = num_biquads;

    for (int i = 0; i < num_biquads; i++) {
        biquad_init(&cascade->biquads[i], &params[i]);
    }
}

void biquad_cascade_process(BiquadCascade *cascade, float *input, float *output, int frame_size) {
    float temp_in[frame_size];
    float temp_out[frame_size];

    // Copy input to output for processing
    for (int i = 0; i < frame_size; i++) {
        temp_in[i] = input[i];
    }

    // Process through each biquad in the cascade
    for (int i = 0; i < cascade->num_biquads; i++) {
        // Process the current biquad
        biquad_process(&cascade->biquads[i], temp_in, temp_out, frame_size);

        // Copy temp_out to temp_in for the next biquad
        for (int j = 0; j < frame_size; j++) {
            temp_in[j] = temp_out[j];
        }
    }

    // Copy the final output to the output array
    for (int i = 0; i < frame_size; i++) {
        output[i] = temp_out[i];
    }
}
