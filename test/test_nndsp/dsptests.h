//
//  dsptests.h
//  test_nndsp
//
//  Created by Raghavendra Prabhu on 4/4/22.
//

#ifndef dsptests_h
#define dsptests_h

#include "dsplib.h"
#include "utils.h"
#include "dsp_processing.h"

#define NUM_INT_BITS 2
#define NUM_PTS 128
#define FFT_SIZE 2
#define NUM_BINS FFT_SIZE/2 + 1

void DSPPROCESSING_TEST(void)
{
//    kiss_fft_cpx X[NUM_BINS], Y[NUM_BINS];
//    float Xmag[NUM_BINS], logXmag[NUM_BINS], mask[NUM_BINS];
//    kiss_fft_cpx_S16 X_S16[NUM_BINS], Y_S16[NUM_BINS];
//    int32_t Xmag_S32[NUM_BINS], logXmag_S32[NUM_BINS];
//    int16_t mask_S16[NUM_BINS];
//    gen_cpxvec(X, NUM_BINS, 15);
////    print_cpxvec(X, NUM_BINS, "X");
//
//    convert_C32toC16(X, X_S16, NUM_BINS);
////    print_cpxvec_S16(X_S16, NUM_BINS, "X_S16", 15);
//
//    compute_magSquared(X, Xmag, NUM_BINS);
//    compute_magSquared_S16(X_S16, Xmag_S32, NUM_BINS);
//
////    print_vector(Xmag, NUM_BINS, "Xmag");
////    print_vector_S32(Xmag_S32, NUM_BINS, "Xmag_S32", 15);
//
//    float mse = check_vec_S32(Xmag_S32, Xmag, NUM_BINS, 15);
//    printf("MSE Mag Squared = %f\n", mse);
//
//    compute_logMag(Xmag, logXmag, NUM_BINS, 10);
//    compute_logMag_S32(Xmag_S32, logXmag_S32, NUM_BINS, 10);
//    mse = check_vec_S32(logXmag_S32, logXmag, NUM_BINS, 15);
//    printf("MSE logMag = %f\n", mse);
//
//    gen_linspace(mask, 0.0f, 1.0, NUM_BINS);
//    print_vector(mask, NUM_BINS, "mask");
//    convert_F32toS16(mask, mask_S16, NUM_BINS);
//    print_vector_S16(mask_S16, NUM_BINS, "mask_S16", 15);
//    apply_spectrum_mask_S16(X_S16, mask_S16, Y_S16, NUM_BINS);
//    apply_spectrum_mask(X, mask, Y, NUM_BINS);
//    print_cpxvec(Y, NUM_BINS, "Y");
//    print_cpxvec_S16(Y_S16, NUM_BINS, "Y_S16", 15);
    
}

void MEAN_TEST(void)
{
//    float input[NUM_PTS], res;
//    int16_t input_S16[NUM_PTS], res_S16 =0;
//
//    gen_randvec(input, NUM_PTS, 15);
//    convert_F32toS16(input, input_S16, NUM_PTS);
//
//    res = calculate_mean(input, NUM_PTS);
//    res_S16 = calculate_mean_S16(input_S16, NUM_PTS);
//
//    printf("Float Mean = %f\n", res);
//    printf("Fixed-point mean = %f\n", res_S16 * powf(2,-15));
    
}

void DOTPROD_TEST(void)
{
    float w[NUM_PTS];
    float input[NUM_PTS];
    float res = 0;
    
    
    gen_randvec(w, NUM_PTS, 7);
    gen_randvec(input, NUM_PTS, 15);
    
    int8_t w_S8[NUM_PTS];
    int16_t input_S16[NUM_PTS], res_S16 =0;
    convert_F32toS8(w, w_S8, NUM_PTS);
    convert_F32toS16(input, input_S16, NUM_PTS, 15);
    
    JMPDSP_dot_prod(w, input, &res, NUM_PTS);
    JMPDSP_dot_prod_S8S16_S16(w_S8, input_S16, &res_S16, NUM_PTS, NUM_INT_BITS);
    
    int res_S32 = 0;
    JMPDSP_dot_prod_S8S16_S32(w_S8, input_S16, &res_S32, NUM_PTS);

    
//    print_vector(w, NUM_PTS, "w    ");
//    print_vector_S8(w_S8, NUM_PTS, "w_S8");
//
//    print_vector(input, NUM_PTS, "input    ");
//    print_vector_S16(input_S16, NUM_PTS, "input_S16", 15);
    
    printf("dot(w,x) = %f dotS16(w_S8,x_S16)=%f  dotS32(w_S8,x_S16)=%f\n", res, res_S16/powf(2,15-NUM_INT_BITS), res_S32/powf(2,22));
}

void RUN_DSPTESTS(void)
{
//    DOTPROD_TEST();
//    DSPPROCESSING_TEST();
//    MEAN_TEST();
}


#endif /* dsptests_h */
