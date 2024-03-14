//
//  nntests.h
//  test_nndsp
//
//  Created by Raghavendra Prabhu on 4/4/22.
//

#ifndef nntests_h
#define nntests_h

#include "nnlib.h"
#include "utils.h"
#include "signalsifter.h"
#include "dsplib.h"
#define NUM_PTS 128

#define MSE_THRESH  1e-5
#define NNTESTS_PASS 1
#define NNTESTS_FAIL 0

int check_mse(float mse)
{
    return mse < MSE_THRESH ? NNTESTS_PASS : NNTESTS_FAIL;
}

extern const SignalSifterModel ss_model;

void ACTIVATION_TEST(JMPNN_ActivationType actType)
{
    float input[NUM_PTS];
    float output[NUM_PTS];
    float ref[NUM_PTS];
    int16_t output_S16[NUM_PTS];
    int32_t input_S32[NUM_PTS];
    int i;
    float output_mse;
    float output_S16_mse;
    
    gen_linspace(input, -4, 4, NUM_PTS);
    convert_F32toS32(input, input_S32, NUM_PTS, 3, 15);
    
    if (actType == ACTIVATION_TANH)
    {
        for (i=0; i<NUM_PTS; i++)
            ref[i] = tanhf(input[i]);
        vec_tanh_F32(output, input, NUM_PTS);
        vec_tanh_S16(output_S16, input_S32, NUM_PTS);
    }
    else if (actType == ACTIVATION_SIGMOID)
    {
        for (i=0; i<NUM_PTS; i++)
            ref[i] = 0.5f + 0.5f * tanh_approx(0.5f*input[i]);
        vec_sigmoid_F32(output, input, NUM_PTS);
        vec_sigmoid_S16(output_S16, input_S32, NUM_PTS);
    }
    else if (actType == ACTIVATION_RELU)
    {
        for (i=0; i<NUM_PTS; i++)
            ref[i] = relu(input[i]);
        vec_relu_F32(output, input, NUM_PTS);
        vec_relu_S16(output_S16, input_S32, NUM_PTS);
    }
    else
    {
        
    }
    output_mse = check_vectors(output,ref, NUM_PTS);
    output_S16_mse = check_vec_S16(output_S16, ref, NUM_PTS, 15);
    printf("Float MSE = %f\t FixedPt MSE =%f\n", output_mse, output_S16_mse);
    
//    print_vector(input, NUM_PTS, "input");
//    print_vector_S32(input_S32, NUM_PTS, "input_S32", 15);
//    print_vector(output, NUM_PTS, "output");
//    print_vector_S16(output_S16, NUM_PTS, "output_S16", 15);

}

void NNLIB_MATXVEC_TEST(void)
{
    const LinearLayer *ll;
    ll = ss_model.linear1_linear;
    float input[ll->input_size];
    float output[ll->hidden_size];
    int16_t input_S16[ll->input_size];
    int32_t output_S32[ll->hidden_size];
    float mse;
    
    gen_randvec(input, ll->input_size, 15);
//    print_vector(input, 10, "input");
    JMPNN_linear_matXvec_S8xF32_F32(ll->weights, input, ll->bias,
                                      output, ll->input_size, ll->hidden_size,
                                      WEIGHTS_SCALE, BIAS_SCALE);
    convert_F32toS16(input, input_S16, ll->input_size, 15);
//    print_vector_S16(input_S16, 10, "input_S16",15);
    JMPNN_linear_matXvec_S8xS16_S32(ll->weights, input_S16, ll->bias,
                                    output_S32, ll->input_size, ll->hidden_size,
                                     15, 7);
    mse = check_vec_S32(output_S32, output, ll->hidden_size, 15);
//    print_vector(output, 10, "output");
//    print_vector_S32(output_S32, 10,"output_S32", 15 );
    printf("MATXVEC MSE = %f\n", mse);
}



void NNLIB_GRU_TEST(void)
{
    const GRULayer *gru;
    gru = ss_model.gru1_gru;
    float input[gru->input_size];
    float state[gru->hidden_size];
    float output[gru->hidden_size];

    float mse;
    int16_t input_S16[gru->input_size];
    int16_t state_S16[gru->hidden_size];
    int16_t output_S16[gru->hidden_size];
    
    gen_randvec(input, gru->input_size, 15);
    gen_randvec(state, gru->hidden_size, 15);
    convert_F32toS16(input, input_S16, gru->input_size, 15);
    convert_F32toS16(state, state_S16, gru->hidden_size, 15);
    
//    print_vector(input, ll->input_size, "input");
    JMPNN_gru_matXvec_S8xF32_F32_act(gru->input_weights, input, gru->bias,
                                        gru->recurrent_weights, state, gru->recurrent_bias,
                                   output, gru->input_size, gru->hidden_size,
                                          WEIGHTS_SCALE, BIAS_SCALE, ACTIVATION_SIGMOID);

    JMPNN_gru_matXvec_S8xS16_S16_act(gru->input_weights, input_S16, gru->bias,
                                     gru->recurrent_weights, state_S16, gru->recurrent_bias,
                                     output_S16, gru->input_size, gru->hidden_size,
                                     15, 7, 15, 7, ACTIVATION_SIGMOID);
    
    mse = check_vec_S16(output_S16, output, gru->hidden_size, 15);
    printf("MSE = %f\n", mse);
    
}

void NNLIB_GRU_NEWGATE_TEST(void)
{
    const GRULayer *gru;
    gru = ss_model.gru1_gru;
    float input[gru->input_size];
    float state[gru->hidden_size];
    float r[gru->hidden_size], h[gru->hidden_size];
    int Ni = gru->hidden_size * gru->input_size;
    int Nh = gru->hidden_size * gru->hidden_size;
    int N = gru->hidden_size;
    float mse;
    
    int16_t input_S16[gru->input_size];
    int16_t state_S16[gru->hidden_size];
    int16_t r_S16[gru->hidden_size], h_S16[gru->hidden_size];
    
    gen_randvec(input, gru->input_size, 15);
    gen_randvec(state, gru->hidden_size, 15);
    gen_randvec(r, gru->hidden_size, 15);
    convert_F32toS16(input, input_S16, gru->input_size, 15);
    convert_F32toS16(state, state_S16, gru->hidden_size, 15);
    convert_F32toS16(r, r_S16, gru->hidden_size, 15);
    
    JMPNN_gru_newGate_S8xF32_F32_act(&gru->input_weights[2*Ni], input, &gru->bias[2*N],
                                          &gru->recurrent_weights[2*Nh], state, &gru->recurrent_bias[2*N], r,
                                          h, gru->input_size, gru->hidden_size,
                                          WEIGHTS_SCALE, BIAS_SCALE, gru->activation);

    JMPNN_gru_newGate_S8xS16_S16_act(&gru->input_weights[2*Ni], input_S16, &gru->bias[2*N],
                                     &gru->recurrent_weights[2*Nh], state_S16, &gru->recurrent_bias[2*N], r_S16,
                                     h_S16, gru->input_size, gru->hidden_size,
                                     15, 7, 15, 7, gru->activation);
    
    mse = check_vec_S16(h_S16, h, gru->hidden_size, 15);
    printf("GRU NEWGATE MSE = %f\n", mse);
    
}

void NNLIB_VECINT_TEST(void)
{
    float input1[NUM_PTS], input2[NUM_PTS], interp_vec[NUM_PTS];
    float output[NUM_PTS];
    
    int16_t input1_S16[NUM_PTS], input2_S16[NUM_PTS], interp_vec_S16[NUM_PTS];
    int16_t output_S16[NUM_PTS];
    float mse;
    
    gen_randvec(input1, NUM_PTS, 15);
    gen_randvec(input2, NUM_PTS, 15);
    gen_randvec(interp_vec, NUM_PTS, 15);
    convert_F32toS16(input1, input1_S16, NUM_PTS, 15);
    convert_F32toS16(input2, input2_S16, NUM_PTS, 15);
    convert_F32toS16(interp_vec, interp_vec_S16, NUM_PTS, 15);
    
    JMPNN_vec_interpolation_F32(output, interp_vec, input1, input2, NUM_PTS);
    
    JMPNN_vec_interpolation_S16(output_S16, interp_vec_S16, input1_S16, input2_S16, NUM_PTS);
    
    mse = check_vec_S16(output_S16, output, NUM_PTS, 15);
    printf("VECINT MSE = %f\n", mse);
//    print_vector(output, 10, "output");
//    print_vector_S16(output_S16, 10,"output_S16", 15);
}

void NNLAYERS_LINEAR_TEST(void)
{
    const LinearLayer *ll;
    ll = ss_model.linear1_linear;
    float input[ll->input_size];
    float output[ll->hidden_size];
    int16_t input_S16[ll->input_size];
    int16_t output_S16[ll->hidden_size];
    float mse;
    
    gen_randvec(input, ll->input_size, 15);
//    print_vector(input, ll->input_size, "input");
    computeLinearLayer(ll, output, input);
    
    convert_F32toS16(input, input_S16, ll->input_size, 15);
    computeLinearLayer_S16(ll, output_S16, input_S16, 15, 7);
    
    mse = check_vec_S16(output_S16, output, ll->hidden_size, 15);
    printf("LINEAR LAYER MSE = %f\n", mse);
//    print_vector(output, 10, "output");
//    print_vector_S16(output_S16, 10,"output_S16", 15 );
    
}

void NNLAYERS_GRU_TEST(void)
{
    const GRULayer *gru;
    gru = ss_model.gru1_gru;
    float input[gru->input_size];
    float state[gru->hidden_size];
    int16_t input_S16[gru->input_size];
    int16_t state_S16[gru->hidden_size];
    float mse;
    
    gen_randvec(input, gru->input_size, 15);
    gen_randvec(state, gru->hidden_size, 15);
    convert_F32toS16(input, input_S16, gru->input_size, 15);
    convert_F32toS16(state, state_S16, gru->hidden_size, 15);
    
//    print_vector(input, ll->input_size, "input");
    computeGRULayer(gru, state, input);
    
    computeGRULayer_S16(gru, state_S16, input_S16, 15, 7, 15, 7);
    
    mse = check_vec_S16(state_S16, state, gru->hidden_size, 15);
    printf("GRU LAYER MSE = %f\n", mse);
//    print_vector(output, 10, "output");
//    print_vector_S32(output_S32, 10,"output_S32", 15 );
    
}

void RUN_NNTESTS(void)
{
    ACTIVATION_TEST(ACTIVATION_TANH);
    ACTIVATION_TEST(ACTIVATION_SIGMOID);
    ACTIVATION_TEST(ACTIVATION_RELU);
    
    NNLIB_MATXVEC_TEST();
    NNLAYERS_LINEAR_TEST();
//    NNLIB_GRU_MATXVEC_TEST();
    NNLIB_GRU_NEWGATE_TEST();
    NNLIB_VECINT_TEST();
    NNLAYERS_GRU_TEST();
}

#endif /* nntests_h */
