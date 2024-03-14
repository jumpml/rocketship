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
//  test_signalsifter.c
//
//  Created by Raghavendra Prabhu on 10/13/21.
//

#include <stdio.h>
#include "signalsifter.h"
#include "signalsifter_testvector.h"
#include "utils.h"
#include <string.h>


int main(int argc, char **argv) {
    float gains[IO_SIZE];
    int16_t gains_S16[IO_SIZE];
   
    SignalSifterState SS;
    createSignalSifterModel(&SS);
    
    SignalSifterState_S16 SS_S16;
    createSignalSifterModel_S16(&SS_S16);
    
    memcpy(SS.gru1_gru_state, test_state_in1, SS.model->gru1_hidden_size * sizeof(float));
    memcpy(SS.gru2_gru_state, test_state_in2, SS.model->gru2_hidden_size * sizeof(float));
    memcpy(SS.gru3_gru_state, test_state_in3, SS.model->gru3_hidden_size * sizeof(float));
    
    memcpy(SS_S16.gru1_gru_state, test_state_in1_S16, SS_S16.model->gru1_hidden_size * sizeof(int16_t));
    memcpy(SS_S16.gru2_gru_state, test_state_in2_S16, SS_S16.model->gru2_hidden_size * sizeof(int16_t));
    memcpy(SS_S16.gru3_gru_state, test_state_in3_S16, SS_S16.model->gru3_hidden_size * sizeof(int16_t));
    
    computeSignalSifterModel(&SS, gains, test_input);
    computeSignalSifterModel_S16(&SS_S16, gains_S16, test_input_S16);
#if ENABLE_SS_MONITOR
    printSignalSifterStats(SS.monitor);
#endif
    
    float mse_state1 = check_vectors(SS.gru1_gru_state, (float *) test_state_out1, SS.model->gru1_hidden_size);
    float mse_state2 = check_vectors(SS.gru2_gru_state, (float *) test_state_out2, SS.model->gru2_hidden_size);
    float mse_state3 = check_vectors(SS.gru3_gru_state, (float *) test_state_out3, SS.model->gru3_hidden_size);
    float mse_out = check_vectors(gains,  (float *)test_output, IO_SIZE);
    printf("Floating-point inference MSE:\n");
    printf("\tMSE(h1) = %e\tMSE(h2) = %e\tMSE(h3) = %e\tMSE(out) = %e\n\n", mse_state1, mse_state2, mse_state3, mse_out);

    float mse_state1_S16 = check_vec_S16(SS_S16.gru1_gru_state, (float *) test_state_out1, SS.model->gru1_hidden_size, 15);
    float mse_state2_S16 = check_vec_S16(SS_S16.gru2_gru_state, (float *) test_state_out2, SS.model->gru2_hidden_size, 15);
    float mse_state3_S16 = check_vec_S16(SS_S16.gru3_gru_state, (float *) test_state_out3, SS.model->gru3_hidden_size, 15);
    float mse_out_S16 = check_vec_S16(gains_S16,  (float *)test_output, IO_SIZE, 15);
    printf("Fixed-point 16-bit inference MSE:\n");
    printf("\tMSE(h1) = %e\tMSE(h2) = %e\tMSE(h3) = %e\tMSE(out) = %e\n", mse_state1_S16, mse_state2_S16, mse_state3_S16, mse_out_S16);
    
    /*
    print_vector(SS.gru1_gru_state, SS.model->gru1_hidden_size, "GRU1 State");
    print_vector((float *) test_state1, SS.model->gru1_hidden_size, "reference GRU1 State");
    
    print_vector(SS.gru2_gru_state, SS.model->gru2_hidden_size, "GRU2 State");
    print_vector((float *) test_state2, SS.model->gru2_hidden_size, "reference GRU2 State");
    
    print_vector(SS.gru3_gru_state, SS.model->gru3_hidden_size, "GRU3 State");
    print_vector((float *) test_state3, SS.model->gru3_hidden_size, "reference GRU3 State");
    
    print_vector(gains, IO_SIZE, "Model Output");
    print_vector((float *) test_output, IO_SIZE, "reference output");
    */
    
    destroySignalSifterModel(&SS);
    return 0;
}
