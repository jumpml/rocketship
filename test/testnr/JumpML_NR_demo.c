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
//  JumpML_NR_demo.c
//
//  Created by Raghavendra Prabhu on 10/13/21.
//

#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include "jumpml_nr.h"
#include "jumpml_nr_tuning.h"


void usage(char* progname) {
    fprintf(stderr, "usage:\n"
    "%s [-n naturalness] [-m min_gain] -i <input file> -o <output file> \n"
    "   -i input_file:  noisy input file path \n"
    "   -o output_file: denoised output file path \n"
    "   -n naturalness: optional float number between 0 (max suppression) and 1 (most natural). Default: 0.5\n"
    "   -m min_gain:    optional minimum suppression gain floor in dB in [-60, 0] dB. Default: -40 dB\n"
    "   -r sample_rate: optional sample rate for input/output in {8000, 16000}. Default: 16000Hz\n"
    "   -h:             print out this help message\n", progname);
}


int main(int argc, char **argv)
{
    FILE *fin, *fout;
    char *fname_in, *fname_out;
    int required_args = 0;
    int opt;
    int16_t input_S16[JUMPML_NR_FRAME_SIZE];
    int16_t output_S16[JUMPML_NR_FRAME_SIZE];
    int8_t jmpnrStBuf[sizeof(DSP_JMPNR_ST_STRU)];
    
    float naturalness = JUMPML_NR_NATURALNESS;
    float min_gain = powf(10, JUMPML_NR_MIN_GAIN/10);
    int sample_rate = 16000; // Default sample rate
    float val;
    
//    DSP_JMPNR_ST_STRU jmpNR;
    int frameCount = 0;
    
    void* jmpnr_st_stru = (void *) jmpnrStBuf;
    while( (opt = getopt(argc, argv, ":hn:m:i:o:r:")) != -1 )
    {
        switch(opt)
        {
            case 'h':
                usage(argv[0]);
                return 1;
            case 'i':
                fname_in = optarg;
                required_args +=1;
                break;
            
            case 'o':
                fname_out = optarg;
                required_args +=1;
                break;
                
            case 'n':
                val = atof(optarg);
                if (val <= 1.0f && val >=0.0f)
                {
                    naturalness = val;
//                    printf("Naturalness = %.4f\n", naturalness);
                }
                else
                {
                    printf("Naturalness value (%f) outside range [0,1]. Using default value: %f\n", val, JUMPML_NR_NATURALNESS);
                }
                break;
            case 'm':
                val = atof(optarg);
                if (val <= 0.0f  && val >= -60.0f)
                {
                    min_gain = powf(10, val/10);
//                    printf("Min gain (linear) = %f\n", min_gain);
                }
                else
                {
                    printf("Min gain value (%f) outside range [-60,0] dB. Using default value: %f \n", val, JUMPML_NR_MIN_GAIN);
                }
                break;
            case 'r':
                val = atoi(optarg);
                if (val == 8000 || val == 16000)
                {
                    sample_rate = val;
                }
                else
                {
                    printf("Sample rate must be either 8000 or 16000 Hz. Using default: 16000 Hz\n");
                }
                break;
            case ':':
                printf("Option needs a value\n");
                return 1;
            case '?':
                printf("Unknown option: %c\n", optopt);
        }
    }

    if (required_args != 2) {
        printf("Input and output filenames are required.\n");
        usage(argv[0]);
        return 1;
    }

    fin = fopen(fname_in, "rb");
    fout = fopen(fname_out, "wb");

    jumpml_nr_init(jmpnr_st_stru, naturalness, min_gain);
    
    while (1) {
        fread(input_S16, sizeof(short), JUMPML_NR_FRAME_SIZE, fin);
        if (feof(fin)) break;
        jumpml_nr_proc(output_S16, input_S16, jmpnr_st_stru, sample_rate);
        frameCount = frameCount + 1;
//        if (frameCount == 10)
//        	break;
        fwrite(output_S16, sizeof(short), JUMPML_NR_FRAME_SIZE, fout);
    }
    
    fclose(fin);
    fclose(fout);
    return 0;
}
