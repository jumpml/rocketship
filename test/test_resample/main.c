#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "resample.h"
#include "utils.h"

#define BUFFER_SIZE 1024

void process_file(const char* input_file, const char* output_file, int input_rate, int output_rate) {
    FILE *fin, *fout;
    short input_buffer_S16[BUFFER_SIZE];
    float float_buffer[BUFFER_SIZE];
    float resampled_buffer_F32[BUFFER_SIZE * 2];
    int16_t resampled_buffer_S16[BUFFER_SIZE * 2];

    float last_sample = 0.0f;
    int16_t last_sample_S16 = 0;
    size_t read_samples, write_samples;

    fin = fopen(input_file, "rb");
    if (!fin) {
        perror("Error opening input file");
        exit(1);
    }

    fout = fopen(output_file, "wb");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        exit(1);
    }

    while ((read_samples = fread(input_buffer_S16, sizeof(short), BUFFER_SIZE, fin)) > 0) {
        // Convert to float
        for (size_t i = 0; i < read_samples; i++) {
            float_buffer[i] = input_buffer_S16[i] / 32768.0f;
        }

        if (input_rate == 8000 && output_rate == 16000) {
            upsample_F32(float_buffer, resampled_buffer_F32, read_samples, &last_sample);
            upsample_S16(input_buffer_S16, resampled_buffer_S16, read_samples, &last_sample_S16);
            write_samples = read_samples * 2;
        } else if (input_rate == 16000 && output_rate == 8000) {
            downsample_F32(float_buffer, resampled_buffer_F32, read_samples / 2);
            downsample_S16(input_buffer_S16, resampled_buffer_S16, read_samples / 2);
            write_samples = read_samples / 2;
        } else {
            memcpy(resampled_buffer_F32, float_buffer, read_samples * sizeof(float));
            memcpy(resampled_buffer_S16, input_buffer_S16, read_samples * sizeof(float));
            write_samples = read_samples;
        }

        float mse = check_vec_S16(resampled_buffer_S16, resampled_buffer_F32, write_samples, 15);
        printf("MSE = %f\n", mse);

        // Convert back to short
        for (size_t i = 0; i < write_samples; i++) {
            short sample = (short)(resampled_buffer_F32[i] * 32767.0f);
            fwrite(&sample, sizeof(short), 1, fout);
        }
    }

    fclose(fin);
    fclose(fout);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input_file> <output_file> <input_rate> <output_rate>\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int input_rate = atoi(argv[3]);
    int output_rate = atoi(argv[4]);

    if (input_rate != 8000 && input_rate != 16000) {
        fprintf(stderr, "Input rate must be either 8000 or 16000\n");
        return 1;
    }

    if (output_rate != 8000 && output_rate != 16000) {
        fprintf(stderr, "Output rate must be either 8000 or 16000\n");
        return 1;
    }

    if (input_rate == output_rate) {
        fprintf(stderr, "Input and output rates are the same. No resampling needed.\n");
        return 1;
    }

    process_file(input_file, output_file, input_rate, output_rate);
    printf("Resampling complete. Output written to %s\n", output_file);

    return 0;
}