#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "utils.h"


int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <file1.raw> <file2.raw>\n", argv[0]);
        return 1;
    }

    FILE *file1 = fopen(argv[1], "rb");
    FILE *file2 = fopen(argv[2], "rb");

    if (!file1 || !file2)
    {
        printf("Error opening files.\n");
        return 1;
    }

    fseek(file1, 0, SEEK_END);
    fseek(file2, 0, SEEK_END);

    long file1_size = ftell(file1);
    long file2_size = ftell(file2);

    if (file1_size != file2_size)
    {
        printf("Files are of different sizes.\n");
        fclose(file1);
        fclose(file2);
        return 1;
    }

    rewind(file1);
    rewind(file2);

    int16_t *buffer1 = (int16_t *)malloc(file1_size);
    int16_t *buffer2 = (int16_t *)malloc(file2_size);

    if (!buffer1 || !buffer2)
    {
        printf("Memory allocation failed.\n");
        fclose(file1);
        fclose(file2);
        return 1;
    }

    fread(buffer1, 1, file1_size, file1);
    fread(buffer2, 1, file2_size, file2);

    int num_samples = file1_size / sizeof(int16_t);

    float mse = check_vec_S16_S16(buffer1, buffer2, num_samples);

    printf("Mean Squared Error: %f\n", mse);

    free(buffer1);
    free(buffer2);
    fclose(file1);
    fclose(file2);

    return 0;
}
