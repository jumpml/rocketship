JUMPML DEMO SDK
================

MacOS (M1) static library and header files to build JumpML Noise Reduction audio processing. 


TECHNICAL SPECIFICATIONS
=========================
FFT Size (STFT/ISTFT) = 240 samples @ 16 kHz
Hop-Length  = 120 samples

Application can take either 120 samples Frame Size or 240 samples (internally it runs two NN predictions). 


TUNING PARAMETERS
=================
# NATURALNESS in [0,1]. 0 means maximum noise suppression, 1 means more natural  (more speech preservation) 
# MINGAIN in [-60, 0] dB. 0 dB means no suppression.


DIRECTORY STRUCTURE 
===================
├── Makefile
├── data
│   ├── outdoor_mix.wav
│   └── output.wav
├── denoise.sh
├── include
│   ├── common_def.h
│   ├── dsp_processing.h
│   ├── dsplib.h
│   ├── fixed_point_math.h
│   ├── jumpml_nr.h
│   ├── jumpml_nr_tuning.h
│   ├── kiss_fft.h
│   ├── kiss_fftr.h
│   ├── nn_layers.h
│   ├── nnlib.h
│   ├── noise_reduction.h
│   ├── signalsifter.h
│   ├── signalsifter_config.h
│   ├── tanh_table.h
│   ├── tanh_table_S16.h
│   └── utils.h
├── lib
│   └── lib_jumpml_nr_x86.a
├── readme.txt
├── src
│   ├── JumpML_NR_demo.c
│   └── jumpml_nr.c
├── testnr
└── version.txt

Relevant headers: jumpml_nr.h, jumpml_nr_tuning.h 
Example program: JumpML_NR_demo.c


USAGE
=====
Prerequisite: Please install SoX (brew install sox)

Step 1: Build testnr executable
make clean
make all     # build testnr exe 
make profile # runs denoise.sh script with sample noisy speech data and produces data/output.wav 

Step 2: Use command-line executable (denoise.sh) to denoise your wave files ("chmod +x", if necessary):
./denoise.sh input.wav output.wav naturalness mingain
./denoise.sh data/outdoor_mix.wav data/output.wav 0.5 -30


LIBRARY USE
===========
Please see JumpML_NR_demo.c for example. These are the steps

1. Include header files: jumpml_nr.h and jumpml_nr_tuning.h
2. Define/alloc a state memory buffer: 
	int8_t jmpnrStBuf[sizeof(DSP_JMPNR_ST_STRU)];
 	void* jmpnr_st_stru = (void *) jmpnrStBuf;
3. Define tuning variables naturalness and min_gain (can be overwritten by user ideally)
    float naturalness = JUMPML_NR_NATURALNESS;
    float min_gain = powf(10, JUMPML_NR_MIN_GAIN/10);
4. Call init routine
    jumpml_nr_init(jmpnr_st_stru, naturalness, min_gain);

5. Call Process function with int16_t (PCM) audio data 120 samples
   jumpml_nr_proc(output_S16, input_S16, JUMPML_NR_APP_FRAME_SIZE, jmpnr_st_stru);





