cmake_minimum_required(VERSION 3.10)
project(JumpMLNR)

# Define options for enabling/disabling features
option(ENABLE_PROFILING "Enable profiling" OFF)
option(USE_FLOAT32_SIGNALSIFTER "Use float32 signal sifter" OFF)

# Set compiler flags based on options
if (ENABLE_PROFILING)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DENABLE_PROFILING")
endif()

if (USE_FLOAT32_SIGNALSIFTER)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_FLOAT32_SIGNALSIFTER")
endif()

# -DUSE_NEON -DENABLE_PROFILING -DUSE_FLOAT32_SIGNALSIFTER
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -INLINE:requested  -fPIC -ffunction-sections -fdata-sections -W -Wall -Os -O2 -Wno-sign-compare -Wno-unused-parameter")

include_directories(third-party/kissFFT include)

# Define PTJ_MODEL_NAME
set(PTJ_MODEL_NAME "${CMAKE_CURRENT_SOURCE_DIR}/models/pretrained_models/jumpmlnr_330k.ptj")

# Separate KissFFT sources
file(GLOB KISS_FFT_SRC "third-party/kissFFT/*.c")

# Project source files
file(GLOB SRC_FILES "src/*.c")

# Create KissFFT library
add_library(kissFFT STATIC ${KISS_FFT_SRC})

# Create main library
add_library(jumpmlnr STATIC ${SRC_FILES})

# Link library with KissFFT

target_link_libraries(jumpmlnr PUBLIC kissFFT m)

# Executable creation for test_nr
add_executable(testnr "test/testnr/JumpML_NR_demo.c")
target_link_libraries(testnr jumpmlnr kissFFT m)

# test target: testss
add_executable(testss "test/testss/test_signalsifter.c")
target_include_directories(testss PRIVATE "test/testss")
target_link_libraries(testss jumpmlnr kissFFT m)

# test target: test_nndsp
add_executable(test_nndsp "test/test_nndsp/main.c")
target_include_directories(test_nndsp PRIVATE "test/test_nndsp")
target_link_libraries(test_nndsp jumpmlnr kissFFT m)

# test target: test_resample
add_executable(test_resample "test/test_resample/main.c")
target_link_libraries(test_resample jumpmlnr m)

# utility target: compare_audio
add_executable(compare_audio "test/compare_audio/main.c")
target_link_libraries(compare_audio jumpmlnr m)

# Custom target: convert_model
add_custom_target(convert_model
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/models/convert_model.py -m ${PTJ_MODEL_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Custom target: draw_system
add_custom_target(draw_system
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/models/draw_system.py ${PTJ_MODEL_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Custom target: check
add_custom_target(check
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target convert_model
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target draw_system
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target jumpmlnr
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target testnr
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target testss
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_nndsp
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/denoise.sh ${CMAKE_CURRENT_SOURCE_DIR}/data/outdoor_mix.wav /tmp/output.wav 0.5 -30
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/models/run_prediction.py -m ${PTJ_MODEL_NAME} -i ${CMAKE_CURRENT_SOURCE_DIR}/data/outdoor_mix.wav -o /tmp/out_pytorch.wav -g -30 -n 0.5
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/models/utils/compare_wave_files.py /tmp/out_pytorch.wav /tmp/output.wav
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/denoise.sh ${CMAKE_CURRENT_SOURCE_DIR}/data/outdoor_mix.wav /tmp/output.wav 0.75 -40
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/models/run_prediction.py -m ${PTJ_MODEL_NAME} -i ${CMAKE_CURRENT_SOURCE_DIR}/data/outdoor_mix.wav -o /tmp/out_pytorch.wav -g -40 -n 0.75
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/models/utils/compare_wave_files.py /tmp/out_pytorch.wav /tmp/output.wav
    COMMAND ${CMAKE_BINARY_DIR}/testss
    COMMAND ${CMAKE_BINARY_DIR}/test_nndsp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)