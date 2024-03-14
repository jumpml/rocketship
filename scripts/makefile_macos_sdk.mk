ARCH := $(shell uname -m)
OS := $(shell uname -s)
CC=clang
CFLAGS=-Wall -Wextra -Iinclude -O3 -Wno-sign-compare -DUSE_FLOAT32_SIGNALSIFTER
LIBDIR=build

ifeq ($(ARCH), x86_64)
	LIB_NAME := jumpmlnr_x86_64
else
	LIB_NAME := jumpmlnr_arm64
endif

EXECUTABLE=build/testnr
RUN_COMMAND=denoise.sh data/outdoor_mix.wav data/output.wav 0.5 -30

all: $(EXECUTABLE)

$(EXECUTABLE): src/JumpML_NR_demo.c src/jumpml_nr.c
	$(CC) $(CFLAGS) -o $@ $^ -L$(LIBDIR) -l$(LIB_NAME)

profile: $(EXECUTABLE)
	time ./$(RUN_COMMAND)

.PHONY: clean

clean:
	rm -f $(EXECUTABLE)

