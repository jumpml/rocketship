# Architecture flags for gcc
ARCH := $(shell uname -m)
OS := $(shell uname -s)
LIBDIR := build
# C -compiler name, can be replaced by another compiler(replace gcc)
ifeq ($(OS), Windows_NT)
    #CC = xt-xcc
	CC = xt-clang
	AS = xt-as
	RM  = rmdir /s/q
	CFLAGS = -W -Wall -Os -O2 -mlongcalls -DXTENSA -mtext-section-literals 
	DEL = del
	OBJDIR := build\obj
	OBJDIR_CMD = mkdir -p $(OBJDIR)\src $(OBJDIR)\third-party\kissFFT
	LIB_NAME := jumpmlnr_hifi
	DECLIB = $(LIBDIR)\lib$(LIB_NAME).a
else
	CC = gcc
    AS = gcc
	RM = rm -rf
	CFLAGS = -W -Wall -Os -O2 -Wno-sign-compare -Wno-unused-parameter
	DEL = rm -f
	OBJDIR := build/obj
	OBJDIR_CMD = mkdir -p $(OBJDIR)/src $(OBJDIR)/third-party/kissFFT
	ifeq ($(ARCH), x86_64)
		LIB_NAME := jumpmlnr_x86
	else
		LIB_NAME := jumpmlnr_arm64
	endif
	DECLIB := $(LIBDIR)/lib$(LIB_NAME).a
endif

# MACRO for creating library that includes all the object files
AR  = ar rcs

# show the path to compiler, where to find header files and libraries
S_SRC := s_src/

# C_SRCS = $(wildcard src/*.c) \
# 		 $(wildcard src/kissFFT/*.c)

C_SRCS = $(filter-out src/jumpml_nr.c, $(wildcard src/*.c) \
		 $(wildcard third-party/kissFFT/*.c))

S_SRCS +=
INCLUDE = -Ithird-party/kissFFT/ -Iinclude

CFLAGS  += -INLINE:requested -ffunction-sections -fdata-sections $(INCLUDE) -DUSE_FLOAT32_SIGNALSIFTER
ASFLAGS  = -W -mlongcalls 
LDFLAGS := -L$(LIBDIR) -l$(LIB_NAME) -lm

# object files will be generated from .c sourcefiles
OBJS    = $(C_SRCS:.c=.o)
OBJSS   = $(S_SRCS:.s=.o)

ALLOBJ = $(OBJS) $(OBJSS)
DIROBJ = $(addprefix $(OBJDIR)/, $(ALLOBJ))

EXECUTABLE=build/testnr

all: build_library $(EXECUTABLE)

convert_model: 
		python models/convert_model.py -m $(PTJ_MODEL_NAME)	

.PHONY: check 
PTJ_MODEL_NAME = models/pretrained_models/jumpmlnr_pro.ptj
ifneq ($(OS),Windows_NT)  # Check if the operating system is not Windows
check: clean convert_model build_library $(EXECUTABLE)
	scripts/denoise.sh data/outdoor_mix.wav /tmp/output.wav 0.5 -30
	python models/run_prediction.py  -m $(PTJ_MODEL_NAME)  -i data/outdoor_mix.wav -o /tmp/out_pytorch.wav -g -30 -n 0.5
	python models/utils/compare_wave_files.py /tmp/out_pytorch.wav /tmp/output.wav

	scripts/denoise.sh data/outdoor_mix.wav /tmp/output.wav 0.75 -40
	python models/run_prediction.py  -m $(PTJ_MODEL_NAME)  -i data/outdoor_mix.wav -o /tmp/out_pytorch.wav -g -40 -n 0.75
	python models/utils/compare_wave_files.py /tmp/out_pytorch.wav /tmp/output.wav
endif

build_library: $(DECLIB) 
	$(AR) $(DECLIB) $(DIROBJ)

$(OBJDIR):
	$(OBJDIR_CMD)

$(DECLIB): $(OBJDIR) $(OBJS) $(OBJSS)


$(EXECUTABLE): test/testnr/JumpML_NR_demo.c src/jumpml_nr.c $(DECLIB)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) 

$(OBJS): %.o: %.c
	$(CC) -c $(CFLAGS) $< -o $(OBJDIR)/$@ >$(OBJDIR)/$(@:.o=.lst)


$(OBJSS): %.o: $(S_SRC)%.s
	$(CC) -c $(ASFLAGS) $< -o $(OBJDIR)/$@ >$(OBJDIR)/$(@:.o=.lst)

.PHONY: clean
clean:
	$(DEL) $(DECLIB)
	$(RM) $(OBJDIR)
	$(DEL) $(EXECUTABLE)
