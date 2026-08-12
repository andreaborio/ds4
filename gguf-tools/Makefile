CC ?= cc
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal
QUALITY_PROFILE := build/metal-$(UNAME_M)/obj
QUALITY_TARGETS := $(addprefix $(QUALITY_PROFILE)/,ds4.o ds4_build.o ds4_ssd.o ds4_profile.o ds4_expert_store.o ds4_qwen.o ds4_qwen_unicode.o ds4_qwen_expert_group.o ds4_metal.o)
QUALITY_OBJS := $(addprefix ../$(QUALITY_PROFILE)/,ds4.o ds4_build.o ds4_ssd.o ds4_profile.o ds4_expert_store.o ds4_qwen.o ds4_qwen_unicode.o ds4_qwen_expert_group.o ds4_metal.o)
QUALITY_LINK = $(CC) $(CFLAGS) -I.. -o
else
NATIVE_CPU_FLAG ?= -march=native
QUALITY_LDLIBS := -lm -pthread
QUALITY_TARGETS := ds4_cpu.o ds4_build_cpu.o ds4_ssd.o ds4_profile.o ds4_expert_store.o ds4_qwen.o ds4_qwen_unicode.o
QUALITY_OBJS := ../ds4_cpu.o ../ds4_build_cpu.o ../ds4_ssd.o ../ds4_profile.o ../ds4_expert_store.o ../ds4_qwen.o ../ds4_qwen_unicode.o
QUALITY_LINK = $(CC) $(CFLAGS) -DDS4_NO_GPU -I.. -o
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quality-score

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)

clean:
	rm -f deepseek4-quantize quality-testing/score_official
