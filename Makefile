CC ?= cc
.DEFAULT_GOAL := all
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc
# Qwen's stable softmax rejects non-finite logits; retain that branch while
# keeping the remaining fast-math optimizations used by the scalar CPU path.
QWEN_CFLAGS ?= -fno-finite-math-only
DEPFLAGS ?= -MMD -MP

BUILD_GIT_SHA ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
BUILD_GIT_SUFFIX ?= $(shell test -z "$$(git status --porcelain --untracked-files=normal 2>/dev/null)" || printf '%s' -dirty)
CFLAGS += -DDS4_BUILD_GIT_SHA=\"$(BUILD_GIT_SHA)$(BUILD_GIT_SUFFIX)\"

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)

BUILD_ROOT ?= build
HEBRUS_PROGRAMS := hebrus hebrus-server hebrus-bench hebrus-eval hebrus-agent
DS4_PROGRAMS := ds4 ds4-server ds4-bench ds4-eval ds4-agent
PROGRAMS := $(HEBRUS_PROGRAMS) $(DS4_PROGRAMS)

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
INSTALL ?= install
INSTALL_MODE ?= 0755
INSTALL_DEST_BINDIR = $(DESTDIR)$(BINDIR)
INSTALL_SOURCE_PROGRAMS = $(addprefix $(INSTALL_SOURCE_BINDIR)/,$(HEBRUS_PROGRAMS))

.PHONY: all help clean test model-free-test premerge context-audit doc-links \
	brand-boundary-audit brand-boundary-test \
	release-contract release-contract-test \
	imatrix-dataset-check prompt-fixture-check cpu FORCE \
	metal build-isolation-test q4k-dot-test qwen-metadata-test \
	qwen-reference-test qwen-unicode-test qwen-tokenizer-test \
	qwen-expert-group-test expert-store-test metal-ssd-profile-test \
	download-model-test capabilities-test command-alias-test \
	visible-identity-test \
	install uninstall install-test $(PROGRAMS) \
	ds4_test ds4_agent_test

download-model-test: tests/test_download_model.sh download_model.sh \
		docs/contracts/qwen-release.json
	sh tests/test_download_model.sh

release-contract: tools/qwen_release_contract.py \
		docs/contracts/qwen-release.json README.md CONTRIBUTING.md \
		QA_BEFORE_RELEASES.md docs/contracts/RUNTIME_SUPPORT.md \
		docs/qwen-expert-major-store.md download_model.sh \
		tests/test_download_model.sh
	python3 tools/qwen_release_contract.py

release-contract-test: release-contract tools/qwen_release_contract.py \
		tests/test_qwen_release_contract.py
	python3 tests/test_qwen_release_contract.py

brand-boundary-audit: tools/brand_boundary_audit.py tools/brand_boundary.json
	python3 tools/brand_boundary_audit.py --check

brand-boundary-test: tools/brand_boundary_audit.py tests/test_brand_boundary_audit.py
	python3 tests/test_brand_boundary_audit.py

ifeq ($(UNAME_S),Darwin)

# A build profile owns every object and binary it produces.  In particular, a
# CPU build can never satisfy a Metal prerequisite (or replace a Metal binary).
METAL_PROFILE := metal-$(UNAME_M)
CPU_PROFILE := cpu-$(UNAME_M)
METAL_OBJDIR := $(BUILD_ROOT)/$(METAL_PROFILE)/obj
METAL_BINDIR := $(BUILD_ROOT)/$(METAL_PROFILE)/bin
CPU_OBJDIR := $(BUILD_ROOT)/$(CPU_PROFILE)/obj
CPU_BINDIR := $(BUILD_ROOT)/$(CPU_PROFILE)/bin
INSTALL_SOURCE_BINDIR := $(METAL_BINDIR)
INSTALL_BACKEND := metal

METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal

METAL_CORE_OBJS := $(addprefix $(METAL_OBJDIR)/,ds4.o ds4_build.o ds4_ssd.o ds4_profile.o ds4_expert_store.o ds4_qwen.o ds4_qwen_unicode.o ds4_qwen_expert_group.o ds4_metal.o)
CPU_CORE_OBJS := $(addprefix $(CPU_OBJDIR)/,ds4.o ds4_build.o ds4_ssd.o ds4_profile.o ds4_expert_store.o ds4_qwen.o ds4_qwen_unicode.o)

METAL_BINS := $(addprefix $(METAL_BINDIR)/,$(PROGRAMS))
CPU_BINS := $(addprefix $(CPU_BINDIR)/,$(PROGRAMS))

METAL_TEST_BINS := \
	$(METAL_BINDIR)/ds4_test \
	$(METAL_BINDIR)/ds4_agent_test \
	$(METAL_BINDIR)/test_q4k_dot \
	$(METAL_BINDIR)/test_q4k_top8 \
	$(METAL_BINDIR)/test_qwen_session \
	$(METAL_BINDIR)/test_qwen_gdn_ref \
	$(METAL_BINDIR)/test_qwen_attention_ref \
	$(METAL_BINDIR)/test_qwen_state \
	$(METAL_BINDIR)/test_qwen_unicode \
	$(METAL_BINDIR)/test_qwen_tokenizer \
	$(METAL_BINDIR)/test_qwen_expert_group \
	$(METAL_BINDIR)/test_expert_store \
	$(METAL_BINDIR)/test_metal_ssd_profile \
	$(METAL_BINDIR)/test_ssd_residency \
	$(METAL_BINDIR)/test_visible_identity

all: metal

help:
	@echo "Hebrus build targets:"
	@echo "  make / make metal Build Metal and publish ./hebrus* plus ./ds4* aliases"
	@echo "  make cpu          Build CPU-only commands in $(CPU_BINDIR); keep root Metal links"
	@echo "  make test         Build and run the Metal test suite"
	@echo "  make model-free-test"
	@echo "                    Run all Metal gates that do not require a GGUF"
	@echo "  make build-isolation-test"
	@echo "                    Prove Metal -> CPU -> Metal cannot mix artifacts"
	@echo "  make install      Install commands under DESTDIR+$(BINDIR)"
	@echo "  make uninstall    Remove only the ten installed command paths"
	@echo "  make install-test Verify staged install layout and capabilities"
	@echo "  make brand-boundary-audit"
	@echo "                    Reject unclassified or increased legacy brand tokens"
	@echo "  make release-contract"
	@echo "                    Reject Qwen release identity drift"
	@echo "  make premerge     Run context/docs, isolation, and model-free gates"
	@echo "  make clean        Remove build outputs and published root binaries"

# Root binaries are a Metal-only compatibility surface on macOS.  These targets
# are phony so an old regular CPU binary is replaced even when its timestamp is
# newer than the namespaced Metal binary.
metal: $(PROGRAMS) $(METAL_BINS)

hebrus ds4: $(METAL_BINDIR)/hebrus
hebrus-server ds4-server: $(METAL_BINDIR)/hebrus-server
hebrus-bench ds4-bench: $(METAL_BINDIR)/hebrus-bench
hebrus-eval ds4-eval: $(METAL_BINDIR)/hebrus-eval
hebrus-agent ds4-agent: $(METAL_BINDIR)/hebrus-agent

$(PROGRAMS):
	@rm -f "$@"
	@ln -s "$<" "$@"

cpu: $(CPU_BINS)
	@echo "CPU-only binaries: $(CPU_BINDIR)"

$(METAL_BINDIR)/hebrus: \
	$(METAL_OBJDIR)/ds4_cli.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/linenoise.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/hebrus-server: \
	$(METAL_OBJDIR)/ds4_server.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_kvstore.o $(METAL_OBJDIR)/rax.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/hebrus-bench: \
	$(METAL_OBJDIR)/ds4_bench.o $(METAL_OBJDIR)/ds4_help.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/hebrus-eval: \
	$(METAL_OBJDIR)/ds4_eval.o $(METAL_OBJDIR)/ds4_help.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/hebrus-agent: \
	$(METAL_OBJDIR)/ds4_agent.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_web.o $(METAL_OBJDIR)/ds4_kvstore.o \
	$(METAL_OBJDIR)/linenoise.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/ds4: $(METAL_BINDIR)/hebrus
$(METAL_BINDIR)/ds4-server: $(METAL_BINDIR)/hebrus-server
$(METAL_BINDIR)/ds4-bench: $(METAL_BINDIR)/hebrus-bench
$(METAL_BINDIR)/ds4-eval: $(METAL_BINDIR)/hebrus-eval
$(METAL_BINDIR)/ds4-agent: $(METAL_BINDIR)/hebrus-agent

$(addprefix $(METAL_BINDIR)/,$(DS4_PROGRAMS)):
	@rm -f "$@"
	@ln -s "$(notdir $<)" "$@"

$(CPU_BINDIR)/hebrus: \
	$(CPU_OBJDIR)/ds4_cli.o $(CPU_OBJDIR)/ds4_help.o \
	$(CPU_OBJDIR)/linenoise.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/hebrus-server: \
	$(CPU_OBJDIR)/ds4_server.o $(CPU_OBJDIR)/ds4_help.o \
	$(CPU_OBJDIR)/ds4_kvstore.o $(CPU_OBJDIR)/rax.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/hebrus-bench: \
	$(CPU_OBJDIR)/ds4_bench.o $(CPU_OBJDIR)/ds4_help.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/hebrus-eval: \
	$(CPU_OBJDIR)/ds4_eval.o $(CPU_OBJDIR)/ds4_help.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/hebrus-agent: \
	$(CPU_OBJDIR)/ds4_agent.o $(CPU_OBJDIR)/ds4_help.o \
	$(CPU_OBJDIR)/ds4_web.o $(CPU_OBJDIR)/ds4_kvstore.o \
	$(CPU_OBJDIR)/linenoise.o $(CPU_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(CPU_BINDIR)/ds4: $(CPU_BINDIR)/hebrus
$(CPU_BINDIR)/ds4-server: $(CPU_BINDIR)/hebrus-server
$(CPU_BINDIR)/ds4-bench: $(CPU_BINDIR)/hebrus-bench
$(CPU_BINDIR)/ds4-eval: $(CPU_BINDIR)/hebrus-eval
$(CPU_BINDIR)/ds4-agent: $(CPU_BINDIR)/hebrus-agent

$(addprefix $(CPU_BINDIR)/,$(DS4_PROGRAMS)):
	@rm -f "$@"
	@ln -s "$(notdir $<)" "$@"

$(METAL_OBJDIR)/%.o: %.c
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

# Textual implementation partitions stay in the ds4.c translation unit.  Keep
# the dependency explicit as well as in the generated .d file so incremental
# builds remain correct before dependency metadata exists.
$(METAL_OBJDIR)/ds4.o: runtime/ds4_glm_graph.inc \
		runtime/ds4_deepseek_cache_phase.inc

$(CPU_OBJDIR)/ds4.o: runtime/ds4_glm_graph.inc \
		runtime/ds4_deepseek_cache_phase.inc

$(CPU_OBJDIR)/%.o: %.c
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

# Build provenance is intentionally refreshed on every invocation.  Keeping it
# in this tiny object prevents a clean/dirty transition from forcing the giant
# engine translation unit to rebuild while ensuring --build-info is truthful.
$(METAL_OBJDIR)/ds4_build.o: ds4_build.c ds4.h ds4_expert_store.h FORCE
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(CPU_OBJDIR)/ds4_build.o: ds4_build.c ds4.h ds4_expert_store.h FORCE
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/ds4_qwen.o: ds4_qwen.c ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(CPU_OBJDIR)/ds4_qwen.o: ds4_qwen.c ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/ds4_qwen_unicode.o: ds4_qwen_unicode.c ds4_qwen_unicode.h \
		ds4_qwen_unicode_data.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(CPU_OBJDIR)/ds4_qwen_unicode.o: ds4_qwen_unicode.c ds4_qwen_unicode.h \
		ds4_qwen_unicode_data.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -DDS4_NO_GPU $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/ds4_test_core.o: ds4.c ds4.h ds4_ssd.h ds4_profile.h \
		ds4_gpu.h ds4_qwen.h ds4_expert_store.h \
		ds4_qwen_unicode.h ds4_streaming_hotlist.inc \
		tests/internal/ds4_qwen_cpu_test_hooks.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -DDS4_NO_GPU \
		-DDS4_TEST_HOOKS -Wno-unused-function -Wno-unused-parameter \
		-c -o $@ $<

$(METAL_OBJDIR)/ds4_metal.o: ds4_metal.m ds4_gpu.h \
		ds4_qwen_expert_group.h runtime/ds4_metal_glm.inc $(METAL_SRCS)
	@mkdir -p "$(@D)"
	$(CC) $(OBJCFLAGS) $(DEPFLAGS) -c -o $@ ds4_metal.m

# These white-box implementation objects deliberately omit normal entrypoints
# or GPU consumers. Keep unused-only suppression scoped to the implementation
# variants until the remaining server/Qwen seams remove this coupling debt.
$(METAL_OBJDIR)/ds4_test.o: tests/ds4_test.c
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -Wno-unused-function -c -o $@ $<

$(METAL_OBJDIR)/ds4_agent_test.o: tests/ds4_agent_test.c \
		tests/internal/ds4_agent_unit.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/ds4_agent_test_impl.o: ds4_agent.c ds4.h ds4_ssd.h \
		ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h \
		tests/internal/ds4_agent_unit.h tests/internal/ds4_agent_unit.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -DDS4_AGENT_TEST \
		-DDS4_AGENT_TEST_NO_MAIN -Wno-unused-function -c -o $@ $<

$(METAL_OBJDIR)/test_q4k_dot.o: tests/test_q4k_dot.c
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -c -o $@ $<

$(METAL_OBJDIR)/test_q4k_top8.o: tests/test_q4k_top8.c \
		tests/internal/ds4_qwen_cpu_test_hooks.h ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -DDS4_NO_GPU \
		-DDS4_TEST_HOOKS -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_session.o: tests/test_qwen_session.c ds4.c ds4.h \
		ds4_ssd.h ds4_profile.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h runtime/ds4_glm_graph.inc \
		runtime/ds4_deepseek_cache_phase.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_tokenizer.o: tests/test_qwen_tokenizer.c ds4.c \
		ds4.h ds4_kvstore.h ds4_ssd.h ds4_profile.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h tests/qwen/qwen36_tokenizer_fixture.inc
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) $(DEPFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -c -o $@ $<

$(METAL_OBJDIR)/test_ssd_residency.o: tests/test_ssd_residency.c \
		ds4_ssd.h ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_expert_group.o: tests/test_qwen_expert_group.c \
		ds4_qwen_expert_group.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_expert_store.o: tests/test_expert_store.c \
		ds4_expert_store.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_metal_ssd_profile.o: tests/test_metal_ssd_profile.c \
		ds4_profile.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_gdn_ref.o: tests/test_qwen_gdn_ref.c ds4_qwen_ref.h \
		ds4_qwen.h tests/qwen/qwen36_gdn_golden.inc
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_attention_ref.o: tests/test_qwen_attention_ref.c \
		ds4_qwen_ref.h ds4_qwen.h tests/qwen/qwen36_attention_golden.inc
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_state.o: tests/test_qwen_state.c ds4_qwen.h
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/test_qwen_unicode.o: tests/test_qwen_unicode.c \
		ds4_qwen_unicode.h
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -I. -c -o $@ $<

$(METAL_OBJDIR)/ds4_qwen_ref.o: ds4_qwen_ref.c ds4_qwen_ref.h
	@mkdir -p "$(@D)"
	$(CC) -O2 -Wall -Wextra -std=c99 $(DEPFLAGS) -c -o $@ $<

$(METAL_BINDIR)/ds4_test: \
	$(METAL_OBJDIR)/ds4_test.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_kvstore.o $(METAL_OBJDIR)/rax.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/ds4_agent_test: \
	$(METAL_OBJDIR)/ds4_agent_test.o \
	$(METAL_OBJDIR)/ds4_agent_test_impl.o $(METAL_OBJDIR)/ds4_help.o \
	$(METAL_OBJDIR)/ds4_web.o $(METAL_OBJDIR)/ds4_kvstore.o \
	$(METAL_OBJDIR)/linenoise.o $(METAL_CORE_OBJS)
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(METAL_LDLIBS)

$(METAL_BINDIR)/test_q4k_dot: $(METAL_OBJDIR)/test_q4k_dot.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm -pthread

$(METAL_BINDIR)/test_q4k_top8: \
		$(METAL_OBJDIR)/test_q4k_top8.o $(METAL_OBJDIR)/ds4_test_core.o \
		$(METAL_OBJDIR)/ds4_build.o \
		$(METAL_OBJDIR)/ds4_ssd.o \
		$(METAL_OBJDIR)/ds4_profile.o \
		$(METAL_OBJDIR)/ds4_expert_store.o \
		$(METAL_OBJDIR)/ds4_qwen.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_qwen_session: \
		$(METAL_OBJDIR)/test_qwen_session.o $(METAL_OBJDIR)/ds4_build.o \
		$(METAL_OBJDIR)/ds4_ssd.o \
		$(METAL_OBJDIR)/ds4_profile.o \
		$(METAL_OBJDIR)/ds4_expert_store.o \
		$(METAL_OBJDIR)/ds4_qwen.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_qwen_tokenizer: \
		$(METAL_OBJDIR)/test_qwen_tokenizer.o $(METAL_OBJDIR)/ds4_kvstore.o \
		$(METAL_OBJDIR)/ds4_build.o \
		$(METAL_OBJDIR)/ds4_ssd.o \
		$(METAL_OBJDIR)/ds4_profile.o \
		$(METAL_OBJDIR)/ds4_expert_store.o \
		$(METAL_OBJDIR)/ds4_qwen.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_ssd_residency: \
	$(METAL_OBJDIR)/test_ssd_residency.o $(METAL_OBJDIR)/ds4_ssd.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_visible_identity: \
		tests/test_visible_identity.c hebrus_identity.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -I. -o $@ $<

$(CPU_BINDIR)/test_visible_identity: \
		tests/test_visible_identity.c hebrus_identity.h
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -DDS4_NO_GPU -I. -o $@ $<

visible-identity-test: $(METAL_BINDIR)/test_visible_identity \
		$(CPU_BINDIR)/test_visible_identity
	$(METAL_BINDIR)/test_visible_identity
	$(CPU_BINDIR)/test_visible_identity

$(METAL_BINDIR)/test_qwen_gdn_ref: \
		$(METAL_OBJDIR)/test_qwen_gdn_ref.o $(METAL_OBJDIR)/ds4_qwen_ref.o \
		$(METAL_OBJDIR)/ds4_qwen.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm

$(METAL_BINDIR)/test_qwen_attention_ref: \
		$(METAL_OBJDIR)/test_qwen_attention_ref.o $(METAL_OBJDIR)/ds4_qwen_ref.o \
		$(METAL_OBJDIR)/ds4_qwen.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm

$(METAL_BINDIR)/test_qwen_state: \
	$(METAL_OBJDIR)/test_qwen_state.o $(METAL_OBJDIR)/ds4_qwen.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^ -lm

$(METAL_BINDIR)/test_qwen_unicode: \
	$(METAL_OBJDIR)/test_qwen_unicode.o $(METAL_OBJDIR)/ds4_qwen_unicode.o
	@mkdir -p "$(@D)"
	$(CC) -O2 -o $@ $^

$(METAL_BINDIR)/test_qwen_expert_group: \
		$(METAL_OBJDIR)/test_qwen_expert_group.o \
		$(METAL_OBJDIR)/ds4_qwen_expert_group.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_expert_store: \
		$(METAL_OBJDIR)/test_expert_store.o \
		$(METAL_OBJDIR)/ds4_expert_store.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(METAL_BINDIR)/test_metal_ssd_profile: \
		$(METAL_OBJDIR)/test_metal_ssd_profile.o \
		$(METAL_OBJDIR)/ds4_profile.o
	@mkdir -p "$(@D)"
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen-expert-group-test: $(METAL_BINDIR)/test_qwen_expert_group
	$<

expert-store-test: $(METAL_BINDIR)/test_expert_store
	DS4_EXPERT_STORE_PROBE=$(METAL_BINDIR)/test_expert_store \
		python3 tests/test_expert_major.py

metal-ssd-profile-test: $(METAL_BINDIR)/test_metal_ssd_profile
	$<

# Preserve the documented direct test-runner commands without letting a CPU
# target publish over them.
ds4_test: $(METAL_BINDIR)/ds4_test
	@rm -f "$@"
	@ln -s "$<" "$@"

ds4_agent_test: $(METAL_BINDIR)/ds4_agent_test
	@rm -f "$@"
	@ln -s "$<" "$@"

q4k-dot-test: $(METAL_BINDIR)/test_q4k_dot
	$<

qwen-metadata-test: $(METAL_BINDIR)/ds4 tests/test_qwen_metadata.py
	python3 tests/test_qwen_metadata.py $(METAL_BINDIR)/ds4

qwen-reference-test: $(METAL_BINDIR)/test_qwen_gdn_ref \
		$(METAL_BINDIR)/test_qwen_attention_ref \
		$(METAL_BINDIR)/test_qwen_unicode
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	$(METAL_BINDIR)/test_qwen_gdn_ref
	$(METAL_BINDIR)/test_qwen_attention_ref
	$(METAL_BINDIR)/test_qwen_unicode

qwen-unicode-test: $(METAL_BINDIR)/test_qwen_unicode
	python3 tests/gen_qwen_unicode.py --check
	$(METAL_BINDIR)/test_qwen_unicode

qwen-tokenizer-test: $(METAL_BINDIR)/test_qwen_tokenizer
	$(METAL_BINDIR)/test_qwen_tokenizer

capabilities-test: metal tests/test_capabilities.py
	python3 tests/test_capabilities.py --bin-dir $(METAL_BINDIR) --backend metal

command-alias-test: metal tests/test_command_aliases.py
	python3 tests/test_command_aliases.py --bin-dir $(METAL_BINDIR) \
		--backend metal --layout profile

model-free-test: metal ds4_test ds4_agent_test $(METAL_BINDIR)/test_q4k_dot \
		$(METAL_BINDIR)/test_q4k_top8 \
		$(METAL_BINDIR)/test_qwen_session \
		$(METAL_BINDIR)/test_qwen_gdn_ref \
		$(METAL_BINDIR)/test_qwen_attention_ref \
		$(METAL_BINDIR)/test_qwen_state \
		$(METAL_BINDIR)/test_qwen_unicode \
		$(METAL_BINDIR)/test_qwen_tokenizer \
		$(METAL_BINDIR)/test_qwen_expert_group \
		$(METAL_BINDIR)/test_expert_store \
		$(METAL_BINDIR)/test_metal_ssd_profile \
		$(METAL_BINDIR)/test_ssd_residency download-model-test \
		visible-identity-test \
		tests/test_capabilities.py tests/test_command_aliases.py
	python3 tests/test_capabilities.py --bin-dir $(METAL_BINDIR) --backend metal
	python3 tests/test_command_aliases.py --bin-dir $(METAL_BINDIR) \
		--backend metal --layout profile
	DS4_BIN_DIR=$(METAL_BINDIR) sh tests/test_retired_distributed_flags.sh
	sh tests/test_benchmark_env_guard.sh
	$(METAL_BINDIR)/ds4-eval --self-test-extractors
	$(METAL_BINDIR)/ds4_agent_test
	$(METAL_BINDIR)/ds4_test --server
	$(METAL_BINDIR)/ds4_test --metal-kernels
	$(METAL_BINDIR)/ds4_test --metal-expert-pack
	$(METAL_BINDIR)/test_q4k_dot
	$(METAL_BINDIR)/test_q4k_top8
	$(METAL_BINDIR)/test_qwen_session
	$(METAL_BINDIR)/test_qwen_gdn_ref
	$(METAL_BINDIR)/test_qwen_attention_ref
	$(METAL_BINDIR)/test_qwen_state
	$(METAL_BINDIR)/test_qwen_unicode
	$(METAL_BINDIR)/test_qwen_tokenizer
	$(METAL_BINDIR)/test_qwen_expert_group
	DS4_EXPERT_STORE_PROBE=$(METAL_BINDIR)/test_expert_store \
		python3 tests/test_expert_major.py
	$(METAL_BINDIR)/test_metal_ssd_profile
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	$(METAL_BINDIR)/test_ssd_residency
	python3 tests/test_qwen_metadata.py $(METAL_BINDIR)/ds4

test: model-free-test
	$(METAL_BINDIR)/ds4_test

context-audit:
	python3 tools/context_audit.py

doc-links:
	python3 tools/check_doc_links.py

imatrix-dataset-check:
	python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py --check

prompt-fixture-check:
	python3 speed-bench/build_long_context_prompt.py --check


# Build isolation removes and rebuilds BUILD_ROOT, so model-free-test must start
# only after it completes even when an agent invokes `make -j premerge`.
premerge: context-audit doc-links brand-boundary-audit brand-boundary-test \
	release-contract release-contract-test \
	imatrix-dataset-check prompt-fixture-check build-isolation-test
	$(MAKE) model-free-test
	$(MAKE) install-test
	git diff --check

build-isolation-test: tests/test_build_isolation.sh tests/test_capabilities.py \
		tests/test_command_aliases.py
	MAKE="$(MAKE)" sh tests/test_build_isolation.sh

-include $(wildcard $(METAL_OBJDIR)/*.d $(CPU_OBJDIR)/*.d)

else

CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CPU_CORE_OBJS := ds4_cpu.o ds4_build_cpu.o ds4_ssd.o \
	ds4_profile.o ds4_expert_store.o ds4_qwen.o ds4_qwen_unicode.o
INSTALL_SOURCE_BINDIR := .
INSTALL_BACKEND := cpu

all: cpu

help:
	@echo "Hebrus build targets:"
	@echo "  make / make cpu          Build ./hebrus* plus ./ds4* aliases"
	@echo "  make test                Build and run tests"
	@echo "  make model-free-test     Run all tests that do not require a GGUF"
	@echo "  make install             Install commands under DESTDIR+$(BINDIR)"
	@echo "  make uninstall           Remove only the ten installed command paths"
	@echo "  make install-test        Verify staged install layout and capabilities"
	@echo "  make brand-boundary-audit"
	@echo "                           Reject unclassified or increased legacy brand tokens"
	@echo "  make release-contract    Reject Qwen release identity drift"
	@echo "  make premerge            Run repository audits and Linux CPU/model-free gates"
	@echo "  make clean               Remove build outputs"

cpu: $(PROGRAMS)

hebrus: ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hebrus-server: ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hebrus-bench: ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hebrus-eval: ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

hebrus-agent: ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4: hebrus
ds4-server: hebrus-server
ds4-bench: hebrus-bench
ds4-eval: hebrus-eval
ds4-agent: hebrus-agent

$(DS4_PROGRAMS):
	@rm -f "$@"
	@ln -s "$(notdir $<)" "$@"

ds4_build_cpu.o: ds4_build.c ds4.h ds4_expert_store.h FORCE
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_build.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_profile.o: ds4_profile.c ds4_profile.h
	$(CC) $(CFLAGS) -c -o $@ ds4_profile.c

ds4_qwen.o: ds4_qwen.c ds4_qwen.h
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -c -o $@ ds4_qwen.c

ds4_qwen_unicode.o: ds4_qwen_unicode.c ds4_qwen_unicode.h \
		ds4_qwen_unicode_data.inc
	$(CC) $(CFLAGS) -c -o $@ ds4_qwen_unicode.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_help.o: ds4_help.c ds4_help.h hebrus_identity.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c tests/internal/ds4_agent_unit.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ tests/ds4_agent_test.c

ds4_agent_test_impl.o: ds4_agent.c ds4.h ds4_ssd.h \
		ds4_help.h hebrus_identity.h ds4_kvstore.h ds4_web.h linenoise.h \
		tests/internal/ds4_agent_unit.h tests/internal/ds4_agent_unit.inc
	$(CC) $(CFLAGS) -DDS4_NO_GPU -DDS4_AGENT_TEST \
		-DDS4_AGENT_TEST_NO_MAIN -Wno-unused-function -c -o $@ ds4_agent.c

tests/test_ssd_residency: tests/test_ssd_residency.c ds4_ssd.o
	$(CC) $(CFLAGS) -I. -o $@ $^ -lm -pthread

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_profile.h ds4_gpu.h ds4_qwen.h \
		ds4_expert_store.h ds4_qwen_unicode.h ds4_streaming_hotlist.inc
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_test_core.o: ds4.c ds4.h ds4_ssd.h ds4_profile.h \
		ds4_gpu.h ds4_qwen.h ds4_expert_store.h ds4_qwen_unicode.h \
		ds4_streaming_hotlist.inc tests/internal/ds4_qwen_cpu_test_hooks.h
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU -DDS4_TEST_HOOKS \
		-Wno-unused-function -Wno-unused-parameter -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_help.h hebrus_identity.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_agent_test: ds4_agent_test.o ds4_agent_test_impl.o ds4_help.o \
		ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

model-free-test: $(PROGRAMS) ds4_test ds4_agent_test q4k-dot-test \
		tests/test_q4k_top8 \
		tests/test_qwen_session \
		tests/test_qwen_tokenizer \
		tests/test_qwen_gdn_ref tests/test_qwen_attention_ref \
		tests/test_qwen_state tests/test_qwen_unicode \
		tests/test_qwen_expert_group \
		tests/test_expert_store \
		tests/test_metal_ssd_profile \
		tests/test_ssd_residency download-model-test visible-identity-test \
		tests/test_capabilities.py \
		tests/test_command_aliases.py
	python3 tests/test_capabilities.py --bin-dir . --backend cpu
	python3 tests/test_command_aliases.py --bin-dir . --backend cpu --layout profile
	sh tests/test_retired_distributed_flags.sh
	sh tests/test_benchmark_env_guard.sh
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test --server
	./tests/test_q4k_top8
	./tests/test_qwen_session
	./tests/test_qwen_tokenizer
	./tests/test_qwen_gdn_ref
	./tests/test_qwen_attention_ref
	./tests/test_qwen_state
	./tests/test_qwen_unicode
	./tests/test_qwen_expert_group
	DS4_EXPERT_STORE_PROBE=./tests/test_expert_store \
		python3 tests/test_expert_major.py
	./tests/test_metal_ssd_profile
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	./tests/test_ssd_residency
	python3 tests/test_qwen_metadata.py ./ds4

capabilities-test: $(PROGRAMS) tests/test_capabilities.py
	python3 tests/test_capabilities.py --bin-dir . --backend cpu

command-alias-test: $(PROGRAMS) tests/test_command_aliases.py
	python3 tests/test_command_aliases.py --bin-dir . --backend cpu --layout profile

tests/test_visible_identity: tests/test_visible_identity.c hebrus_identity.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -I. -o $@ $<

visible-identity-test: tests/test_visible_identity
	./tests/test_visible_identity

test: model-free-test
	./ds4_test

context-audit:
	python3 tools/context_audit.py

doc-links:
	python3 tools/check_doc_links.py

imatrix-dataset-check:
	python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py --check

prompt-fixture-check:
	python3 speed-bench/build_long_context_prompt.py --check

premerge: context-audit doc-links brand-boundary-audit brand-boundary-test \
	release-contract release-contract-test \
	imatrix-dataset-check prompt-fixture-check model-free-test
	$(MAKE) install-test
	git diff --check

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

test_q4k_top8.o: tests/test_q4k_top8.c \
		tests/internal/ds4_qwen_cpu_test_hooks.h ds4_qwen.h
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU -DDS4_TEST_HOOKS \
		-I. -c -o $@ $<

tests/test_q4k_top8: test_q4k_top8.o ds4_test_core.o ds4_build_cpu.o \
		ds4_ssd.o ds4_profile.o ds4_expert_store.o \
		ds4_qwen.o ds4_qwen_unicode.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_qwen_session: tests/test_qwen_session.c ds4.c ds4.h ds4_ssd.h ds4_profile.h \
		ds4_gpu.h ds4_qwen.h ds4_qwen_unicode.h \
		ds4_build.c ds4_expert_store.o ds4_ssd.c \
		ds4_profile.c ds4_qwen.c \
		ds4_qwen_unicode.c ds4_qwen_unicode_data.inc \
		ds4_streaming_hotlist.inc runtime/ds4_glm_graph.inc \
		runtime/ds4_deepseek_cache_phase.inc
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -o $@ \
		$(filter-out ds4.c,$(filter %.c %.o,$^)) $(LDLIBS)

tests/test_qwen_tokenizer: tests/test_qwen_tokenizer.c ds4.c ds4.h \
		ds4_kvstore.c ds4_kvstore.h ds4_ssd.h ds4_profile.c ds4_profile.h ds4_gpu.h ds4_qwen.h \
		ds4_qwen_unicode.h ds4_build.c ds4_expert_store.o ds4_ssd.c \
		ds4_qwen.c ds4_qwen_unicode.c ds4_qwen_unicode_data.inc \
		ds4_streaming_hotlist.inc tests/qwen/qwen36_tokenizer_fixture.inc
	$(CC) $(CFLAGS) $(QWEN_CFLAGS) -DDS4_NO_GPU \
		-Wno-unused-function -Wno-unused-parameter -I. -o $@ \
		$(filter-out ds4.c,$(filter %.c %.o,$^)) $(LDLIBS)

qwen-metadata-test: ds4 tests/test_qwen_metadata.py
	python3 tests/test_qwen_metadata.py ./ds4

tests/test_qwen_gdn_ref: tests/test_qwen_gdn_ref.c ds4_qwen_ref.c ds4_qwen.c \
		ds4_qwen_ref.h ds4_qwen.h tests/qwen/qwen36_gdn_golden.inc
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_gdn_ref.c ds4_qwen_ref.c ds4_qwen.c -lm

tests/test_qwen_attention_ref: tests/test_qwen_attention_ref.c ds4_qwen_ref.c \
		ds4_qwen.c ds4_qwen_ref.h ds4_qwen.h \
		tests/qwen/qwen36_attention_golden.inc
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_attention_ref.c ds4_qwen_ref.c ds4_qwen.c -lm

tests/test_qwen_state: tests/test_qwen_state.c ds4_qwen.c ds4_qwen.h
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_state.c ds4_qwen.c -lm

tests/test_qwen_unicode: tests/test_qwen_unicode.c ds4_qwen_unicode.c \
		ds4_qwen_unicode.h ds4_qwen_unicode_data.inc
	$(CC) -O2 -Wall -Wextra -std=c99 -I. -o $@ \
		tests/test_qwen_unicode.c ds4_qwen_unicode.c

qwen-reference-test: tests/test_qwen_gdn_ref tests/test_qwen_attention_ref \
		tests/test_qwen_unicode
	python3 tests/qwen/collect_gdn_reference.py --check
	python3 tests/qwen/collect_attention_reference.py --check
	python3 tests/qwen/test_v_tiling_contract.py
	python3 tests/gen_qwen_unicode.py --check
	./tests/test_qwen_gdn_ref
	./tests/test_qwen_attention_ref
	./tests/test_qwen_unicode

qwen-unicode-test: tests/test_qwen_unicode
	python3 tests/gen_qwen_unicode.py --check
	./tests/test_qwen_unicode

qwen-tokenizer-test: tests/test_qwen_tokenizer
	./tests/test_qwen_tokenizer

tests/test_qwen_expert_group: tests/test_qwen_expert_group.c \
		ds4_qwen_expert_group.c ds4_qwen_expert_group.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_qwen_expert_group.c \
		ds4_qwen_expert_group.c $(LDLIBS)

qwen-expert-group-test: tests/test_qwen_expert_group
	./tests/test_qwen_expert_group

tests/test_expert_store: tests/test_expert_store.c ds4_expert_store.c \
		ds4_expert_store.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_expert_store.c \
		ds4_expert_store.c $(LDLIBS)

expert-store-test: tests/test_expert_store
	DS4_EXPERT_STORE_PROBE=./tests/test_expert_store \
		python3 tests/test_expert_major.py

tests/test_metal_ssd_profile: tests/test_metal_ssd_profile.c \
		ds4_profile.c ds4_profile.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_metal_ssd_profile.c \
		ds4_profile.c $(LDLIBS)

metal-ssd-profile-test: tests/test_metal_ssd_profile
	./tests/test_metal_ssd_profile

endif

install: $(INSTALL_SOURCE_PROGRAMS)
	@set -eu; \
		case "$(BINDIR)" in \
			/*) ;; \
			*) echo "install: BINDIR must be absolute: $(BINDIR)" >&2; exit 2 ;; \
		esac; \
		dest="$(INSTALL_DEST_BINDIR)"; \
		mkdir -p "$$dest"; \
		for name in $(PROGRAMS); do \
			path="$$dest/$$name"; \
			if [ -d "$$path" ] && [ ! -L "$$path" ]; then \
				echo "install: refusing to replace directory $$path" >&2; \
				exit 2; \
			fi; \
		done; \
		tmp=; \
		trap 'test -z "$$tmp" || rm -f "$$tmp"' 0 1 2 3 15; \
		for name in $(HEBRUS_PROGRAMS); do \
			tmp="$$dest/.$$name.install.$$$$"; \
			rm -f "$$tmp"; \
			$(INSTALL) -m "$(INSTALL_MODE)" \
				"$(INSTALL_SOURCE_BINDIR)/$$name" "$$tmp"; \
			rm -f "$$dest/$$name"; \
			mv "$$tmp" "$$dest/$$name"; \
			tmp=; \
		done; \
		for canonical in $(HEBRUS_PROGRAMS); do \
			legacy=ds4$${canonical#hebrus}; \
			tmp="$$dest/.$$legacy.install.$$$$"; \
			rm -f "$$tmp"; \
			ln -s "$$canonical" "$$tmp"; \
			rm -f "$$dest/$$legacy"; \
			mv "$$tmp" "$$dest/$$legacy"; \
			tmp=; \
		done

uninstall:
	@set -eu; \
		case "$(BINDIR)" in \
			/*) ;; \
			*) echo "uninstall: BINDIR must be absolute: $(BINDIR)" >&2; exit 2 ;; \
		esac; \
		dest="$(INSTALL_DEST_BINDIR)"; \
		for name in $(PROGRAMS); do \
			path="$$dest/$$name"; \
			if [ -d "$$path" ] && [ ! -L "$$path" ]; then \
				echo "uninstall: refusing to remove directory $$path" >&2; \
				exit 2; \
			fi; \
		done; \
		for name in $(PROGRAMS); do \
			rm -f "$$dest/$$name"; \
		done

install-test: $(INSTALL_SOURCE_PROGRAMS) tests/test_install.sh \
		tests/test_capabilities.py tests/test_command_aliases.py
	HEBRUS_INSTALL_BACKEND="$(INSTALL_BACKEND)" MAKE="$(MAKE)" \
		sh tests/test_install.sh

clean:
	rm -rf "$(BUILD_ROOT)"
	rm -f hebrus hebrus-server hebrus-bench hebrus-eval hebrus-agent \
		ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native \
		ds4_server_test ds4_test ds4_agent_test tests/test_q4k_dot \
		tests/test_q4k_top8 \
		tests/test_qwen_session \
		tests/test_qwen_tokenizer \
		tests/test_qwen_gdn_ref tests/test_qwen_attention_ref \
		tests/test_qwen_state tests/test_qwen_unicode \
		tests/test_qwen_expert_group \
		tests/test_expert_store \
		tests/test_metal_ssd_profile \
		tests/test_ssd_residency \
		tests/test_visible_identity *.o
