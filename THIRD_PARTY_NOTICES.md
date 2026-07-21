# Third-party notices

This file collects notices for third-party code bundled in or materially
adapted by this repository. Source headers remain authoritative and must not be
removed. The project-level [`LICENSE`](LICENSE) does not replace the terms
below.

## linenoise

Files: `linenoise.c`, `linenoise.h`

Upstream: <https://github.com/antirez/linenoise>

```text
Copyright (c) 2010-2023, Salvatore Sanfilippo <antirez at gmail dot com>
Copyright (c) 2010-2013, Pieter Noordhuis <pcnoordhuis at gmail dot com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Rax

Bundled copyright lines:

`rax.c`:

```text
Copyright (c) 2017-2026, Salvatore Sanfilippo <antirez at gmail dot com>
```

`rax.h`:

```text
Copyright (c) 2017-2018, Salvatore Sanfilippo <antirez at gmail dot com>
```

`rax_malloc.h`:

```text
Copyright (c) 2017, Salvatore Sanfilippo <antirez at gmail dot com>
```

All three files carry the following terms:

```text
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of Redis nor the names of its contributors may be used
    to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## GGML and llama.cpp-derived implementation

The root MIT license retains:

```text
Copyright (c) 2023-2026 The ggml authors
```

The repository retains or adapts narrow pieces of MIT-licensed GGML/llama.cpp
work, including quant layouts and tables, CPU quantization and dot-product
logic, selected kernel techniques, and GGUF/converter behavior. See
[`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md), [`LICENSE`](LICENSE), and the
source-file comments for scope.

## YaRN-derived RoPE algorithm

File: `metal/dsv4_rope.metal`

The source contains the following notice:

```text
YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
```

## Preservation

When redistributing source or binaries, preserve the notices and conditions
required by the relevant source headers. When copying or moving one of these
files, carry its attribution and license text with it. No project or contributor
listed here endorses Hebrus merely because their work is credited.
