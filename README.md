# ShMOAI: MOAI Accelerated by Shared-a Batched Bootstrapping

This repository contains the experiment code accompanying the paper

> **ShMOAI: Accelerating Secure Transformer Inference via Shared-a
> Batched Bootstrapping**

ShMOAI brings the shared-a ciphertext format and S2C-first batched bootstrapping
into the GPU inference pipeline of [MOAI](https://eprint.iacr.org/2025/991)
(ICLR 2026), and evaluates the result end-to-end on the full 12-layer BERT-base
model (hidden width 768, 12 attention heads, intermediate width 3072) under CKKS
fully homomorphic encryption.

The code is built on top of [PhantomFHE](https://github.com/encryptorion-lab/phantom-fhe)
(GPU CKKS library) and the MOAI_GPU implementation. All secret keys are generated
in memory at runtime; no key material is persisted.

## 1. Environment

The paper's experiments were run on the following platform:

| Item        | Value                                             |
|-------------|---------------------------------------------------|
| GPU         | NVIDIA RTX PRO 6000 Blackwell (96 GB), sm_120     |
| CUDA        | 12.8                                              |
| Compiler    | gcc/g++ 11.4.0, C++17 / CUDA C++17                |
| CPU parallel| OpenMP 4.5 (`OMP_NUM_THREADS=4`)                  |
| Libraries   | NTL 11.5.1, GMP                                   |
| CMake       | >= 3.20 (3.31.8 used)                             |

Notes:

- The full-width (768) single-layer and the 12-layer experiments need a GPU with
  **>= 96 GB** memory (peak usage ~80 GB for the original pipeline, ~88 GB for
  the shared-a pipeline). Quarter-width (192) variants fit in 16 GB.
- The build targets sm_120 (Blackwell). For A100/H200, change
  `CMAKE_CUDA_ARCHITECTURES` to `80;90` in `CMakeLists.txt`.

## 2. Build

```bash
cmake -S . -B build      # configures and copies src/data/ into build/
cd build
make -j
```

This produces a single executable `build/test`. All modes must be run **from the
`build/` directory**, because the binary reads model data with relative paths
(e.g. `att_block_weights/query_weight.txt`).

## 3. Repository layout

```
src/test.cu                 entry point: mode dispatch on argv[1]
src/include/                all implementation modules (header-heavy, namespace moai)
  source/bootstrapping/     CKKS bootstrapping; SharedACiphertext, FmtSwitch,
                            S2C-first batched pipelines
  source/matrix_mul/        batched encoding, Ct-Pt / Ct-Ct matrix multiplication
  source/non_linear_func/   GELU (Chebyshev basis), LayerNorm, Softmax, ...
src/data/                   BERT-base weights and plaintext reference activations
  layer_0 ... layer_11/     per-layer weights (parms/) and plaintext references
                            (allresults/) for each of the 13 sublayers
  att_block_weights/        attention / FFN weights and inputs for the
  feed_forward_weights/     single-layer test (full width; sliced on read for
  self_output_weights/      the quarter-width variant)
thirdparty/phantom-fhe/     PhantomFHE, compiled directly into the test binary
run_scripts/                the automation scripts used on the experiment server
results.csv                 appended micro-benchmark timings
single_layer_results.csv    appended per-stage single-layer timings
```

The model weights under `src/data/` are derived from the public
[BERT-base](https://github.com/google-research/bert) model (uncased), which is
distributed under the Apache License 2.0; the plaintext reference activations
were produced by running that model locally. Both are redistributed here solely
for reproducibility.

## 4. Experiments and how to reproduce them

All experiments below follow MOAI's timing convention: **reported times are
totals over 256 inputs of up to 128 tokens each**; divide by 256 to obtain the
amortized per-input time.

### 4.1 Standalone batched bootstrapping (Table 3 of the paper)

768 ciphertexts, bootstrapped with the per-ciphertext pipeline vs. the shared-a
S2C-first batched pipeline with group sizes n = 4 and n = 8:

```bash
cd build
OMP_NUM_THREADS=4 ./test s2c-both4x768    # per-ciphertext vs shared-a, n=4
OMP_NUM_THREADS=4 ./test s2c-both8x768    # per-ciphertext vs shared-a, n=8
```

Mode syntax: `s2c-{ordinary|shared|both|sham}[N[xM]]` — group size `N`,
total ciphertexts `M`. `-sham` additionally caches per-chunk key material
(amortized key generation). Expected output (per run): total time and maximum
decryption error for each pipeline; the amortized time is total/768.
Reference numbers (RTX PRO 6000): 334.1 s (per-ciphertext), 209.3 s (n=4),
193.3 s (n=8), decryption error ~1.33e-5 for all three.

### 4.2 Single-layer end-to-end comparison (Tables 4 and 5)

One full-width (768) BERT layer, original per-ciphertext bootstrapping vs.
shared-a S2C-first batched bootstrapping:

```bash
OMP_NUM_THREADS=4 ./test layer-orig       # original MOAI pipeline
OMP_NUM_THREADS=4 ./test layer-sham       # ShMOAI pipeline (amortized keys)
```

Per-stage timings (Attention, SelfOutput, 1st-4th Bootstrapping, LayerNorm1/2,
FFN) are printed and appended to `../single_layer_results.csv`; correctness is
checked by decrypting after each stage and comparing against the plaintext
references under `src/data/`. The paper reports the average of two independent
runs. Reference numbers: total 2647.1 s vs. 1754.7 s, of which bootstrapping
1899.5 s vs. 1007.9 s (1.88x).

Quarter-width (192) variants for smaller GPUs: `layerq-orig` / `layerq-sham`.

### 4.3 Full 12-layer end-to-end comparison (Table 6)

The complete BERT-base model with per-layer real weights:

```bash
MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=12 MOAI_DUMP=1 OMP_NUM_THREADS=4 ./test layers-orig
MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=12 MOAI_DUMP=1 OMP_NUM_THREADS=4 ./test layers-sham
```

- `MOAI_REAL_WEIGHTS=1` reloads each layer's own weights from
  `src/data/layer_k/` (default reuses layer-0 weights).
- `MOAI_LAYERS=N` sets the number of layers (default 3).
- `MOAI_DUMP=1` exports decrypted ciphertexts after each of the 13 stages of
  every layer for the stage-by-stage correctness check (Tables 7 and 8). Unset it
  for clean timing runs.
- `MOAI_TOKENS=128` selects the 128-token input variant (default 5-token).

The per-layer time is printed after each layer; the paper reports the mean over
the 12 layers. Reference numbers: 3104.7 s vs. 2124.1 s per layer (1.46x);
single bootstrapping pass 558 s vs. 295 s (1.9x). Peak GPU memory (observed via
`nvidia-smi`): ~80 GB (original) vs. ~88 GB (shared-a).

### 4.4 Other modes

- `./test ordinary|shared|both[N[xM]]` — MOAI's original C2S-first pipeline
  with the shared-a format (used during development; the paper's pipeline is
  S2C-first).
- `./test sanity`, `sanity2`-`sanity6`, `s2c-probe`, `s2c-scan`, `s2c-shprobe`,
  `ln1-repro` — debug/probe modes used during development; not needed for
  reproducing the paper.
- Many unit tests (batch encode/encrypt, Ct-Pt and Ct-Ct matrix multiplication,
  GELU, LayerNorm, Softmax, bootstrapping, ...) exist as commented-out calls at
  the bottom of `src/test.cu`; uncomment the corresponding call to run them.

### 4.5 run_scripts/

The scripts under `run_scripts/` are the exact automation used on the rented
experiment server (AutoDL): each builds (if needed), launches one experiment
with the environment variables above, checks the log for completion markers
(e.g. `final_L11_orig`) and error patterns, and shuts the instance down to
avoid idle billing. They document precisely which command produced each log;
paths inside refer to the server layout (`/root/autodl-tmp/MOAI_GPU_second`).

## 5. Outputs

- Timings are appended to `results.csv` (micro-benchmarks) and
  `single_layer_results.csv` (per-stage single-layer timings) in the repository
  root (i.e. `../` relative to `build/`).
- Correctness reports (max/relative decryption error per stage) are printed to
  stdout; with `MOAI_DUMP=1` the decrypted values are also written to disk for
  offline comparison against `src/data/layer_k/allresults/`.

## 6. Citation

If you use this code, please cite our paper (citation information will be
updated upon publication) and the original MOAI paper:

```bibtex
@inproceedings{zhang2026moai,
  author    = {Zhang, Linru and Wang, Xiangning and Sim, Jun Jie and Huang,
               Zhicong and Zhong, Jiahao and Wang, Huaxiong and Duan, Pu and
               Lam, Kwok-Yan},
  title     = {{MOAI}: Module-Optimizing Architecture for Non-Interactive
               Secure Transformer Inference},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
