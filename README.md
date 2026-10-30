# ShMOAI: MOAI Accelerated by Shared-a Batched Bootstrapping

Experiment code accompanying the paper *"ShMOAI: Accelerating Secure
Transformer Inference via Shared-a Batched Bootstrapping"*.

ShMOAI integrates the shared-a ciphertext format and S2C-first batched
bootstrapping into the GPU pipeline of [MOAI](https://eprint.iacr.org/2025/991),
built on top of [PhantomFHE](https://github.com/encryptorion-lab/phantom-fhe).

## Environment

- NVIDIA RTX PRO 6000 Blackwell (96 GB), sm_120; CUDA 12.8; gcc/g++ 11.4.0
- NTL 11.5.1 GMP, OpenMP 4.5, CMake >= 3.20
- Full-width (768) experiments need >= 96 GB GPU memory; quarter-width (192)
  variants fit in 16 GB
- For A100/H200, set `CMAKE_CUDA_ARCHITECTURES` to `80;90` in `CMakeLists.txt`

## Build

```bash
cmake -S . -B build && cd build && make -j

```

Run all commands from `build/` (model data is read via relative paths).

## Reproducing the paper

Timing convention (from MOAI): reported times are totals over 256 inputs of
up to 128 tokens; divide by 256 for the amortized per-input time.

- **Table 3** (standalone batched bootstrapping, 768 ciphertexts):
  `OMP_NUM_THREADS=4 ./test s2c-both4x768` (n=4), `./test s2c-both8x768` (n=8).
  Reference: 334.1 s / 209.3 s / 193.3 s, decryption error ~1.33e-5.
- **Tables 4-5** (single layer, full width):
  `OMP_NUM_THREADS=4 ./test layer-orig` and `./test layer-sham`.
  Reference: 2647.1 vs 1754.7 s total; bootstrapping 1899.5 s vs. 1007.9 s.
- **Tables 6-8** (full 12-layer model):
  `MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=12 MOAI_TOKENS=128 MOAI_DUMP=1 OMP_NUM_THREADS=4 ./test layers-orig`
  (and `layers-sham`). Reference: 3104.7 s vs. 2124.1 s per layer.
  Unset `MOAI_DUMP=1` for clean timing runs.

Smaller GPUs: `layerq-orig` / `layerq-sham` (quarter width).

## Citation

Please cite our paper (citation updated upon publication) and MOAI
(ePrint 2025/991).
