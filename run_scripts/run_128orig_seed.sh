#!/bin/bash
# 128-token orig, seeded PRNG (MOAI_PRNG_SEED=42) for reproducibility,
# full stage dump + softmax internal probes, plain (untiled) input.
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) starting 128-token orig SEEDED run (seed=42, DUMP+SM_DBG on)" >> /root/orig128s_chain.log
MOAI_PRNG_SEED=42 MOAI_REAL_WEIGHTS=1 MOAI_TOKENS=128 MOAI_LAYERS=12 MOAI_DUMP=1 MOAI_SM_DBG=1 OMP_NUM_THREADS=4 timeout 60000 ./test layers-orig > /root/orig128s.log 2>&1
rc=$?
echo "$(date) orig seeded run ended rc=$rc" >> /root/orig128s_chain.log
shutdown -h +3
