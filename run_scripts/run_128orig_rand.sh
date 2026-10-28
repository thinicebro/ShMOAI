#!/bin/bash
# 128-token orig, RANDOM key (no MOAI_PRNG_SEED) -- reproducibility test of the
# seeded clean run: does a fresh random key also pass 12 layers now?
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) starting 128-token orig RANDOM-KEY run (DUMP+SM_DBG on)" >> /root/orig128r_chain.log
MOAI_REAL_WEIGHTS=1 MOAI_TOKENS=128 MOAI_LAYERS=12 MOAI_DUMP=1 MOAI_SM_DBG=1 OMP_NUM_THREADS=4 timeout 60000 ./test layers-orig > /root/orig128r.log 2>&1
rc=$?
echo "$(date) orig random-key run ended rc=$rc" >> /root/orig128r_chain.log
shutdown -h +3
