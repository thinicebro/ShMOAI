#!/bin/bash
# 128-token orig, TILED input (all 256 sample lanes = real data, paper-faithful).
# Tests the padding-garbage hypothesis and yields orig 128 precision+timing.
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) starting 128-token orig TILED run (MOAI_DUMP on)" >> /root/orig128tile_chain.log
MOAI_TILE_INPUT=1 MOAI_REAL_WEIGHTS=1 MOAI_TOKENS=128 MOAI_LAYERS=12 MOAI_DUMP=1 OMP_NUM_THREADS=4 timeout 60000 ./test layers-orig > /root/orig128tile.log 2>&1
rc=$?
echo "$(date) orig tiled run ended rc=$rc, shutdown in 3min" >> /root/orig128tile_chain.log
shutdown -h +3
