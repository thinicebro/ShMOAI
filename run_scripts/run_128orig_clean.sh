#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) starting 128-token orig CLEAN-TIMING run (MOAI_DUMP off)" >> /root/orig128t_chain.log
MOAI_REAL_WEIGHTS=1 MOAI_TOKENS=128 MOAI_LAYERS=12 OMP_NUM_THREADS=4 timeout 60000 ./test layers-orig > /root/orig128t.log 2>&1
rc=$?
echo "$(date) orig clean run ended rc=$rc, shutdown in 3min" >> /root/orig128t_chain.log
shutdown -h +3
