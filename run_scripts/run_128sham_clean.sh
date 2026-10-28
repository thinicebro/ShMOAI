#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) starting 128-token sham CLEAN-TIMING run (MOAI_DUMP off)" >> /root/sham128t_chain.log
MOAI_REAL_WEIGHTS=1 MOAI_TOKENS=128 MOAI_LAYERS=12 OMP_NUM_THREADS=4 timeout 60000 ./test layers-sham > /root/sham128t.log 2>&1
rc=$?
echo "$(date) run ended rc=$rc, shutdown in 3min" >> /root/sham128t_chain.log
shutdown -h +3
