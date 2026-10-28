#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) starting 128-token 12-layer sham run" >> /root/sham128_chain.log
MOAI_REAL_WEIGHTS=1 MOAI_TOKENS=128 MOAI_LAYERS=12 MOAI_DUMP=1 OMP_NUM_THREADS=4 ./test layers-sham > /root/sham128.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q 'final_L11_sham' /root/sham128.log && ! grep -qi 'illegal memory access\|EXPLOSION' /root/sham128.log; then
  echo "$(date) 128tok 12 layers clean, shutdown in 3min" >> /root/sham128_chain.log
  shutdown -h +3
else
  echo "$(date) run ended rc=$rc, staying up" >> /root/sham128_chain.log
fi
