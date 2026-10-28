#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build
echo "$(date) starting 12-layer run (REAL WEIGHTS)" >> /root/abfix12rw_chain.log
OMP_NUM_THREADS=4 MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=12 MOAI_DUMP=1 ./test layers-sham > /root/abfix12rw_12layer_sham.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q "final_L11_sham" /root/abfix12rw_12layer_sham.log && ! grep -qi "illegal memory access\|EXPLOSION" /root/abfix12rw_12layer_sham.log; then
  echo "$(date) 12 layers clean, shutdown in 3min" >> /root/abfix12rw_chain.log
  shutdown -h +3
else
  echo "$(date) run ended rc=$rc (explosion/error), staying up for forensics" >> /root/abfix12rw_chain.log
fi
