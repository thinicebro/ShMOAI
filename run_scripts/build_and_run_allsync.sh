#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build
if make -j$(nproc) > /root/build_allsync.log 2>&1; then
  echo "$(date) build ok, starting 12-layer run (ALL-SYNC ALLOC)" >> /root/allsync_chain.log
  OMP_NUM_THREADS=4 MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=12 MOAI_DUMP=1 ./test layers-sham > /root/allsync_12layer_sham.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q "final_L11_sham" /root/allsync_12layer_sham.log && ! grep -qi "illegal memory access\|EXPLOSION" /root/allsync_12layer_sham.log; then
    echo "$(date) 12 layers clean, shutdown in 3min" >> /root/allsync_chain.log
    shutdown -h +3
  else
    echo "$(date) run ended rc=$rc (explosion/error), staying up for forensics" >> /root/allsync_chain.log
  fi
else
  echo "$(date) BUILD FAILED" >> /root/allsync_chain.log
fi
