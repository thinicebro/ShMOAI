#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build
if make -j$(nproc) > /root/build_softmaxfix.log 2>&1; then
  echo "$(date) build ok, starting 12-layer run (SOFTMAX-FIX)" >> /root/softmaxfix_chain.log
  OMP_NUM_THREADS=4 MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=12 MOAI_DUMP=1 ./test layers-sham > /root/softmaxfix_12layer_sham.log 2>&1
  rc=$?
  echo "$(date) run ended rc=$rc" >> /root/softmaxfix_chain.log
  shutdown -h +3
else
  echo "$(date) BUILD FAILED" >> /root/softmaxfix_chain.log
fi
