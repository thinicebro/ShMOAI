#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build
if make -j$(nproc) > /root/build_maskv2.log 2>&1; then
  echo "$(date) build ok, starting 12-layer run (MASK-V2)" >> /root/maskv2_chain.log
  OMP_NUM_THREADS=4 MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=12 MOAI_DUMP=1 MOAI_SM_DBG=1 ./test layers-sham > /root/maskv2_12layer_sham.log 2>&1
  rc=$?
  echo "$(date) run ended rc=$rc" >> /root/maskv2_chain.log
  shutdown -h +3
else
  echo "$(date) BUILD FAILED" >> /root/maskv2_chain.log
fi
