#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build
if make -j$(nproc) > /root/build_abfix.log 2>&1; then
  echo "$(date) build ok, starting run" >> /root/abfix_chain.log
  MOAI_LAYERS=3 MOAI_DUMP=1 OMP_NUM_THREADS=4 ./test layers-sham > /root/abfix_3layer_sham.log 2>&1
  rc=$?
  if [ $rc -eq 0 ] && grep -q "final_L2_sham" /root/abfix_3layer_sham.log && ! grep -qi "illegal memory access\|EXPLOSION" /root/abfix_3layer_sham.log; then
    echo "$(date) run clean (L0-L2 pass), shutdown in 3min" >> /root/abfix_chain.log
    shutdown -h +3
  else
    echo "$(date) run ended rc=$rc (explosion/error), staying up for forensics" >> /root/abfix_chain.log
  fi
else
  echo "$(date) BUILD FAILED" >> /root/abfix_chain.log
fi
