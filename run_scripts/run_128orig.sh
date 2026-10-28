#!/bin/bash
# 128-token 12-layer orig run. v2: shut down on ANY outcome (success OR
# failure) -- v1 stayed up on failure for debugging and burned ~14.5h idle.
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) starting 128-token 12-layer orig run (v2)" >> /root/orig128_chain.log
MOAI_REAL_WEIGHTS=1 MOAI_TOKENS=128 MOAI_LAYERS=12 MOAI_DUMP=1 OMP_NUM_THREADS=4 ./test layers-orig > /root/orig128.log 2>&1
rc=$?
if [ $rc -eq 0 ] && grep -q 'final_L11_orig' /root/orig128.log && ! grep -qi 'illegal memory access\|EXPLOSION' /root/orig128.log; then
  echo "$(date) 128tok orig 12 layers clean, shutdown in 3min" >> /root/orig128_chain.log
else
  echo "$(date) orig run ended rc=$rc, shutting down anyway" >> /root/orig128_chain.log
fi
shutdown -h +3
