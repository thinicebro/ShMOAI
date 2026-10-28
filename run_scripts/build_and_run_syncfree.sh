#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build
while pgrep -x make >/dev/null || pgrep -x nvcc >/dev/null || pgrep -x ptxas >/dev/null || pgrep -x cicc >/dev/null; do sleep 30; done
if ! grep -q "Built target test" /root/build_syncfree.log; then
  echo "$(date) BUILD_FAILED" >> /root/syncfree_chain.log
  sleep 60; shutdown -h +1; exit 1
fi
echo "$(date) build ok, starting run" >> /root/syncfree_chain.log
rm -f final_L*_sham.txt stage_*_L*_sham.txt
mv /root/realw5_12layer_sham.log /root/realw5_12layer_sham_poolfix.log.bak 2>/dev/null
OMP_NUM_THREADS=4 MOAI_REAL_WEIGHTS=1 MOAI_LAYERS=3 MOAI_DUMP=1 ./test layers-sham > /root/realw5_12layer_sham.log 2>&1
echo "$(date) test ended, shutdown in 3min" >> /root/syncfree_chain.log
sleep 180
shutdown -h +1
