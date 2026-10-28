#!/bin/bash
cd /root/autodl-tmp/MOAI_GPU_second/build128
echo "$(date) rebuild start" >> /root/orig128_chain.log
make -j 16 > /root/build128_rebuild.log 2>&1
if [ $? -ne 0 ]; then
  echo "$(date) rebuild FAILED, shutting down" >> /root/orig128_chain.log
  shutdown -h +3
  exit 1
fi
echo "$(date) rebuild ok" >> /root/orig128_chain.log
nohup /root/watch_128o.sh > /dev/null 2>&1 &
exec /root/run_128orig.sh
