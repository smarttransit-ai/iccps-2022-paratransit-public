#!/bin/bash
cd /home/cc
wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.3-linux-x86_64.tar.gz
tar zxvf julia-1.6.3-linux-x86_64.tar.gz
echo 'export PATH="$PATH:/home/cc/julia-1.6.3/bin"' >> .bashrc
echo "export JULIA_NUM_THREADS=30" >> .bashrc