#!/bin/sh
#PBS -lwalltime=12:00:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
module load NCCL

export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

for m in LSTM RNN
do
    for seq in 18 19 20 22 24 26 28 30 35 40
    do
        echo
        echo ${m} ${seq}
        python /home/lgpu0080/part1/train.py --model_type ${m} --input_length ${seq}
    done
done