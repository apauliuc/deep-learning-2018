#!/bin/sh
#PBS -lwalltime=00:30:00
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
module load NCCL

export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

#mkdir "$TMPDIR"/lgpu0080

python lstm/train.py --txt_file "/home/lgpu0080/lstm/rsc/book_EN_grimms_fairy_tails.txt" --summary_path "/home/lgpu0080/lstm_results/"

#cp -r "$TMPDIR"/lgpu0080 /home/lgpu0080/train_results/