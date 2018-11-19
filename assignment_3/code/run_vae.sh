#!/bin/sh
#PBS -lwalltime=00:30:00
#PBS -qgpu
#PBS -lnodes=1:ppn=12
#PBS -lmem=250GB

module load eb
module load Python/3.6.3-foss-2017b
#module load cuDNN/7.0.5-CUDA-9.0.176
module load cuDNN/7.3.1-CUDA-10.0.130
module load OpenMPI/2.1.1-GCC-6.4.0-2.28
module load NCCL

#export LD_LIBRARY_PATH=/hpc/sw/NCCL/2.0.5/lib:/hpc/eb/Debian9/cuDNN/7.0.5-CUDA9.0.176/lib64:/hpc/eb/Debian9/CUDA/9.0.176/lib64:$LD_LIBRARY_PATH

python a3/a3_vae.py --epochs "5" --zdim "20"
#python a3/a3_gan.py