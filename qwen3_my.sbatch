#!/bin/bash
#SBATCH --job-name=qwen3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node 1
##SBATCH --output %x_%j.out
##SBATCH --error %x_%j.err
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

set -x

export FI_PROVIDER=efa

export OMP_NUM_THREADS=8
export PYTHONFAULTHANDLER=1

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
SLURM_GPUS_PER_NODE=8

srun -l \
    --mpi=pmix --cpu-bind=none \
        torchrun \
            --nnodes=$SLURM_JOB_NUM_NODES \
            --nproc_per_node=$SLURM_GPUS_PER_NODE \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$(hostname) \
            main.py --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 --impl my