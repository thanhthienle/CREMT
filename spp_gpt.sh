#!/bin/bash -e

#SBATCH --job-name=gpt # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/thienlt3/asa/mtl/nash_ntask_slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/thienlt3/asa/mtl/nash_ntask_slurm_%A.err # create a error file
#SBATCH --partition=applied # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.thienlt3@vinai.io

eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/vinai/users/thienlt3/envs/cremt

python3 run_continual.py \
    --logname nash \
    --dataname TACRED \
    --mtl nashmtl \
    --encoder_epochs 50 --encoder_lr 2e-5 \
    --prompt_pool_epochs 20  --prompt_pool_lr 2e-4 \
    --classifier_epochs 500 --classifier_lr 2e-5 \
    --replay_epochs 100 \
    --total_rounds 1 \
    --gpu 0
