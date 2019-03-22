#!/usr/bin/env bash
#SBATCH --job-name=denoise
#SBATCH -o gypsum/logs/%j_denoise.txt 
#SBATCH -e gypsum/errs/%j_denoise.txt
#SBATCH -p 1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=100000


python code/dip.py \
    --img_file $1 \
    --output_file $2 \
    --lr $3 \
    --niter $4 \
    --traj_iter $5 \


