#!/usr/bin/env bash
#SBATCH --job-name=inpaint
#SBATCH -o gypsum/logs/%j_inpaint.txt 
#SBATCH -e gypsum/errs/%j_inpaint.txt
#SBATCH -p 180ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=100000



python code/dip.py \
    --img_file $1 \
    --output_file $2 \
    --lr $3 \
    --niter $4 \
    --traj_iter $5 \
    --net_depth $6 \
    --mask $7 \


