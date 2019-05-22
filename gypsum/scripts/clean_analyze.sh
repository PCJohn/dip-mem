#!/usr/bin/env bash
#SBATCH --job-name=denoise
#SBATCH -o gypsum/logs/%j_denoise.txt 
#SBATCH -e gypsum/errs/%j_denoise.txt
#SBATCH -p titanx-short
#SBATCH --gres=gpu:1
#SBATCH --mem=100000



python code/dip.py \
    --img_file $1 \
    --output_file $2 \
    --lr $3 \
    --niter $4 \
    --traj_iter $5 \
    --net_struct $6 \
    --fixed_start $7 \
    --langevin $8 \
    --reg_noise_std $9 \
    --n_ch_down 128 \
    --n_ch_up 128 \
    --skip_conn 4 \
    --depth 5 \

