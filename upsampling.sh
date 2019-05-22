P="titanx-short"

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/clean/zebra/zebra.png  --noisy_img data/clean/zebra/zebra.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/clean/barbara/barbara.png  --noisy_img data/clean/barbara/barbara.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/Baboon/Baboon.png  --noisy_img data/denoise/Baboon/Baboon.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/F16/F16.png  --noisy_img data/denoise/F16/F16.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/House/House.png  --noisy_img data/denoise/House/House.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/kodim01/kodim01.png  --noisy_img data/denoise/kodim01/kodim01.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/kodim02/kodim02.png  --noisy_img data/denoise/kodim02/kodim02.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/kodim03/kodim03.png  --noisy_img data/denoise/kodim03/kodim03.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/kodim12/kodim12.png  --noisy_img data/denoise/kodim12/kodim12.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/Lena/Lena.png  --noisy_img data/denoise/Lena/Lena.png  --task denoise --cfg configs/upsampling/vary_strides.yaml

srun --mem 30000 -p $P python code/traj_exp.py --clean_img data/denoise/Peppers/Peppers.png  --noisy_img data/denoise/Peppers/Peppers.png  --task denoise --cfg configs/upsampling/vary_strides.yaml





