c=$1
p="1080ti-short"

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/Baboon/Baboon.png  --noisy_img data/denoise/Baboon/Baboon_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/Baboon/Baboon.png  --noisy_img data/denoise/Baboon/Baboon_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/F16/F16.png  --noisy_img data/denoise/F16/F16_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/House/House.png  --noisy_img data/denoise/House/House_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/kodim01/kodim01.png  --noisy_img data/denoise/kodim01/kodim01_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/kodim02/kodim02.png  --noisy_img data/denoise/kodim02/kodim02_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/kodim03/kodim03.png  --noisy_img data/denoise/kodim03/kodim03_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/kodim12/kodim12.png  --noisy_img data/denoise/kodim12/kodim12_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/Lena/Lena.png  --noisy_img data/denoise/Lena/Lena_s25.png  --task denoise --cfg $c

srun --mem 30000 -p $p python code/traj_exp.py --clean_img data/denoise/Peppers/Peppers.png  --noisy_img data/denoise/Peppers/Peppers_s25.png  --task denoise --cfg $c




