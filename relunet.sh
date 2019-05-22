p="titanx-short"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/Baboon/Baboon_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/F16/F16_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/House/House_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/kodim01/kodim01_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/kodim02/kodim02_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/kodim03/kodim03_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/kodim12/kodim12_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/Lena/Lena_s25.png"

sbatch -p $p --gres=gpu:1 --wrap "python code/dip_relunet.py --cfg configs/dip_relunet/relunet.yaml --noisy_img data/denoise_small/Peppers/Peppers_s25.png"


