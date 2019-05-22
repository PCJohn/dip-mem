p="titanx-short"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/Baboon/  --clean_img data/denoise_small/Baboon/Baboon.png --noisy_img data/denoise_small/Baboon/Baboon_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/F16/  --clean_img data/denoise_small/F16/F16.png --noisy_img data/denoise_small/F16/F16_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/House/  --clean_img data/denoise_small/House/House.png --noisy_img data/denoise_small/House/House_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim01/  --clean_img data/denoise_small/kodim01/kodim01.png --noisy_img data/denoise_small/kodim01/kodim01_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim02/  --clean_img data/denoise_small/kodim02/kodim02.png --noisy_img data/denoise_small/kodim02/kodim02_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim03/  --clean_img data/denoise_small/kodim03/kodim03.png --noisy_img data/denoise_small/kodim03/kodim03_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim12/  --clean_img data/denoise_small/kodim12/kodim12.png --noisy_img data/denoise_small/kodim12/kodim12_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/Lena/  --clean_img data/denoise_small/Lena/Lena.png --noisy_img data/denoise_small/Lena/Lena_s25.png --cfg configs/denoise/denoise_default.yaml"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/Peppers/  --clean_img data/denoise_small/Peppers/Peppers.png --noisy_img data/denoise_small/Peppers/Peppers_s25.png --cfg configs/denoise/denoise_default.yaml"



