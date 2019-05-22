p="titanx-short"
c=$1

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/Baboon/  --clean_img data/denoise/Baboon/Baboon.png --noisy_img data/denoise/Baboon/Baboon_s25.png --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/F16/  --clean_img data/denoise/F16/F16.png --noisy_img data/denoise/F16/F16_s25.png --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/House/  --clean_img data/denoise/House/House.png --noisy_img data/denoise/House/House_s25.png --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim01/  --clean_img data/denoise/kodim01/kodim01.png --noisy_img data/denoise/kodim01/kodim01_s25.png --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim02/  --clean_img data/denoise/kodim02/kodim02.png  --noisy_img data/denoise/kodim02/kodim02_s25.png --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim03/  --clean_img data/denoise/kodim03/kodim03.png --noisy_img data/denoise/kodim03/kodim03_s25.png --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/kodim12/  --clean_img data/denoise/kodim12/kodim12.png --noisy_img data/denoise/kodim12/kodim12_s25.png  --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/Lena/  --clean_img data/denoise/Lena/Lena.png --noisy_img data/denoise/Lena/Lena_s25.png --cfg $c"

sbatch --mem 100000 -p $p --wrap "python code/hypopt.py --output_dir Outputs/denoise/Peppers/  --clean_img data/denoise/Peppers/Peppers.png --noisy_img data/denoise/Peppers/Peppers_s25.png --cfg $c"



