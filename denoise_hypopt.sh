
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/Baboon/  --clean_img data/denoise/Baboon/Baboon.png
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/F16/  --clean_img data/denoise/F16/F16.png
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/House/  --clean_img data/denoise/House/House.png
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/kodim01/  --clean_img data/denoise/kodim01/kodim01.png
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/kodim02/  --clean_img data/denoise/kodim02/kodim02.png
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/kodim03/  --clean_img data/denoise/kodim03/kodim03.png
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/kodim12/  --clean_img data/denoise/kodim12/kodim12.png

srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/Lena/  --clean_img data/denoise/Lena/Lena.png
srun --mem 100000 -p titanx-short python code/hypopt.py --output_dir Outputs/denoise/Peppers/  --clean_img data/denoise/Peppers/Peppers.png



