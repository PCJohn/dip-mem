
if [ "$1" = "run" ]; then
# natural images
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle/triangle.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/barbara/barbara.png  --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/peppers256/peppers256.png  --task traj

# controlled amount of high freq.
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-020/triangle-020.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-030/triangle-030.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-040/triangle-040.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-050/triangle-050.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-060/triangle-060.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-070/triangle-070.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-080/triangle-080.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-090/triangle-090.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/triangle-100/triangle-100.png --task traj

# failure cases: low freq. noise
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/baboon-low_freq_noise/baboon-low_freq_noise.png --task traj
srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/barbara-low_freq_noise/barbara-low_freq_noise.png --task traj


#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/boat/boat.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/Cameraman256/Cameraman256.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/couple/couple.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/fingerprint/fingerprint.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/hill/hill.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/house/house.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/Lena512/Lena512.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/man/man.png  --task traj
#srun --mem 30000 -p titanx-short python code/clean_analyze.py --img data/clean/montage/montage.png  --task traj
fi
