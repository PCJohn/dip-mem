# 1 component signals
#for i in 0 1 2 3 4 5
#do
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/freq-1-"$i"5_vary-channels.yaml"
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/freq-1-"$i"5_vary-depth.yaml"
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/freq-1-"$i"5_vary-filter.yaml"
#done
##########

# 2 component signals -- different initializations
# sample from normal -- near 1
#for i in 1 2 3 4 5 6 7 8 9
#do
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/normal/freq-2_normal-1"$i"e-1.yaml"
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/normal/freq-2_normal-"$i"e-1.yaml"
#done
# sample from normal -- macro
sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/normal/freq-2_normal-1e2.yaml"
sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/normal/freq-2_normal-1e1.yaml"
sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/normal/freq-2_normal-1.yaml"
for i in 1 2 3 4 5 6 7
do
    for j in 1 3 5 7
    do
        sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/normal/freq-2_normal-"$j"e-"$i".yaml"
        #sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/normal/freq-2_normal-5e-"$i".yaml"
    done
done

# xavier -- near 1
#for i in 1 2 3 4 5 6 7 8 9
#do
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/xavier/freq-2_xavier-1"$i"e-1.yaml"
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/xavier/freq-2_xavier-"$i"e-1.yaml"
#done
# xavier -- macro
#sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/xavier/freq-2_xavier-1e2.yaml"
#sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/xavier/freq-2_xavier-1e1.yaml"
#sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/xavier/freq-2_xavier-1.yaml"
#for i in 1 2 3 4 5 6 7
#do
#    sbatch -p 1080ti-short  --mem 20000 --gres=gpu:1 --wrap "python code/oned.py --cfg configs/one_d/xavier/freq-2_xavier-1e-"$i".yaml"
#done


