import os
import sys
import json
import numpy as np


if __name__ == '__main__':
    output_dir = sys.argv[1]
    print('Collecting PSNR results in:',output_dir,'\n')
    print('\t\t'.join(['Img',
                       'True Best',
                       'Pred Best']))
    for img in os.listdir(output_dir):
        img_dir = os.path.join(output_dir,img)
        with open(os.path.join(img_dir,'res.json'),'r') as f:
            res = json.load(f)
        f.close()
        true_best_psnr = '{:.3f}'.format(res['true_best_psnr'])
        pred_best_psnr = '{:.3f}'.format(res['pred_best_psnr'])

        baseline_res = []
        for baseline in os.listdir(os.path.join(img_dir,'baselines')):
            pass

        disp_res = list(map(str,[img,true_best_psnr,pred_best_psnr]))
        print('\t\t'.join(disp_res))


