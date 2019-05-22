import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

"""
##### freq 2, vary depth conv ####
xname = 'Depth'
files = ['Outputs/one_d_res_multiple_freq/freq-2_vary-depth/depth_vs_iter_k-50_conv.json',
         'Outputs/one_d_res_multiple_freq/freq-2_vary-depth/depth_vs_iter_k-5_conv.json',
        ]
x_func = (lambda d: [k for k in d.keys()])
fnames = ['k = 50','k = 5']
std = []

plt.ylim(0,900)
###########
"""
"""
##### freq 2, vary channel conv ####
xname = 'Channels'
# freq 50
j1 = 'Outputs/one_d/freq-2_vary-channels/channels_vs_iter_k-50_conv.json'
f1name = 'k = 50'
std = []
# freq 5
j2 = 'Outputs/one_d/freq-2_vary-channels/channels_vs_iter_k-5_conv.json'
f2name = 'k = 5'
plt.ylim(0,250)
############
"""
"""
##### freq 2, vary depth fc ####
xname = 'Depth'
# freq 50
j1 = 'Outputs/one_d/freq-2_vary-depth/depth_vs_iter_k-50_fc.json'
f1name = 'k = 50'
# freq 5
j2 = 'Outputs/one_d/freq-2_vary-depth/depth_vs_iter_k-5_fc.json'
f2name = 'k = 5'
plt.ylim(0,200)
############
"""
"""
##### freq 2, vary channel conv ####
xname = 'Channels'
# freq 50
j1 = 'Outputs/one_d/freq-2_vary-channels/channels_vs_iter_k-50_fc.json'
f1name = 'k = 50'
std = []
# freq 5
j2 = 'Outputs/one_d/freq-2_vary-channels/channels_vs_iter_k-5_fc.json'
f2name = 'k = 5'
plt.ylim(0,200)
############
"""
"""
##### k=50, vary freq, vary channel conv ####
xname = 'Depth'
# freq 8, k = 50
j1 = 'Outputs/one_d/freq-8_vary-depth/depth_vs_iter_k-5_conv.json'
f1name = 'freq-8'
std = []
# freq 4, k = 50
j2 = 'Outputs/one_d/freq-6_vary-depth/depth_vs_iter_k-5_conv.json'
x_func = (lambda d: [k for k in d.keys()])
f2name = 'freq-4'
plt.ylim(0,900)
files = [j1,j2]
############
"""
"""
##### xavier vary init variance -- micro level ####
xname = 'Init. Variance'
files = ['Outputs/one_d/xavier/freq-2_xavier-1'+str(i)+'e-1/depth_vs_iter_k-50_conv.json' for i in range(9,0,-1)]
files += ['Outputs/one_d/xavier/freq-2_xavier-1/depth_vs_iter_k-50_conv.json']
files += ['Outputs/one_d/xavier/freq-2_xavier-'+str(i)+'e-1/depth_vs_iter_k-50_conv.json' for i in range(9,0,-1)]
l = len(files)
files += ['Outputs/one_d/xavier/freq-2_xavier-1'+str(i)+'e-1/depth_vs_iter_k-5_conv.json' for i in range(9,0,-1)]
files += ['Outputs/one_d/xavier/freq-2_xavier-1/depth_vs_iter_k-5_conv.json']
files += ['Outputs/one_d/xavier/freq-2_xavier-'+str(i)+'e-1/depth_vs_iter_k-5_conv.json' for i in range(9,0,-1)]
x_func = (lambda d: [d[k][2] for k in d.keys()])
fnames = ['k = 50' for _ in range(l)] + ['k = 5' for _ in range(l)]
std = []
plt.ylim(0,10000)
############
"""
"""
##### xavier vary init variance -- macro level ####
files = []
xname = 'Init. Variance'
for i in [2,1]:
    files += ['Outputs/one_d/xavier/freq-2_xavier-1e'+str(i)+'/depth_vs_iter_k-50_conv.json' for i in [2,1]]
files += ['Outputs/one_d/xavier/freq-2_xavier-1/depth_vs_iter_k-50_conv.json']
for i in range(1,7,1):
    files += ['Outputs/one_d/xavier/freq-2_xavier-1e-'+str(i)+'/depth_vs_iter_k-50_conv.json' for i in range(1,7,1)]
l = len(files)
for i in [2,1]:
    files += ['Outputs/one_d/xavier/freq-2_xavier-1e'+str(i)+'/depth_vs_iter_k-5_conv.json' for i in [2,1]]
files += ['Outputs/one_d/xavier/freq-2_xavier-1/depth_vs_iter_k-5_conv.json']
for i in range(1,7,1):
    files += ['Outputs/one_d/xavier/freq-2_xavier-1e-'+str(i)+'/depth_vs_iter_k-5_conv.json' for i in range(1,7,1)]
x_func = (lambda d: [np.log(d[k][2])/np.log(10) for k in d.keys()])
fnames = ['k = 50' for _ in range(l)] + ['k = 5' for _ in range(l)]
std = [] 
plt.ylim(0,10000)
############
"""

###### normal vary init variance -- macro level ####
files = []
xname = 'Init. Variance'
#files = ['Outputs/one_d/normal/freq-2_normal-1e'+str(i)+'/depth_vs_iter_k-50_conv.json' for i in [2,1]]
#files += ['Outputs/one_d/normal/freq-2_normal-1/depth_vs_iter_k-50_conv.json']
for i in range(2,8,1):
    for j in [7,5,3,1]:
        files += ['Outputs/one_d/normal/freq-2_normal-'+str(j)+'e-'+str(i)+'/depth_vs_iter_k-50_conv.json']
    #files += ['Outputs/one_d/normal/freq-2_normal-1e-'+str(i)+'/depth_vs_iter_k-50_conv.json']
l = len(files)
#files += ['Outputs/one_d/normal/freq-2_normal-1e'+str(i)+'/depth_vs_iter_k-5_conv.json' for i in [2,1]]
#files += ['Outputs/one_d/normal/freq-2_normal-1/depth_vs_iter_k-5_conv.json']
for i in range(2,8,1):
    for j in [7,5,3,1]:
        files += ['Outputs/one_d/normal/freq-2_normal-'+str(j)+'e-'+str(i)+'/depth_vs_iter_k-5_conv.json']
    #files += ['Outputs/one_d/normal/freq-2_normal-1e-'+str(i)+'/depth_vs_iter_k-5_conv.json']
x_func = (lambda d: [np.log(d[k][2])/np.log(10) for k in d.keys()])
fnames = ['k = 50' for _ in range(l)] + ['k = 5' for _ in range(l)]
std = []
plt.ylim(0,12000)
###########

"""
###### normal vary init variance -- near 1 ####
xname = 'Init. Variance'
files = ['Outputs/one_d/normal/freq-2_normal-1'+str(i)+'e-1/depth_vs_iter_k-50_conv.json' for i in range(9,0,-1)]
files += ['Outputs/one_d/normal/freq-2_normal-1/depth_vs_iter_k-50_conv.json']
files += ['Outputs/one_d/normal/freq-2_normal-'+str(i)+'e-1/depth_vs_iter_k-50_conv.json' for i in range(9,0,-1)]
l = len(files)
files += ['Outputs/one_d/normal/freq-2_normal-1'+str(i)+'e-1/depth_vs_iter_k-5_conv.json' for i in range(9,0,-1)]
files += ['Outputs/one_d/normal/freq-2_normal-1/depth_vs_iter_k-5_conv.json']
files += ['Outputs/one_d/normal/freq-2_normal-'+str(i)+'e-1/depth_vs_iter_k-5_conv.json' for i in range(9,0,-1)]
x_func = (lambda d: [np.log(d[k][2])/np.log(10) for k in d.keys()])
fnames = ['k = 50' for _ in range(l)] + ['k = 5' for _ in range(l)]
std = []
plt.ylim(0,5000)
############
"""






groups = {fn:None for fn in set(fnames)}
for j,flname in zip(files,fnames):
    with open(j,'r') as f:
        fl = json.load(f)
    f.close()
    x = sorted(fl.keys(),key=lambda t:int(t))
    y = [fl[x_][0] for x_ in x]
    std = [fl[x_][1] for x_ in x]
    xaxis = x_func(fl)
    
    if groups[flname] is None:
        groups[flname] = {'x':xaxis,'y':y,'std':std}
    else:
        groups[flname]['x'].extend(xaxis)
        groups[flname]['y'].extend(y)
        groups[flname]['std'].extend(std)
    
for g,p in groups.items():
    print('>>>',p)
    plt.errorbar(p['x'],p['y'],p['std'],label=g,capsize=5)
    #plt.errorbar(xaxis,y,std,label=flname,capsize=5)
plt.legend(prop={'size':10})
plt.ylabel('Iteration')
plt.xlabel(xname)
plt.savefig('plot_'+xname+'.png')


"""
with open(j1,'r') as f:
    f1 = json.load(f)
f.close()
with open(j2,'r') as f:
    f2 = json.load(f)
f.close()
x = sorted(f1.keys(),key=lambda t:int(t))
y1 = [f1[x_][0] for x_ in x]
std1 = [f1[x_][1] for x_ in x]
y2 = [f2[x_][0] for x_ in x]
std2 = [f2[x_][1] for x_ in x]
plt.errorbar(x,y1,std1,label=f1name,capsize=5)
plt.errorbar(x,y2,std2,label=f2name,capsize=5)
plt.legend(prop={'size':10})
plt.ylabel('Iteration')
plt.xlabel(xname)
plt.savefig('plot_'+xname+'.png')
"""

