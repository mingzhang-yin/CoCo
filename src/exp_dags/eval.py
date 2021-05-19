from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns 
import os
import glob

#%%
root = './results8'
dirs = glob.glob(os.path.join(root, '*'))
MAEs = []
for dirs_i in dirs:
    R = pickle.load(open(os.path.join(dirs_i,'DAG.pkl'),'rb'))
    MAE = []
    for eg_results in R:  
        r1, r2, r3, r4, r5, r6, r7, beta = eg_results  
        #choose optimal over tuning parameters
        mae_r1 = [np.mean(np.abs(rr-beta)) for rr in r1]
        mae_r5 = [np.mean(np.abs(rr-beta)) for rr in r5]
        mae_r6 = [np.mean(np.abs(rr-beta)) for rr in r6]
        
        MAE.append([np.min(mae_r1),np.mean(np.abs(r2-beta)),\
                    np.mean(np.abs(r3-beta)),np.mean(np.abs(r4-beta)),
                    np.min(mae_r5), np.min(mae_r6), np.min(mae_r6)])
    MAEs.append(MAE)
    
MAEs = np.array(MAEs)
output = np.mean(MAEs, axis=0).T
std = np.std(MAEs, axis=0).T

patterns = [ ""  , "/" , "-", "\\",  ""  , "/" , "-", "\\"] 
colors = matplotlib.cm.get_cmap('Spectral')
x = np.arange(len(output.T))*2.65
tick = ['case '+str(_+1) for _ in range(len(output.T))]
w = 0.3
plt.figure(figsize=(10,4))

plt.bar(x-3*w, output[1], width=w, yerr=std[1],color=colors(1/len(output)), \
        edgecolor='black', hatch=patterns[0], align='center',capsize=3, label='CoCo')
plt.bar(x-2*w, output[5], width=w, yerr=std[5],color=colors(2/len(output)), \
        edgecolor='black', hatch=patterns[1], align='center',capsize=3, label='RVP')
plt.bar(x-w, output[4], width=w, yerr=std[4],color=colors(3/len(output)), \
        edgecolor='black', hatch=patterns[2], align='center',capsize=3, label='V-REx')
plt.bar(x, output[3], width=w,  yerr=std[3], color=colors(4/len(output)), \
        edgecolor='black', hatch=patterns[3], align='center',capsize=3, label='Naive-CoCo')
plt.bar(x+w, output[-1], width=w, yerr=std[-1], color=colors(5/len(output)), \
        edgecolor='black', hatch=patterns[4], align='center',capsize=3, label='Dantzig')
plt.bar(x+2*w, output[0], width=w, yerr=std[0], color=colors(6/len(output)), \
        edgecolor='black', hatch=patterns[5], align='center',capsize=3, label='IRM')
plt.bar(x+3*w, output[2], width=w, yerr=std[2], color=colors(7/len(output)), \
    edgecolor='black', hatch=patterns[6], align='center',capsize=3, label='ERM')
plt.xticks(x, tick,fontsize='14')
plt.ylabel('MAE',fontsize='14')
plt.ylim([0,0.8])

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
          ncol=7, fancybox=True, shadow=True,fontsize=11)
plt.tight_layout()
plt.show() 
plt.savefig('DAG.pdf')




