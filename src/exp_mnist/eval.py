from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import pickle
from matplotlib import pyplot as plt
import os
import glob
import seaborn as sns
sns.set_style("white")
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./mnist', help='The path results to be saved.')
    parser.add_argument('--metric', type=int, default=0,help='0/1 for accu on noised/noise-free label ')
    args = parser.parse_args()
    
    metric = args.metric
    
    dirs = glob.glob(os.path.join(args.path, '*'))
    coco = []
    irm = []
    erm = []
    for dir_i in dirs:
        if len(os.listdir(dir_i)) == 0:
            continue       
        method = os.path.basename(dir_i).split('_')[1]
        if method == 'CoCO':
            R = pickle.load(open(os.path.join(dir_i,'mnist.pkl'),'rb'))
            R1 = np.hstack([np.array(R[0])[:,None], np.array(R[1])[:,metric,None],
                np.array(R[2])[:,metric,None]])
            coco.append(R1)
        if method == 'IRM':
            R = pickle.load(open(os.path.join(dir_i,'mnist.pkl'),'rb'))
            R1 = np.hstack([np.array(R[0])[:,None], np.array(R[1])[:,metric,None],
                            np.array(R[2])[:,metric,None]])
            irm.append(R1)
        if method == 'ERM':
            R = pickle.load(open(os.path.join(dir_i,'mnist.pkl'),'rb'))
            R1 = np.hstack([np.array(R[0])[:,None], np.array(R[1])[:,metric,None],
                np.array(R[2])[:,metric,None]])
            erm.append(R1)
    
    mean_co = np.mean(coco, axis=0)
    std_co = np.std(coco, axis=0)
    mean_irm = np.mean(np.array(irm).astype(np.float32), axis=0)
    std_irm = np.std(irm, axis=0)
    mean_erm = np.mean(erm, axis=0)
    std_erm = np.std(erm, axis=0)
    #%%
    plt.figure(figsize=(8,5))
    idx = mean_irm[:,0]
    
    plt.plot(idx, mean_co[:,1],label='CoCo_Training',color='C0')
    plt.fill_between(idx, mean_co[:,1] - std_co[:,1], mean_co[:,1] + std_co[:,1],
                      color='C0', alpha=0.2)
    plt.plot(idx, mean_co[:,2], '--', label='CoCo_Testing',  color='C0')
    plt.fill_between(idx, mean_co[:,2] - std_co[:,2], mean_co[:,2] + std_co[:,2],
                      color='C0', alpha=0.2)
    print('CoCO testing', np.mean(mean_co[-10:,2]))
    
    plt.plot(idx, mean_irm[:,1],label='IRM_Training',color='C1')
    plt.fill_between(idx, mean_irm[:,1] - std_irm[:,1], mean_irm[:,1] + std_irm[:,1],
                     color='C1', alpha=0.2)
    plt.plot(idx, mean_irm[:,2], '--',label='IRM_Testing', color='C1')
    plt.fill_between(idx, mean_irm[:,2] - std_irm[:,2], mean_irm[:,2] + std_irm[:,2],
                     color='C1', alpha=0.2)
    print('IRM testing', np.mean(mean_irm[-10:,2]))
    
    plt.plot(idx, mean_erm[:,1],label='ERM_Training',color='C2')
    plt.fill_between(idx, mean_erm[:,1] - std_erm[:,1], mean_erm[:,1] + std_erm[:,1],
                      color='C2', alpha=0.2)
    plt.plot(idx, mean_erm[:,2], '--',label='ERM_Testing',color='C2')
    plt.fill_between(idx, mean_erm[:,2] - std_erm[:,2], mean_erm[:,2] + std_erm[:,2],
                      color='C2', alpha=0.2)
    
    print('ERM testing', np.mean(mean_erm[-10:,2]))

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=16)
      
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Iteration',fontsize=16)
    lb = 'Accuracy(noised label)' if metric==0 else 'Accuracy(clean label)'
    plt.ylabel(lb,fontsize=16)
    
    plt.savefig(lb+'.pdf', bbox_inches='tight')  