from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import argparse
from matplotlib import pyplot as plt

import os
import glob

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./R1', help='The path results to be saved.')
    args = parser.parse_args()
    root = args.path
    dirs = glob.glob(os.path.join(root, '*'))
    
    coco = []
    irm = []
    erm = []
    for dir_i in dirs:
        method = os.path.basename(dir_i).split('_')[1]
        if method == 'CoCO':
            R = pickle.load(open(os.path.join(dir_i,'gmm.pkl'),'rb'))
            coco.append(R)
        if method == 'IRM':
            R = pickle.load(open(os.path.join(dir_i,'gmm.pkl'),'rb'))
            irm.append(R)
        if method == 'ERM':
            R = pickle.load(open(os.path.join(dir_i,'gmm.pkl'),'rb'))
            erm.append(R)
    
    mean_co = np.mean(coco, axis=0)
    std_co = np.std(coco, axis=0)
    mean_irm = np.mean(irm, axis=0)
    std_irm = np.std(irm, axis=0)
    mean_erm = np.mean(erm, axis=0)
    std_erm = np.std(erm, axis=0)
    
    plt.figure(figsize=(7,6))
    idx = mean_co[0]
    mean = mean_co[1]
    std = std_co[1]
    
    plt.plot(idx, mean,label=r'$CoCO -Training$',color='C0')
    plt.fill_between(idx, mean - std, mean + std,
                     color='C0', alpha=0.2)
    
    print('CoCO training', np.mean(mean[-5:]))
    
    mean = mean_co[2]
    std = std_co[2]
    plt.plot(idx, mean,'--', label='$CoCO-Testing$',color='C0')
    plt.fill_between(idx, mean - std, mean + std,
                     color='C0', alpha=0.2)
    
    print('CoCO testing', np.mean(mean[-5:]))
    
    mean = mean_irm[1]
    std = std_irm[1]
    plt.plot(idx, mean,label='$IRM-Training$',color='C1')
    plt.fill_between(idx, mean - std, mean + std,
                     color='C1', alpha=0.2)
    
    print('IRM training', np.mean(mean[-5:]))
    
    mean = mean_irm[2]
    std = std_irm[2]
    plt.plot(idx, mean,'--', label='$IRM-Testing$',color='C1')
    plt.fill_between(idx, mean - std, mean + std,
                     color='C1', alpha=0.2)
    
    print('IRM testing', np.mean(mean[-5:]))
    
    
    mean = mean_erm[1]
    std = std_erm[1]
    plt.plot(idx, mean,label='$ERM-Training$',color='C2')
    plt.fill_between(idx, mean - std, mean + std,
                     color='C2', alpha=0.2)
    
    print('ERM training', np.mean(mean[-5:]))
    
    mean = mean_erm[2]
    std = std_erm[2]
    plt.plot(idx, mean,'--', label='$ERM-Testing$',color='C2')
    plt.fill_between(idx, mean - std, mean + std,
                     color='C2', alpha=0.2)
    
    print('ERM testing', np.mean(mean[-5:]))
    
    # ax = plt.gca()
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.84, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=12)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Iteration',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    #plt.tight_layout()
    plt.show()
    
    plt.savefig('gmm1.pdf', bbox_inches='tight')
    
    #%%
    root = './mean'
    dirs = glob.glob(os.path.join(root, '*'))
    dirs = np.sort(dirs)
    idx = []
    mean = []
    std = []
    for dir_i in dirs:
        idx.append(float(os.path.basename(dir_i)))
        sub_dirs = glob.glob(os.path.join(dir_i, '*'))
        
        r = []
        for dir_j in sub_dirs:
            R = pickle.load(open(os.path.join(dir_j,'gmm.pkl'),'rb'))
            r.append(R[-1][-1])
        mean.append(np.mean(r))
        std.append(np.std(r))
    
    index = np.argsort(idx)
    idx = np.sort(idx)
    mean = np.array(mean)[index]
    std = np.array(std)[index]
    
    plt.figure()
    plt.plot(idx, 1-np.array(mean),'o-', color='C0')
    plt.errorbar(idx, 1-np.array(mean), yerr = std,  color='C0', capsize=5)
    
    plt.axhline(y=0.475, color='k', linestyle='--',linewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Noise scale',fontsize=16)
    plt.ylabel('Error',fontsize=16)
    plt.ylim([0.0,0.65])
    plt.locator_params(nbins=8)
    plt.tight_layout()
    
    #plt.savefig('noise_c.pdf', bbox_inches='tight')
    
    #%%
    root = './env'
    dirs = glob.glob(os.path.join(root, '*'))
    dirs = np.sort(dirs)
    
    idx = []
    mean = []
    std = []
    for dir_i in dirs:
        idx.append(float(os.path.basename(dir_i)))
        sub_dirs = glob.glob(os.path.join(dir_i, '*'))
        
        r = []
        for dir_j in sub_dirs:
            R = pickle.load(open(os.path.join(dir_j,'gmm.pkl'),'rb'))
            r.append(R[-1][-1])
        mean.append(np.mean(r))
        std.append(np.std(r))
    
    index = np.argsort(idx)
    idx = np.sort(idx)
    mean = np.array(mean)[index]
    std = np.array(std)[index]
    
    plt.plot(idx[:-2], (1-np.array(mean))[:-2],'o-', color='C0')
    plt.axhline(y=0.475, color='k', linestyle='--',linewidth=2)
    plt.errorbar(idx[:-2], (1-np.array(mean))[:-2], yerr = std[:-2],  color='C0', capsize=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Number of Environments',fontsize=16)
    plt.ylabel('Error',fontsize=16)
    plt.ylim([0.,0.9])
    plt.locator_params(nbins=8)
    plt.tight_layout()
    
    #plt.savefig('n_env.pdf', bbox_inches='tight')
    #%%
    root = './lmbd'
    dirs = glob.glob(os.path.join(root, '*'))
    dirs = np.sort(dirs)
    idx = []
    mean = []
    std = []
    for dir_i in dirs:
        idx.append(float(os.path.basename(dir_i)))
        sub_dirs = glob.glob(os.path.join(dir_i, '*'))
        
        r = []
        for dir_j in sub_dirs:
            R = pickle.load(open(os.path.join(dir_j,'gmm.pkl'),'rb'))
            r.append(R[-1][-1])
        mean.append(np.mean(r))
        std.append(np.std(r))
    index = np.argsort(idx)
    idx = np.sort(idx)
    mean = np.array(mean)[index]
    std = np.array(std)[index]
    
    plt.plot(idx[:-1], (1-np.array(mean))[:-1],'o-', color='C0')
    plt.errorbar(idx[:-1], (1-np.array(mean))[:-1], yerr = std[:-1],  color='C0', capsize=5)
    plt.axhline(y=0.475, color='k', linestyle='--',linewidth=2)
    plt.ylim([0.0,0.9])
    plt.locator_params(nbins=8)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('$1/\lambda_r$',fontsize=16)
    plt.ylabel('Error',fontsize=16)
    plt.tight_layout()
    #plt.savefig('lmbd.pdf', bbox_inches='tight')
