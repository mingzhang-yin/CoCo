from __future__ import print_function
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
import errno

def get_dataset(name, path, overwrite=True):
    if name == 'WILDCAM':
        return get_WildCam(path, overwrite=overwrite)

class WildCamFolder(Dataset):
    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    '''
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.all_items=find_classes(os.path.join(self.root))
        self.idx_classes=index_classes(self.all_items)

    def __getitem__(self, index):
        filename=self.all_items[index][0]
        img=str.join('/',[self.all_items[index][2],filename])
        target=self.all_items[index][1] #self.idx_classes[self.all_items[index][1]]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        return  img, target

    def __len__(self):
        return len(self.all_items)

def find_classes(root_dir):
    retour=[]
    for (root,dirs,files) in os.walk(root_dir):
        for f in sorted(files):
            if (f.endswith("jpg")):
                r=root.split('/')
                lr=len(r)
                retour.append((f,"/"+r[lr-1],root))                
    print("== Found %d items "%len(retour))
    return retour

def index_classes(items):
    idx={}
    for i in items:
        if (not i[1] in idx):
            idx[i[1]]=len(idx)
    print("== Found %d classes"% len(idx))
    return idx

def create_nparray(dataset, dataroot, processedroot, overwrite=False):
    """
    Constructs a numpy array of image paths and labels for the dataset - train/test 
    dataset - dataset type - train/test/val
    dataroot - data dir path
    processedroot - where to save the nparray
    overwrite - whether to overwrite the existing numpy array, default false
    """
    if overwrite:
        print(str.join('/', [dataroot, dataset]))
        x = WildCamFolder(str.join('/', [dataroot, dataset]))
        images = []
        labels = []
        for (img, label) in x:
            images.append(img)
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        
        if not os.path.exists(os.path.join(processedroot, dataset)):
            os.mkdir(os.path.join(processedroot, dataset))

        if os.path.isfile(os.path.join(processedroot, dataset, 'images.npy')):
            os.remove(os.path.join(processedroot, dataset, 'images.npy'))
            os.remove(os.path.join(processedroot, dataset, 'labels.npy'))
        np.save(os.path.join(processedroot, dataset, 'images.npy'), images)
        np.save(os.path.join(processedroot, dataset, 'labels.npy'), labels)
    else:
        images = np.load(os.path.join(processedroot, dataset, 'images.npy'))
        labels = np.load(os.path.join(processedroot, dataset, 'labels.npy'))
    return images, labels

def get_WildCam(dataroot, overwrite=False):
    processedroot = str.join('/', [dataroot, 'processed'])
    if not os.path.exists(processedroot):
        os.mkdir(os.path.join(processedroot))
    env_list = [43, 46] 
    envs = []
    for env in env_list:
        dataset_name = 'train' + '_' + str(env)
        X_tr, Y_tr = create_nparray(dataset_name, dataroot, processedroot, overwrite)
        print("train environment: ", env)
        print(sorted(set(list(Y_tr))))
        #print(X_tr[1:5])
        class_indices = []
        for x in enumerate(sorted(set(list(Y_tr)))):
            class_indices.append(x)
        print("class_indices: ", class_indices)
        Y_tr_upd = np.empty(len(Y_tr), dtype=int)
        for i, x in enumerate(list(Y_tr)):
            index = [i for i, y in enumerate(class_indices) if y[1] == x]
            Y_tr_upd[i] = int(index[0])
        Y_tr = torch.from_numpy(Y_tr_upd)
        envs.append({'images': X_tr, 'labels': Y_tr})
    
    X_te, Y_te = create_nparray('test', dataroot, processedroot, overwrite)
    print(sorted(set(list(Y_te))))
    #print(X_te[1:5])
    # Making sure that the class indices are consistent across train and valid sets
    
    Y_te_upd = np.empty(len(Y_te), dtype=int)
    for i, x in enumerate(list(Y_te)):
        index = [i for i, y in enumerate(class_indices) if y[1] == x]
        Y_te_upd[i] = int(index[0])
    Y_te = torch.from_numpy(Y_te_upd)
    
    X_val, Y_val = create_nparray('validation', dataroot, processedroot, overwrite)
    print(sorted(set(list(Y_val))))
    
    Y_val_upd = np.empty(len(Y_val), dtype=int)
    for i, x in enumerate(list(Y_val)):
        index = [i for i, y in enumerate(class_indices) if y[1] == x]
        Y_val_upd[i] = int(index[0])
    Y_val = torch.from_numpy(Y_val_upd)
    
    return envs, X_te, Y_te, X_val, Y_val

def get_handler(name):
    if name == 'WILDCAM':
        return WildCamHandler

class WildCamHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # If there are some gray/ single channel images
        x = Image.open(x).convert('RGB').resize((256, 256))
        if self.transform is not None:
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
