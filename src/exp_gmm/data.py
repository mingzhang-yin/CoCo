import numpy as np
import torch

def fp(*args):  
    tmps = []
    if len(np.shape(args))>1:
        args = np.squeeze(args)
    for arg in args:
        arg = round(float(arg), 4)  
        tmps.append(str(arg))
    print(" ".join(tmps))


def datagen(sudo, frac, args): 
    K = args.K
    N = args.N    
    dim_in = K+sudo.shape[1] if args.spurious else K
    D = np.empty([0, dim_in])
    y = []
    y_true = []
    means = np.sqrt(1.5*K)*args.sigma*np.eye(K)
    for e in range(K):
        D_e = np.random.multivariate_normal(means[e], args.sigma*np.eye(K),[N])
        y_e = np.array([e]*N)
        y_true.extend(list(y_e))
        
        if args.spurious:
            spur_e = np.tile(sudo[[e],:],[N,1])  
              
            flips_id = np.random.choice(N, int(frac*N),replace=False)
            flips = np.random.choice(K, int(frac*N), replace=True)
            spur_e[flips_id]= sudo[[flips],:]
                   
            D_e = np.hstack((D_e, spur_e))
        y.extend(list(y_e))
        D = np.vstack((D,D_e))
    return D, np.array(y)

def testing(net,args,T=10):
    c_r = []
    for i in range(T):
        sudo_test = np.random.uniform(low=0, high=1, size=[args.K, args.K//2+1])
        D_test, y_test = datagen(sudo_test, 0., args)  
        correct = 0
        total = 0
        with torch.no_grad():        
            xs, labels = torch.tensor(D_test), torch.tensor(y_test)
            outputs = net(xs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        c_r.append(correct/total)
        
    print('Accuracy on the test: %.2f%% (%.2f%%)' % (
        100 * np.mean(c_r),100*np.std(c_r)))
    return c_r

def training_eval(net, envs):
    c_r = []
    for [inputs, labels] in envs:      
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        outputs = net(inputs)
        with torch.no_grad():   
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
        c_r.append(correct/labels.size(0))
    print('Accuracy on the train: %.2f%% (%.2f%%)' % (
        100 * np.mean(c_r),100*np.std(c_r)))
    return c_r

def envgen(args, spurious=True):   
    envs = []
    np.random.seed(args.seed)
    for i in range(args.n_env):
        sudo = np.random.uniform(low=0, high=1, size=[args.K, args.K//2+1])
        #D, y = datagen(sudo, 0.01*i, args)
        D, y = datagen(sudo, 0.05, args)
        envs.append([D,y])
    return envs
