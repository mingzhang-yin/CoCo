from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import os
import argparse
import torch
from torch import distributions as td

#%%
def example1(e=1, N=10000):
    #independent causes
    beta = np.array([2.0, 1.5, 0])
    x2_e = torch.normal(1, 0.5,[N,1])
    x1_e = td.Uniform(-1,1).sample([N,1]) + x2_e 
    y_e = beta[0] * x1_e + beta[1] * x2_e + torch.randn(N,1)
    z = e*y_e +  torch.randn(N,1)  
    return (torch.cat((x1_e, x2_e, z), 1), y_e, beta)

def example2(e=1, N=10000):
    #correlated causes
    beta = np.array([2.0, 1.5, 0, 0])
    x2_e = torch.normal(1, 0.5,[N,1])
    x1_e = td.Uniform(-1,1).sample([N,1]) + x2_e 
    x3_e = torch.sin(x1_e) + torch.normal(0, 0.5,[N,1])
    y_e = beta[0] * x1_e + beta[1] * x3_e + torch.randn(N,1)
    z = e*y_e +  torch.randn(N,1)  
    return (torch.cat((x1_e, x3_e, x2_e, z), 1), y_e, beta)

def example3(e=1, N=10000):
    #mediator
    beta = np.array([2.0, 1.5, 1.0, 0])
    x2_e = torch.normal(1, 1/2.,[N,1])
    x1_e = td.Uniform(-1,1).sample([N,1]) + x2_e 
    x3_e = torch.sin(x1_e) + torch.normal(0, 0.5,[N,1])
    y_e = beta[0] * x1_e + beta[1] * x3_e + beta[2] * x2_e + torch.randn(N,1)
    z = e*y_e +  torch.randn(N,1)
    return (torch.cat((x1_e, x3_e, x2_e, z), 1), y_e, beta)

def example4(e=1, N=10000):
    #unobserved direct cause, observe ancestor
    beta = np.array([2.0, 1.0, 0])
    x2_e = torch.normal(1, 1/2.,[N,1])
    x1_e = x2_e + td.Uniform(-1,1).sample([N,1])
    u1 = x1_e + x2_e + torch.normal(0, 0.5,[N,1])
    y_e = beta[1] * x2_e + beta[0] * u1 + torch.randn(N,1)
    z = e*y_e +  torch.randn(N,1) 
    beta[1] += beta[0] 
    return (torch.cat((x1_e, x2_e, z), 1), y_e, beta)

def example5(e=1, N=10000):
    #collider
    beta = np.array([1.0, 0])
    x1_e = torch.normal(0, e, [N,1])
    y_e = x1_e  + torch.normal(0, 1,[N,1])
    z = 2*y_e + x1_e + torch.normal(0, e, [N,1])
    return (torch.cat((x1_e, z), 1), y_e, beta)

def IRMv1(environments, args, lmbd):
    estimate_r = []
    phi = torch.nn.Parameter(torch.normal(1,0.2,[environments[0][0].shape[1],1]))
    dummy_w = torch.nn.Parameter(torch.Tensor([1.0])) 
    opt1 = torch.optim.Adam([phi], lr=args.lrs) 
    phi_old = 0
    for iteration in range(args.max_iter):
        error = 0
        penalty = 0
        for i in range(len(environments)):
            x_e, y_e, beta = environments[i]
            error_e = 0.5*mse(x_e @ phi * dummy_w, y_e).mean()   
            error += error_e
            
            phi_grad_out = torch.autograd.grad(error_e, dummy_w, create_graph=True)
            penalty += torch.square(phi_grad_out[0]) 
  
        opt1.zero_grad()
        total_loss =  ((1/lmbd)*error +penalty)*100
        total_loss.backward()     
        opt1.step()
        
        estimate = phi.view(-1).detach().numpy()
        estimate_r.append(estimate)
        
        if iteration % 2000 == 0: 
            phi_new = np.mean(estimate_r[-100:],axis=0)
            print(phi_new)
            if ((np.sum(np.abs(phi_new - phi_old))<0.001) & (iteration>=10000)):
                break
            else:
                phi_old = phi_new
    
    return np.mean(estimate_r[-100:],axis=0)

def Naive_CoCO(environments, args):
    estimate_r = []
    phi = torch.nn.Parameter(torch.normal(1,0.2,[environments[0][0].shape[1],1]))
    opt1 = torch.optim.Adam([phi], lr=args.lrs) 
    phi_old = 0
    for iteration in range(args.max_iter):
        error = 0
        penalty = 0
        for i in range(len(environments)):
            x_e, y_e, beta = environments[i]
            error_e = 0.5*mse(x_e @ phi, y_e).mean()   
            error += error_e
            
            phi_grad_out = torch.autograd.grad(error_e, phi, create_graph=True)
            penalty += torch.square(phi_grad_out[0][0] + \
                        torch.sum(phi_grad_out[0][1:]*phi[1:])) 
  
        opt1.zero_grad()
        total_loss =  (penalty)*100
        total_loss.backward()     
        opt1.step()
        
        estimate = phi.view(-1).detach().numpy()
        estimate_r.append(estimate)
        
        if iteration % 2000 == 0: 
            phi_new = np.mean(estimate_r[-100:],axis=0)
            print(phi_new)
            if ((np.sum(np.abs(phi_new - phi_old))<0.001) & (iteration>=10000)):
                break
            else:
                phi_old = phi_new                    
    return np.mean(estimate_r[-100:],axis=0)

def CoCO(environments, args):
    estimate_r = []
    phi = torch.nn.Parameter(torch.normal(1,0.2,[environments[0][0].shape[1],1]))
    opt1 = torch.optim.Adam([phi], lr=args.lrs)  
    phi_old = 0
    for iteration in range(args.max_iter):
        error = 0
        penalty = 0
        for i in range(len(environments)):
            x_e, y_e, beta = environments[i]           
            error_e = 0.5*mse(x_e @ phi, y_e).mean()  
            error += error_e

            phi_grad_out = torch.autograd.grad(error_e, phi,create_graph=True)
            penalty += torch.square(phi_grad_out[0][0]) + \
                torch.sum(torch.square(phi_grad_out[0][1:]*phi[1:])) 
     
        opt1.zero_grad()
        total_loss =  torch.sqrt(penalty)
        total_loss.backward()     
        opt1.step()
        
        estimate = phi.view(-1).detach().numpy()
        estimate_r.append(estimate)
        if iteration % 2000 == 0: 
            phi_new = np.mean(estimate_r[-100:],axis=0)
            print(phi_new)
            if ((np.sum(np.abs(phi_new - phi_old))<0.001) & (iteration>=10000)):
                break
            else:
                phi_old = phi_new                
    return np.mean(estimate_r[-100:],axis=0)

def ERM(environments, args):
    estimate_r = []
    phi = torch.nn.Parameter(torch.normal(1,0.2,[environments[0][0].shape[1],1]))
    opt1 = torch.optim.SGD([phi], lr=0.002) 
    phi_old = 0
    for iteration in range(args.max_iter):
        error = 0
        for i in range(len(environments)):
            x_e, y_e, beta = environments[i]              
            error_e = 0.5*mse(x_e @ phi , y_e).mean()  
            error += error_e           
        opt1.zero_grad()
        error.backward()     
        opt1.step()   
        
        estimate = phi.view(-1).detach().numpy()
        estimate_r.append(estimate)
        
        if iteration % 2000 == 0:
            phi_new = np.mean(estimate_r[-100:],axis=0)
            print(phi_new)
            if ((np.sum(np.abs(phi_new - phi_old))<0.001) & (iteration>=10000)):
                break
            else:
                phi_old = phi_new                
    return np.mean(estimate_r[-100:],axis=0)

def run(methods, environments, args): 
    beta = environments[0][-1]      
    if 'IRMv1' in methods:
        result1 = []
        if isinstance(args.lmbd,int):
            print('IRMv1,', 'causal coef=', beta)
            result1.append(IRMv1(environments, args, lmbd=args.lmbd))
        else:
            for lmbd in args.lmbd:
                print('IRMv1,', 'causal coef=', beta, 'lmbd=', lmbd)
                result1.append(IRMv1(environments, args, lmbd=lmbd))
    if 'CoCO' in methods:
        print('CoCO,', 'causal coef=', beta)
        result2 = CoCO(environments, args)           
    if 'ERM' in methods:  
        print('ERM,', 'causal coef=', beta)
        result3 = ERM(environments, args)        
    if 'Naive_CoCO' in methods:
        print('Naive_CoCO,', 'causal coef=', beta)
        result4 = Naive_CoCO(environments, args)
    
    return [result1, result2, result3, result4, beta]

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=30000, help='max iteration.')
    parser.add_argument('--N', type=int, default=10000, help='number of data per env.')
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--lmbd', type=int, default=[2, 20, 200], help='lmbd for irmv1')
    args = parser.parse_args()
    
    np.set_printoptions(suppress=True, precision=2, linewidth=300)
    path = os.path.join(args.path, f'dag_{args.seed}')
    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    examples = [example1, example2, example3, example4, example5]
    methods = ['Naive_CoCO', 'IRMv1', 'CoCO','ERM']
    results = []
    for data_gen in examples:
        print('#######################')
        print(data_gen)
        print('#######################')
        mse = torch.nn.MSELoss(reduction="none") 
        environments = [data_gen(e = 0.2, N=args.N), \
                        data_gen(e = 1.0, N=args.N), \
                        data_gen(e = 1.5, N=args.N), \
                        data_gen(e = 2.5, N=args.N)]  
        args.lrs = 0.1 if data_gen == example5 else 0.01
        results.append(run(methods, environments, args))
    pickle.dump(results,open(os.path.join(path, 'DAG.pkl'),'wb'))



