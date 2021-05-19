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
    beta = np.array([3.0, 2, 0])
    x2_mean = np.random.uniform(0,1)
    x2_e = torch.normal(x2_mean, e,[N,1])
    x1_mean = np.random.uniform(0,1)
    x1_e = torch.normal(x1_mean, e,[N,1])
    y_e = beta[0] * x1_e + beta[1] * x2_e + torch.randn(N,1)
    z = e*y_e +  torch.normal(0, e, [N,1])
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
    y_e = beta[0] * x1_e + beta[1] * x3_e + beta[2] * x2_e + torch.normal(0, e,[N,1])
    z = e*y_e +  torch.randn(N,1)
    return (torch.cat((x1_e, x3_e, x2_e, z), 1), y_e, beta)

def example4(e=1, N=10000):
    #unobserved mediator, observe ancestor
    beta = np.array([2.0, 1.0, 0])
    x2_e = torch.normal(1, 1/2.,[N,1])
    x1_range = np.random.uniform(1,2)
    x1_e = x2_e + td.Uniform(0,x1_range).sample([N,1])
    u1 = x1_e + x2_e + torch.normal(0, 0.5,[N,1])
    y_e = beta[0] * u1 + beta[1] * x2_e + torch.randn(N,1)
    z = e*y_e +  torch.randn(N,1) 
    beta[1] += beta[0] 
    return (torch.cat((x1_e, x2_e, z), 1), y_e, beta)

def example5(e=1, N=10000):
    #collider
    beta = np.array([2.0, 0])
    x1_e = torch.normal(1, 1/2.,[N,1])
    y_e = beta[0] * x1_e + torch.randn(N,1)
    z = e*y_e/2 + 0.5*x1_e + torch.randn(N,1) 
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

def Naive_CoCo(environments, args):
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

def CoCo(environments, args):
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



def Rex(environments, args, lmbd):
    estimate_r = []
    phi = torch.nn.Parameter(torch.normal(1,0.2,[environments[0][0].shape[1],1]))
    opt1 = torch.optim.Adam([phi], lr=args.lrs) 
    phi_old = 0
    for iteration in range(args.max_iter):
        error = []
        for i in range(len(environments)):
            x_e, y_e, beta = environments[i]
            error_e = 0.5*mse(x_e @ phi, y_e).mean()   
            error.append(error_e)
            
        losses = torch.stack(error)
  
        opt1.zero_grad()
        total_loss =  losses.sum() + lmbd * losses.var()
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

def RVP(environments, args, lmbd):
    estimate_r = []
    phi = torch.nn.Parameter(torch.normal(1,0.2,[environments[0][0].shape[1],1]))
    opt1 = torch.optim.Adam([phi], lr=args.lrs) 
    phi_old = 0
    for iteration in range(args.max_iter):
        error = []
        for i in range(len(environments)):
            x_e, y_e, beta = environments[i]
            error_e = 0.5*mse(x_e @ phi, y_e).mean()   
            error.append(error_e)
            
        losses = torch.stack(error)
  
        opt1.zero_grad()
        total_loss =  (losses.sum() + lmbd * torch.sqrt(losses.var()+1e-8))*10
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

def danzig(environments, args):
    Gs = []
    Zs = []
    for i in range(len(environments)):
        x_e, y_e, beta = environments[i]
        Gs.append(np.matmul(np.transpose(x_e),x_e)/len(x_e))
        Zs.append(np.matmul(np.transpose(x_e),y_e)/len(x_e))
    phi = torch.matmul(torch.inverse(Gs[0]-Gs[1]), Zs[0]-Zs[1])
    return torch.squeeze(phi)

def run(methods, environments, args):   
    beta = environments[0][-1]    
    if 'IRMv1' in methods:
        result1 = []
        for lmbd in [2,20,200]:
            print('IRMv1,', 'causal coef=', beta, 'lmbd=', lmbd)
            result1.append(IRMv1(environments, args, lmbd=lmbd))
    if 'CoCo' in methods:
        print('CoCo,', 'causal coef=', beta)
        result2 = CoCo(environments, args)           
    if 'ERM' in methods: 
        print('ERM,', 'causal coef=', beta)
        result3 = ERM(environments, args)        
    if 'Naive_CoCo' in methods:
        print('Naive_CoCo,', 'causal coef=', beta)
        result4 = Naive_CoCo(environments, args)
    if 'Rex' in methods:
        result5 = []
        for lmbd in [100,1000,10000]:
            print('Rex,', 'causal coef=', beta)
            result5.append(Rex(environments, args, lmbd=lmbd))
    if 'RVP' in methods:
        result6 = []
        for lmbd in [10,100,1000]:
            print('RVP,', 'causal coef=', beta)
            result6.append(RVP(environments, args, lmbd=lmbd))
    if 'danzig' in methods: 
        print('danzig,', 'causal coef=', beta)
        result7 = danzig(environments, args) 
    return [result1, result2, result3, result4, result5, result6, result7, beta]
    

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=100000, help='max iteration.')
    parser.add_argument('--N', type=int, default=10000, help='number of data per env.')
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    args = parser.parse_args()
    
    np.set_printoptions(suppress=True, precision=2, linewidth=300)
    path = os.path.join(args.path, f'dag_{args.seed}')
    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    examples = [example1, example2, example3, example4, example5]
    methods = ['Naive_CoCo', 'IRMv1', 'CoCo','ERM','Rex','RVP', 'danzig']
    results = []
    for data_gen in examples:
        print('#######################')
        print(data_gen)
        print('#######################')
        mse = torch.nn.MSELoss(reduction="none")  
        environments = [data_gen(e = 0.5, N=args.N), 
                        data_gen(e = 2, N=args.N)]  
        args.lrs = 0.1 if data_gen == example5 else 0.01
        results.append(run(methods, environments, args))
    pickle.dump(results,open(os.path.join(path, 'DAG.pkl'),'wb'))



