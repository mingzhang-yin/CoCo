import numpy as np
import pickle
import torch
import argparse
import os
from data import envgen, training_eval, testing

mse = torch.nn.MSELoss(reduction="none") 
torch.set_default_dtype(torch.float64)

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--N', type=int, default=1000, help='data in one component')
    parser.add_argument('--K', type=int, default=5, help='number of component')
    parser.add_argument('--sigma', type=float, default=1., help='std of GMM')
    parser.add_argument('--lmbd', type=float,default=30.0, help='weight for coco term')
    parser.add_argument('--lmbd_irm', type=float,default=100.0, help='weight for irm')
    parser.add_argument('--lmbd_rex', type=float,default=10000.0, help='weight for rex')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n_env', type=int, default=5)
    parser.add_argument('--steps', type=int, default=5001)
    parser.add_argument('--spurious', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--method', default='Rex', help='ERM, IRM, Rex, CoCo')
    args = parser.parse_args()
    
    path = os.path.join(args.path, f'gmm_{args.method}_{args.lmbd_rex}')
    if not os.path.exists(path):
        os.makedirs(path)

    dim_in = args.K+(args.K//2+1) if args.spurious else args.K
    dim_out = args.K
    dims =  [dim_in, 10, 10, dim_out]
    net = torch.nn.Sequential(
        torch.nn.Linear(dims[0], dims[1]),
        torch.nn.Sigmoid(),
        torch.nn.Linear(dims[1], dims[2]),
        torch.nn.Sigmoid(), 
        torch.nn.Linear(dims[2], dims[3])
    )
    if args.method == 'IRM':
        dummy_w  = torch.nn.Parameter(torch.tensor([1.]))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  
    
    envs = envgen(args)
    
    test_r = []
    train_r = []
    iter_r = []
    for epoch in range(args.steps): 
        if ((args.method == 'CoCo') or (args.method == 'ERM')):
            risk = 0
            coco_loss = 0 #over envs
            for [inputs, labels] in envs:  
                inputs = torch.tensor(inputs)
                labels = torch.tensor(labels)
                outputs = net(inputs)
                risk_e = criterion(outputs, labels)
                risk += risk_e
                phi_grad = torch.autograd.grad(risk_e, \
                            net.parameters(),create_graph=True)
                coco_loss_e = 0 
                for i, phi in enumerate(net.parameters()):
                    coco_loss_e += torch.mean(torch.square(phi*phi_grad[i]))
                coco_loss += torch.sqrt(coco_loss_e)
            risk = risk/len(envs)
            coco_loss = coco_loss/len(envs)
        
            optimizer.zero_grad()  
            if args.method == 'CoCo':
                lmbd_risk = 1 if epoch <(args.steps/2) else 0.1
                tot_loss = lmbd_risk*risk + args.lmbd*coco_loss if args.spurious else risk
            if args.method == 'ERM':
                tot_loss = risk
            tot_loss.backward()
            optimizer.step()  
        
        if (args.method == 'IRM'):
            risk = 0
            loss = 0
            for [inputs, labels] in envs:  
                inputs = torch.tensor(inputs)
                labels = torch.tensor(labels)
                outputs = net(inputs)        
                
                risk_e = criterion(outputs*dummy_w, labels)             
                w_grad = torch.autograd.grad(risk_e, \
                            dummy_w,create_graph=True)[0]
                loss_e = torch.square(w_grad)
                loss += loss_e
                risk += risk_e
            
            optimizer.zero_grad() 
            risk = risk/len(envs)
            loss = loss/len(envs) 
    
            lmbd_risk = 1 if epoch <(args.steps/2) else 0.1
            tot_loss = risk*lmbd_risk + args.lmbd_irm*loss  
            tot_loss.backward()
            optimizer.step()              
        
        if (args.method == 'Rex'):
            risk = 0
            loss = 0
            risks = []
            for [inputs, labels] in envs:  
                inputs = torch.tensor(inputs)
                labels = torch.tensor(labels)
                outputs = net(inputs)        
                
                risk_e = criterion(outputs, labels)  
                risks.append(risk_e)
                risk += risk_e
            optimizer.zero_grad() 
            risk = risk/len(envs)
            loss = torch.stack(risks).var()
    
            lmbd_risk = 1 if epoch <(args.steps/2) else 0.1
            tot_loss = lmbd_risk*risk + args.lmbd_rex*loss  
            tot_loss.backward()
            optimizer.step()   
            
        if epoch % 100 == 0: 
            print('epoch',epoch, '########################',flush=True) 
            test_perform = testing(net, args)
            train_perform = training_eval(net, envs)
            test_r.append([np.mean(test_perform), np.std(test_perform)])
            train_r.append([np.mean(train_perform), np.std(train_perform)])
            iter_r.append(epoch)
    test_r = np.array(test_r)[:,0]
    train_r = np.array(train_r)[:,0]
    
    pickle.dump([iter_r, train_r, test_r],open(os.path.join(path, 'gmm_' + str(args.seed)+'.pkl'),'wb'))
    

    




