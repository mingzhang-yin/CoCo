import argparse
import numpy as np
import torch
import os
from aux import make_environment, MLP, mean_nll, mean_accuracy, pretty_print
from torchvision import datasets
from torch import optim
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colored MNIST')
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--steps', type=int, default=30001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--method', default='CoCO', help='ERM, IRM, CoCO')    
    parser.add_argument('--grayscale_model', action='store_true')   
    #IRM
    parser.add_argument('--penalty_anneal_iters', type=int, default=190)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.00110794568 )
    parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)
    #CoCO
    parser.add_argument('--coco_weight', type=float, default=500)
    flags = parser.parse_args()
    
    path = os.path.join(flags.path, f'mnist_{flags.method}_{flags.seed}')
    if not os.path.exists(path):
        os.makedirs(path)

    np.random.seed(flags.seed)
    torch.manual_seed(flags.seed)
    torch.cuda.manual_seed(flags.seed)
    
    print('Flags:')
    for k,v in sorted(vars(flags).items()):
      print("\t{}: {}".format(k, v))
      
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

      
    envs = [
      make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
      make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
      make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]
    
    mlp = MLP(flags).cuda()
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    
    pretty_print('step', 'train acc', 'train acc_true', 'train nll', \
                 'strong penalty', 'weak penalty', 'test nll', 'test acc', 'test acc true')
        
    step_r = []
    train_acc_r = []
    test_acc_r = []
    for step in range(flags.steps):
        if ((flags.method == 'CoCO') or (flags.method == 'ERM')):
          risk = 0
          coco = 0
          for env in envs:
            logits = mlp(env['images'])
            
            risk_e = mean_nll(logits, env['labels'])
            phi_grad = torch.autograd.grad(risk_e, \
                      mlp.parameters(),create_graph=True)
                
            coco_e = 0    
            for i, phi in enumerate(mlp.parameters()):
              coco_e += torch.mean(torch.square(phi*phi_grad[i]))
            coco += torch.sqrt(coco_e)
           
            env['nll'] = risk_e
            env['acc'] = mean_accuracy(logits, env['labels'])  
            env['acc_true'] = mean_accuracy(logits, env['labels_nonoise'])        
            env['coco'] = coco       
        
          train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
          train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
          train_acc_true = torch.stack([envs[0]['acc_true'], envs[1]['acc_true']]).mean()
          if flags.method == 'CoCO':
              coco_loss = torch.stack([envs[0]['coco'], envs[1]['coco']]).mean()
              train_penalty = flags.coco_weight * coco_loss
              entropy_loss = train_nll.clone()
              
              lambda_r = 1 if step < (flags.steps/2) else 0.1    
              loss = flags.coco_weight*coco_loss + lambda_r*entropy_loss 
          if flags.method == 'ERM':
              loss = train_nll.clone()
              train_penalty = torch.tensor(0.)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          test_nll = envs[2]['nll']
          test_acc = envs[2]['acc']
          test_acc_true = envs[2]['acc_true']
          
        if flags.method == 'IRM':
          def penalty(logits, y):
            scale = torch.tensor(1.).cuda().requires_grad_()
            loss = mean_nll(logits * scale, y)
            grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
            return torch.sum(grad**2)  
          for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['acc_true'] = mean_accuracy(logits, env['labels_nonoise'])
            env['penalty'] = penalty(logits, env['labels'])
      
          train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
          train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
          train_acc_true = torch.stack([envs[0]['acc_true'], envs[1]['acc_true']]).mean()
          train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()
        
          weight_norm = torch.tensor(0.).cuda()
          for w in mlp.parameters():
            weight_norm += w.norm().pow(2)
        
          loss = train_nll.clone()
          loss += flags.l2_regularizer_weight * weight_norm
            
          penalty_weight = (flags.penalty_weight 
                if step >= flags.penalty_anneal_iters else 1.0)
          loss += penalty_weight * train_penalty
          if penalty_weight > 1.0:
              loss /= penalty_weight
        
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
            
          test_acc = envs[2]['acc']
          test_acc_true = envs[2]['acc_true']
        
        if step % 100 == 0:
          pretty_print(
              np.int32(step),
              train_acc.detach().cpu().numpy(),
              train_acc_true.detach().cpu().numpy(),
              train_nll.detach().cpu().numpy(),
              train_penalty.detach().cpu().numpy(),
              test_acc.detach().cpu().numpy(),
              test_acc_true.detach().cpu().numpy()
          )
          step_r.append(step)
          train_acc_r.append([train_acc.detach().cpu().numpy(),train_acc_true.detach().cpu().numpy()])
          test_acc_r.append([test_acc.detach().cpu().numpy(), test_acc_true.detach().cpu().numpy()])
    
    pickle.dump([step_r, train_acc_r, test_acc_r], open(os.path.join(path, 'mnist.pkl'),'wb'))
    
      
      
      
      
      
      
      
      
    
      
      
      
      
