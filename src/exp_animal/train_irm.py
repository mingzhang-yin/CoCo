import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle

from torch import autograd
from torch.utils.data import DataLoader

from sklearn.metrics import  precision_score, recall_score
  
def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))    
        
class Train:
    def __init__(self, envs, X_te, Y_te, net, handler, args, args_cmd):
        self.envs = envs
        self.X_te = X_te
        self.Y_te = Y_te
        self.net = net
        self.handler = handler
        self.args = args
        self.args_cmd = args_cmd
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_distribution(self):
        return self.class_distribution
    
    def mean_nll(self, logits, y):
        return F.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(self, logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean(), preds

    def predict(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y,
                                            transform=self.args['transform']['test']),
                               shuffle=True, **self.args['loader_te_args'])
        self.clf.eval()
        nll = acc = 0.0
        preds = torch.zeros(len(Y), 1, dtype=torch.float)
        preds_Y = torch.zeros(len(Y), 1, dtype=torch.float)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                y.resize_((y.shape[0], 1))
                train_nll = self.mean_nll(out, y.float())
                train_acc, temp_preds = self.mean_accuracy(out, y.float())
                
                nll += train_nll
                acc += train_acc

                probs = torch.sigmoid(out)
                if str(self.device) == 'cuda':
                    preds[idxs] = probs.cpu()
                    preds_Y[idxs] = temp_preds.cpu()
                else:
                    preds[idxs] = probs           
                    preds_Y[idxs] = temp_preds

        return nll/len(loader_te), acc/len(loader_te), preds_Y, preds     

    def train(self):        
        n_classes = self.args['n_classes']
        self.clf = self.net(n_classes=n_classes).to(self.device)
        
        pretty_print('step', 'loss', 'train nll', 'train acc', 'train penalty', 'test nll', 'test acc', 'test prec', 'test rec')
        step_r = []
        train_acc_r = []
        test_acc_r = []
        lr = self.args_cmd.lr
        optimizer = optim.Adam(self.clf.fc.parameters(), lr)
        for step in range(self.args['steps']): 
            for env_idx, env in enumerate(self.envs):
                x = env['images']
                y = env['labels']
                loader_tr = DataLoader(self.handler(x, y, transform=self.args['transform']['train']), 
                                       shuffle=True, **self.args['loader_tr_args'])
                self.clf.train()
                nll = acc = 0.0
                if self.use_cuda:
                    scale = torch.tensor(1.).cuda().requires_grad_()
                else:
                    scale = torch.tensor(1.).requires_grad_()
                for batch_idx, (x, y, idxs) in enumerate(loader_tr):
                    x, y = x.to(self.device), y.to(self.device)
                    
                    logits = self.clf(x)
            
                    y.resize_((y.shape[0], 1))

                    train_nll = self.mean_nll(logits*scale, y.float())
                    train_acc, _ = self.mean_accuracy(logits, y.float())
                    
                    nll += train_nll
                    acc += train_acc
                    
                nll_e = nll / len(loader_tr)
                env['nll'] = nll / len(loader_tr)
                env['acc'] = acc / len(loader_tr)
                grad_e = autograd.grad(nll_e, [scale], create_graph=True)[0]
                env['penalty'] = (grad_e**2)
            
            
            train_nll = torch.stack([self.envs[0]['nll'], self.envs[1]['nll']]).mean()
            train_acc = torch.stack([self.envs[0]['acc'], self.envs[1]['acc']]).mean()
            train_penalty = torch.stack([self.envs[0]['penalty'], self.envs[1]['penalty']]).mean()
            if self.use_cuda:
                weight_norm = torch.tensor(0.).cuda()
            else:
                weight_norm = torch.tensor(0.)
            if self.args['fc_only']:
                for w in self.clf.fc.parameters():
                    weight_norm += w.norm().pow(2)
            else:
                for w in self.clf.parameters():
                    weight_norm += w.norm().pow(2)     
            
            
            
            if self.args_cmd.lmbd>0:
                #loss += self.args['optimizer_args']['l2_regularizer_weight'] * weight_norm
                
                #erm_weight =  100 if step < self.args_cmd.anneal else 0.001
                
                penalty_weight = 1 if step < self.args_cmd.anneal else self.args_cmd.lmbd
                loss = train_nll + penalty_weight*train_penalty

                #if penalty_weight > 1.0:
                #    # Rescale the entire loss to keep gradients in a reasonable range
                #    loss /= penalty_weight
    
                #if erm_weight>1:
                #loss /= erm_weight  
            else:
                penalty_weight = 0
                loss = train_nll.clone()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            test_loss, test_acc, preds, probs = self.predict(self.X_te, self.Y_te)
            
            test_prec = precision_score(self.Y_te.detach().cpu().numpy(), preds.detach().cpu().numpy())
            test_rec = recall_score(self.Y_te.detach().cpu().numpy(), preds.detach().cpu().numpy())
            if self.args_cmd.lmbd >0:    
                model_name = "wild_irm_" + str(self.args_cmd.lmbd) 
            else:
                model_name = "wild_erm_" + str(self.args_cmd.lmbd) 
            if step % 10 == 0:
                pretty_print(np.int32(step), loss.detach().cpu().numpy(), train_nll.detach().cpu().numpy(), 
                             train_acc.detach().cpu().numpy(), \
                             (penalty_weight *train_penalty).detach().cpu().numpy(), 
                             test_loss.detach().cpu().numpy(), test_acc.detach().cpu().numpy(),
                             test_prec, test_rec)
                step_r.append(step)
                train_acc_r.append([train_acc.detach().cpu().numpy(),train_nll.detach().cpu().numpy()])
                test_acc_r.append([test_acc.detach().cpu().numpy(), test_loss.detach().cpu().numpy()])
                pkl_name = model_name  +  '.pkl'
                pickle.dump([step_r, train_acc_r, test_acc_r], open(pkl_name,'wb'))
            if step % 100 == 0:
                # if step>0:
                #     lr = np.max([lr/2, 1e-5])
                #     for g in optimizer.param_groups:
                #         g['lr'] = lr
                torch.save(self.clf.state_dict(), self.args['model_path'] + model_name + ".pth")                
        return train_acc.detach().cpu().numpy(), test_acc.detach().cpu().numpy(), preds, probs
