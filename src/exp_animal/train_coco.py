import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from sklearn.metrics import precision_score, recall_score

   
def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values), flush=True)    
        
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
    
    # Define loss function helpers
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
        
        lr = self.args_cmd.lr
        optimizer = optim.Adam(self.clf.fc.parameters(), lr)

        pretty_print('step', 'loss', 'train nll', 'train acc', 'dir_w', 'inv_w','test nll', 'test acc', 'test prec', 'test rec')
        step_r = []
        train_acc_r = []
        test_acc_r = []        
        for step in range(self.args['steps']): 
            
            for env_idx, env in enumerate(self.envs):
                x = env['images']
                y = env['labels']
                loader_tr = DataLoader(self.handler(x, y, transform=self.args['transform']['train']), 
                                       shuffle=True, **self.args['loader_tr_args'])
                self.clf.train()
                nll = acc = 0.0
                
                for batch_idx, (x, y, idxs) in enumerate(loader_tr):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    logits = self.clf(x)
            
                    y.resize_((y.shape[0], 1))
                    train_nll = self.mean_nll(logits, y.float())
                    train_acc, _ = self.mean_accuracy(logits, y.float())
                                    
                    nll += train_nll
                    acc += train_acc
                
                # directional loss
                nll_e = nll / len(loader_tr)
                phi_grad_e = torch.autograd.grad(nll_e, \
                           self.clf.fc.parameters(),create_graph=True)
                if self.use_cuda:
                    train_direct_loss = torch.tensor(0.).cuda()
                    train_irm_loss = torch.tensor(0.).cuda()
                else:  
                    train_direct_loss = torch.tensor(0.)
                    train_irm_loss = torch.tensor(0.)
                    
                for i, phi in enumerate(self.clf.fc.parameters()):
                    train_direct_loss += torch.mean(torch.square(phi*phi_grad_e[i]))
                    train_irm_loss += torch.sum(phi*phi_grad_e[i])
                direct_loss_e = torch.sqrt(train_direct_loss)
                train_irm_loss_e = torch.square(train_irm_loss)
                
                env['nll'] = nll / len(loader_tr)
                env['acc'] = acc / len(loader_tr)
                env['direct_loss'] = direct_loss_e 
                env['irm_loss'] = train_irm_loss_e 
                
            train_nll = torch.stack([self.envs[0]['nll'], self.envs[1]['nll']]).mean()
            train_acc = torch.stack([self.envs[0]['acc'], self.envs[1]['acc']]).mean()
            train_direct_loss = torch.stack([self.envs[0]['direct_loss'], \
                                             self.envs[1]['direct_loss']]).mean()
            train_irm_loss =  torch.stack([self.envs[0]['irm_loss'], \
                                             self.envs[1]['irm_loss']]).mean()
            
            inv_weight = 1 
            dir_weight = 1e-4 
            factor = 1 if step < self.args_cmd.anneal else self.args_cmd.factor
            train_direct_loss_w = dir_weight*train_direct_loss * factor                               
            train_irm_loss_w = inv_weight*train_irm_loss * factor
            
            loss = train_nll + train_direct_loss_w + train_irm_loss_w
            
            loss.backward()
            optimizer.step()
            
            test_loss, test_acc, preds, probs = self.predict(self.X_te, self.Y_te)
            
            test_prec = precision_score(self.Y_te.detach().cpu().numpy(), preds.detach().cpu().numpy())
            test_rec = recall_score(self.Y_te.detach().cpu().numpy(), preds.detach().cpu().numpy())
            model_name = "wild_coco_" + \
                    str(self.args_cmd.lmbd1) + "_" + \
                    str(self.args_cmd.lmbd2)         
            if step % 10 == 0:
                if dir_weight == 0:
                    train_direct_loss_w = train_direct_loss
                pretty_print(np.int32(step), loss.detach().cpu().numpy(), \
                             train_nll.detach().cpu().numpy(), 
                             train_acc.detach().cpu().numpy(), \
                             (train_direct_loss_w).detach().cpu().numpy(), 
                             (train_irm_loss_w).detach().cpu().numpy(), 
                             test_loss.detach().cpu().numpy(), test_acc.detach().cpu().numpy(),
                             test_prec, test_rec)
                step_r.append(step)
                train_acc_r.append([train_acc.detach().cpu().numpy(),train_nll.detach().cpu().numpy()])
                test_acc_r.append([test_acc.detach().cpu().numpy(), test_loss.detach().cpu().numpy()])
                pkl_name = model_name  +  '.pkl'
                pickle.dump([step_r, train_acc_r, test_acc_r], open(pkl_name,'wb'))
            if step % 100 == 0:
                torch.save(self.clf.state_dict(), self.args['model_path'] + model_name + ".pth")                            
        return train_acc.detach().cpu().numpy(), test_acc.detach().cpu().numpy(), preds, probs
