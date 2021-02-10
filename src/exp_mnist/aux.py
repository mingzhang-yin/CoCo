import numpy as np
import torch
from torch import nn


def make_environment(images, labels, e):
  def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
  def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1
  # 2x subsample for computational convenience
  images = images.reshape((-1, 28, 28))[:, ::2, ::2]
  # Assign a binary label based on the digit; flip label with probability 0.25
  labels = (labels > 5).float()
  labels_nonoise = labels.clone()
  labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
  # Assign a color based on the label; flip the color with probability e
  colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
  # Apply the color to the image by zeroing out the other color channel
  images = torch.stack([images, images], dim=1)
  images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
  return {
    'images': (images.float() / 255.).cuda(),
    'labels': labels[:, None].cuda(),
    'labels_nonoise': labels_nonoise[:, None].cuda()
  }


class MLP(nn.Module):
  def __init__(self, flags):
    super(MLP, self).__init__()
    if flags.grayscale_model:
      lin1 = nn.Linear(14 * 14, flags.hidden_dim)
    else:
      lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
    lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
    lin3 = nn.Linear(flags.hidden_dim, 1)
    for lin in [lin1, lin2, lin3]:
      nn.init.xavier_uniform_(lin.weight)
      nn.init.zeros_(lin.bias)
    self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    self.grayscale_model = flags.grayscale_model
  def forward(self, input):
    if self.grayscale_model:
      out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
    else:
      out = input.view(input.shape[0], 2 * 14 * 14)
    out = self._main(out)
    return out

def mean_nll(logits, y):
  return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
  preds = (logits > 0.).float()
  return ((preds - y).abs() < 1e-2).float().mean()


def pretty_print(*values):
  col_width = 13
  def format_val(v):
    if not isinstance(v, str):
      v = np.array2string(v, precision=5, floatmode='fixed')
    return v.ljust(col_width)
  str_values = [format_val(v) for v in values]
  print("   ".join(str_values), flush=True)
  

  
  
  
  
  
  
  
    
      
      
      
      
