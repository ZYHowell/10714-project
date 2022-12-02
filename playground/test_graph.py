import numpy as np

import needle as ndl
from needle import nn
from needle.graph import build_graph_from_tensor

import torch

batch_size = 1
input_size = 1
hidden_size = 1
bias = True
nonlinearity = 'tanh'
device = ndl.cpu()
ndl.autograd.LAZY_MODE = True

x = np.random.randn(batch_size, input_size).astype(np.float32)
h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

model_ = torch.nn.RNNCell(input_size,
                          hidden_size,
                          nonlinearity=nonlinearity,
                          bias=bias)

model = nn.RNNCell(input_size,
                   hidden_size,
                   device=device,
                   bias=bias,
                   nonlinearity=nonlinearity)
model.W_ih = ndl.Tensor(model_.weight_ih.detach().numpy().transpose(),
                        device=device)
model.W_hh = ndl.Tensor(model_.weight_hh.detach().numpy().transpose(),
                        device=device)

model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)
h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))

graph = build_graph_from_tensor(h)
