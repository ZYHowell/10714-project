import numpy as np

import needle as ndl
from needle import nn, init
from needle.graph import build_graph_from_tensor, pattern_matching_elementwise

from needle.nn import RNN

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, device, dtype):
        kwargs = {
            "low": -1,
            "high": 1,
            "device": device,
            "dtype": dtype,
            "requires_grad": True
        }
        self.W_ih = nn.Parameter(init.rand(input_size, hidden_size, **kwargs))
        self.W_hh = nn.Parameter(init.rand(hidden_size, hidden_size, **kwargs))
        self.bias_ih = nn.Parameter(init.rand(hidden_size, **kwargs))
        self.bias_hh = nn.Parameter(init.rand(hidden_size, **kwargs))
        self.act = nn.ReLU()
    def forward(self, X):
        # ret = self.act(-X @ self.W_ih + self.bias_ih) @ self.W_hh + self.bias_hh
        ret = X @ self.W_ih + self.bias_ih
        return self.act(ret)

device = ndl.cuda()
dtype = "float32"
ndl.autograd.LAZY_MODE = True

input_size = 4
hidden_size = 4
batch_size = 1
seq_len = 128
num_layers = 1
x = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)
h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
model = RNN(input_size, hidden_size, device=device, dtype=dtype)
c,h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))

graph = build_graph_from_tensor(h)

ndl.Tensor.__str__ = ndl.Value.__str__
ndl.Tensor.__repr__ = ndl.Value.__repr__
h: ndl.Tensor

print(graph)
# Test replace by fused op:
# topo_order = graph.topo_order()
# pattern_matching_elementwise(graph)
# print(graph)

# check graph.exec's correctness
assert all(node.cached_data is None for node in graph.nodes)
# cor = model(ndl.Tensor(x, device=device)).realize_cached_data()
# import numpy as np
# import time
# import torch
# torch.cuda.synchronize()
# t0 = time.time()
# for i in range(1000):
#     tmp = graph(*graph.params)
# t1 = time.time()
# print("time:", (t1 - t0)/1000)
# np.testing.assert_allclose(cor.numpy(), tmp.numpy())
# print("done")
