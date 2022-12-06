import numpy as np

import needle as ndl
from needle import nn, init
from needle.graph import build_graph_from_tensor, pattern_matching_elementwise



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
        ret = self.act(-X @ self.W_ih + self.bias_ih) @ self.W_hh + self.bias_hh
        return self.act(ret)

batch_size = 2048
input_size = 2048
hidden_size = 2048
device = ndl.cuda()
dtype = "float32"
ndl.autograd.LAZY_MODE = True
x = np.random.randn(batch_size, input_size).astype(np.float32)

model = Model(input_size, hidden_size, device=device, dtype=dtype)
h = model(ndl.Tensor(x, device=device))

graph = build_graph_from_tensor(h)

ndl.Tensor.__str__ = ndl.Value.__str__
ndl.Tensor.__repr__ = ndl.Value.__repr__
h: ndl.Tensor

print(graph)
# Test replace by fused op:
topo_order = graph.topo_order()
pattern_matching_elementwise(graph)
print(graph)

# check graph.exec's correctness
assert all(node.cached_data is None for node in graph.nodes)
cor = model(ndl.Tensor(x, device=device)).realize_cached_data()
assert graph(*graph.params) == cor
