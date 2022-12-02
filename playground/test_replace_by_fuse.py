import numpy as np

import needle as ndl
from needle import nn
from needle.graph import build_graph_from_tensor

batch_size = 1
input_size = 1
hidden_size = 1
bias = True
nonlinearity = 'tanh'
device = ndl.cpu()
ndl.autograd.LAZY_MODE = True

x = np.random.randn(batch_size, input_size).astype(np.float32)
h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

model = nn.RNNCell(input_size,
                   hidden_size,
                   device=device,
                   bias=bias,
                   nonlinearity=nonlinearity)
h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))

graph = build_graph_from_tensor(h)

ndl.Tensor.__str__ = ndl.Value.__str__
ndl.Tensor.__repr__ = ndl.Value.__repr__
h: ndl.Tensor

print(graph)
# Test replace by fused op:
topo_order = graph.topo_order()
graph.replace_nodes_by_fused_op(topo_order[-2:], topo_order[-1])
print(graph)

# check graph.exec's correctness
assert all(node.cached_data is None for node in graph.nodes)
cor = model(ndl.Tensor(x, device=device),
            ndl.Tensor(h0, device=device)).realize_cached_data()
assert graph(*graph.params) == cor
print(cor)
