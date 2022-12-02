from queue import Queue
from typing import Sequence

from needle.autograd import Op, Tensor, Value
from needle.ops import make_tuple, tuple_get_item
from needle.utils import OrderedSet


class Graph:

    def __init__(self):
        self.op_to_tensor = {}
        self.users = {}
        self.params = OrderedSet()

    def add_user(self, src, user):
        if src not in self.users:
            self.users[src] = set()
        self.users[src].add(user)

    def add_node(self, tensor: Value, user: Value):
        self.op_to_tensor[tensor.op] = tensor
        self.add_user(tensor, user)

    def add_param(self, val: Value, user: Value):
        self.params.add(val)
        self.add_user(val, user)


class Fused(Op):

    def __init__(self, graph, arg_to_params):
        self.graph = graph
        self.arg_to_params = arg_to_params

    def exec_graph(self, *args):
        pass

    def compute(self, *args):
        # TODO(hongyi): replace it by a fused kernel
        self.exec_graph(*args)


def build_graph_from_tensor(root: Tensor):
    queue = Queue()
    queue.put(root)
    visited = set()
    graph = Graph()
    # although we can use topo_dfs here, we use bfs instead because topo_dfs
    # cannot handle very large graph.
    while not queue.empty():
        node = queue.get()
        for input_tensor in node.inputs:
            if input_tensor.cached_data is not None:
                graph.add_param(input_tensor, node)
            elif input_tensor in visited:
                continue
            else:
                graph.add_node(input_tensor, node)
                queue.put(input_tensor)
                visited.add(input_tensor)
    return graph


def build_graph_from_tensors(root: Sequence[Tensor]):
    root = make_tuple(*root)
    return build_graph_from_tensor(root)


# Given a subgraph, replace it by a fused operator. Then replace all use
def _replace_nodes_by_fused_op(graph):
    pass

# A fused operator may have multiple outputs. We add a tuple operator and get
# its elements as the original outputs
def _add_tuple_and_get_elements():
    pass

# version 1: An elementwise unary op after any op
def pattern_matching_unary(graph):
    pass

# version 2: A series of elementwise unary ops after any op

# version 3: Add elementwise binary ops as well
