from queue import Queue

from needle.autograd import Op, Tensor, Value
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


class SubGraph(Op):

    def __init__(self, graph):
        self.graph = graph

    def compute(self, *args):
        pass


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


def pattern_matching_on_graph(graph):
    pass
