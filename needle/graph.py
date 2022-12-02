from queue import Queue
from typing import Dict, Sequence

from needle.autograd import Op, Tensor, Value
from needle.ops import make_tuple, tuple_get_item
from needle.utils import OrderedSet


class Graph:

    def __init__(self, root):
        self.root = root
        self.users: Dict[Value, set[Value]] = {}
        self.nodes = OrderedSet()
        self.params = OrderedSet()

        self.nodes.add(root)

    def add_user(self, src, user):
        if src not in self.users:
            self.users[src] = set()
        self.users[src].add(user)

    def add_node(self, tensor: Value, user: Value):
        self.nodes.add(tensor)
        self.add_user(tensor, user)

    def add_param(self, val: Value, user: Value):
        self.params.add(val)
        self.add_user(val, user)

    def topo_order(self) -> Sequence[Value]:
        order = OrderedSet()
        use_count = {n: len(n.inputs) for n in list(self.nodes)}
        q = Queue()
        for p in self.params:
            q.put(p)
        while not q.empty():
            node: Value = q.get()
            order.add(node)
            if node is self.root:
                assert q.empty()
                continue
            for user in self.users[node]:
                use_count[user] -= 1
                if use_count[user] == 0:
                    q.put(user)
        return tuple(order)

    def exec(self, *args):
        val_map = {}
        assert len(args) == len(self.params)
        
        def map_or_cached(val: Value):
            if val in self.params or val in self.nodes:
                return val_map[val]
            return val.realize_cached_data()

        for p, v in zip(self.params, args):
            val_map[p] = v.realize_cached_data()

        topo_order = self.topo_order()
        for t in topo_order:
            if t not in val_map:
                val_map[t] = t.op.compute(*[map_or_cached(i) for i in t.inputs])
        return val_map[self.root]


class Fused(Op):

    def __init__(self, graph: Graph):
        self.graph = graph

    def compute(self, *args):
        # TODO(hongyi): replace it by a fused kernel
        self.graph.exec(*args)


def build_graph_from_tensor(root: Tensor):
    queue = Queue()
    queue.put(root)
    visited = set()
    graph = Graph(root)
    # although we can use topo_dfs here, we use bfs instead because topo_dfs
    # cannot handle very large graph.
    while not queue.empty():
        node = queue.get()
        if node.inputs is None:
            continue
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


def replace_all_use_with(origin: Value, new: Value, graph: Graph):
    assert origin in graph.users
    for user in list(graph.users):
        for idx in range(len(user.inputs)):
            if user.inputs[idx] == origin:
                user.inputs[idx] = new


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
