from queue import Queue
from typing import Dict, Sequence

from needle.autograd import Op, Tensor, Value
from needle.ops import make_tuple, tuple_get_item
from needle.utils import OrderedSet

indent_len = 4


def get_indent():
    return " " * indent_len


def pp_shape(data):
    dtype = str(data.dtype)
    d_str = ""
    if "float" in dtype:
        d_str = "f"
    elif "int" in dtype:
        d_str = "i"
    elif "bool" in dtype:
        d_str = "pred"
    else:
        raise NotImplementedError(f"unsupported dtype {dtype}")
    if "32" in dtype:
        d_str += "32"
    elif "16" in dtype:
        d_str += "16"
    elif "8" in dtype:
        d_str += "8"
    elif "bool" in dtype:
        pass
    else:
        raise NotImplementedError(f"unsupported dtype {dtype}")
    s_str = "["
    for d in data.shape:
        s_str += str(d) + ","
    if len(s_str) > 1:
        s_str = s_str[:-1]
    s_str += "]"
    return d_str + s_str


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

    def pp_graph(self):
        param_str = "params: "
        for param in self.params:
            param_str += param.name() + f" {pp_shape(param.cached_data)}" + ", "
        if len(self.params) > 0:
            param_str = param_str[:-1]
        param_dict = {p: idx for idx, p in enumerate(self.params)}

        eqn_str = "eqns:"
        for node in self.topo_order():
            cur_eqn_str = f"\n{get_indent()}"
            cur_eqn_str += node.name() + " = "
            if node in param_dict:
                cur_eqn_str += f"Parameter({param_dict[node]})"
                eqn_str += cur_eqn_str
                continue

            # The op name is just a hack.
            op_name = str(type(node.op))
            op_name = op_name[len("<class 'needle.ops."):-2]
            cur_eqn_str += op_name + " ("
            for inv in node.inputs:
                cur_eqn_str += inv.name() + ", "
            if len(node.inputs) > 0:
                cur_eqn_str = cur_eqn_str[:-2]
            cur_eqn_str += ")"
            eqn_str += cur_eqn_str

        root_str = "root var: " + self.root.name()
        ret_str = "{\n" + param_str + "\n" + eqn_str + "\n" + root_str + "\n}"
        return ret_str

    def __str__(self):
        return self.pp_graph()


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


# A fused operator may have multiple outputs. We add a tuple operator and get
# its elements as the original outputs
def _add_tuple_and_get_elements(*args):
    t = make_tuple(*args)
    return tuple(tuple_get_item(t, i) for i in range(len(args)))


# Given a subgraph, replace it by a fused operator. Then replace all use
def _replace_nodes_by_fused_op(subgraph, graph):
    pass


# version 1: An elementwise unary op after any op
def pattern_matching_single_unary(graph):
    pass


# version 2: A series of elementwise unary ops after any op
def pattern_matching_unarys(graph):
    pass


# version 3: Add elementwise binary ops as well
def pattern_matching_elementwise(graph):
    pass
