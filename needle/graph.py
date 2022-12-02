from queue import Queue
from typing import Callable, Dict, Sequence

from needle.autograd import Op, Tensor, Value
from needle.ops import (make_tuple, tuple_get_item, op_name, register_op,
                        is_ewise_binary, is_ewise_unary)
from needle.utils import OrderedSet

indent_len = 2


def get_indent():
    return " " * indent_len


def pp_shape(data):
    if data is None:
        return "?"
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
        self.users: Dict[Value, set[Value]] = {root: set()}
        self.nodes = OrderedSet()
        self.params = OrderedSet()

        self.nodes.add(root)

    # Graph construction
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
        # we add set() here for the b = a + a case.
        use_count = {n: len(set(n.inputs)) for n in list(self.nodes)}
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
            val_map[p] = v

        topo_order = self.topo_order()
        for t in topo_order:
            if t not in val_map:
                val_map[t] = t.op.compute(*[map_or_cached(i) for i in t.inputs])
        return val_map[self.root]

    def __call__(self, *args):
        return self.exec(*[arg.realize_cached_data() for arg in args])

    # Graph modification
    def replace_all_use_with(self, origin: Value, new: Value):
        """
        Replace a node in the original graph by a new one.
        Cannot replace a parameter.
        """
        assert origin in self.nodes
        for user in list(self.users[origin]):
            for idx in range(len(user.inputs)):
                if user.inputs[idx] == origin:
                    user.inputs[idx] = new
        self.users[new] = self.users[origin]
        self.nodes.add(new)
        self.users[origin].clear()
        if self.root == origin:
            self.root = new

    def remove_node(self, node: Value):
        assert self.root != node
        self.users.pop(node)
        self.nodes.remove(node)
        for invar in node.inputs:
            if invar in self.nodes:
                self.users[invar].discard(node)

    # Given a subgraph, replace it by a fused operator. Then replace all use
    def replace_nodes_by_fused_op(self, nodes: Sequence[Value], root: Value):
        nodes_set = OrderedSet(nodes)
        assert all(node in self.nodes for node in nodes)
        # construct subgraph, fused op and fused tensor from nodes
        subgraph = build_graph_from_tensor(root, lambda x: x not in nodes_set)
        fused_op = Fused(subgraph)
        invars = list(subgraph.params)
        fused_tensor = Tensor.make_from_op(fused_op, invars)
        for inv in invars:
            self.add_user(inv, fused_tensor)
        # replace all nodes in the graph
        self.replace_all_use_with(root, fused_tensor)
        for node in nodes:
            self.remove_node(node)
        return fused_tensor

    # print related
    def pp_graph(self):
        param_str = get_indent() + "params: "
        for param in self.params:
            param_str += param.name() + f" {pp_shape(param.cached_data)}" + ", "
        if len(self.params) > 0:
            param_str = param_str[:-1]
        param_dict = {p: idx for idx, p in enumerate(self.params)}

        eqn_str = get_indent() + "eqns:"
        for node in self.topo_order():
            cur_eqn_str = f"\n{get_indent() * 2}"
            cur_eqn_str += node.name() + " = "
            if node in param_dict:
                cur_eqn_str += f"Parameter({param_dict[node]})"
                eqn_str += cur_eqn_str
                continue

            cur_eqn_str += op_name(node.op) + " ("
            for inv in node.inputs:
                cur_eqn_str += inv.name() + ", "
            if len(node.inputs) > 0:
                cur_eqn_str = cur_eqn_str[:-2]
            cur_eqn_str += ")"
            eqn_str += cur_eqn_str

        root_str = get_indent() + "root var: " + self.root.name()
        ret_str = "{\n" + param_str + "\n" + eqn_str + "\n" + root_str + "\n}"
        return ret_str

    def __str__(self):
        return self.pp_graph()


class Fused(Op):

    def __init__(self, graph: Graph):
        self.graph = graph

    def compute(self, *args):
        # TODO(hongyi): replace it by a fused kernel
        return self.graph.exec(*args)


register_op(Fused, "fused")


def build_graph_from_tensor(root: Tensor, is_leaf_fn=None):
    queue = Queue()
    queue.put(root)
    visited = set()
    graph = Graph(root)
    # handle custom is_leaf(for subgraph)
    default_is_leaf = lambda x: x.cached_data is not None
    if is_leaf_fn is None:
        is_leaf = default_is_leaf
    else:
        assert isinstance(is_leaf_fn, Callable)
        is_leaf = lambda x: is_leaf_fn(x) or default_is_leaf(x)
    # although we can use topo_dfs, we pick bfs instead because topo_dfs cannot
    # handle very deep graph.
    while not queue.empty():
        node = queue.get()
        if node.inputs is None:
            continue
        for input_tensor in node.inputs:
            if is_leaf(input_tensor):
                graph.add_param(input_tensor, node)
            elif input_tensor in visited:
                continue
            else:
                graph.add_node(input_tensor, node)
                queue.put(input_tensor)
                visited.add(input_tensor)
    return graph


# A fused operator may have multiple outputs. We add a tuple operator and get
# its elements as the original outputs
def _add_tuple_and_get_elements(*args):
    t = make_tuple(*args)
    return t, tuple(tuple_get_item(t, i) for i in range(len(args)))


def _collect_multi_roots(nodes: Sequence[Value], graph: Graph):
    assert len(nodes) > 1
    root, new_nodes = _add_tuple_and_get_elements(nodes)
    for outv, new_outv in zip(nodes, new_nodes):
        graph.replace_all_use_with(outv, new_outv)
        graph.remove_node(outv)
    return root, new_nodes


# version 1: An elementwise unary op after any op
def pattern_matching_single_unary(graph: Graph):
    fused = set()
    for node in graph.topo_order():
        if node in graph.params or node in fused:
            continue
        if is_ewise_unary(node.op):
            src = node.inputs[0]
            if src not in fused and len(graph.users[src]) == 1:
                nodes = (src, node)
                graph.replace_nodes_by_fused_op(nodes, node)
                fused.update(nodes)


# version 2: A series of elementwise unary ops after any op
def pattern_matching_unarys(graph):
    pass


# version 3: Add elementwise binary ops as well
def pattern_matching_elementwise(graph):
    pass
