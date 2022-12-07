from queue import Queue
from typing import Any, Callable, Dict, Sequence, Tuple, Union

from needle.autograd import Op, Tensor, Value
from needle.ops import (make_tuple, tuple_get_item, op_name, register_op,
                        is_ewise_binary, is_ewise_unary, is_broadcast, _should_broadcast_to,
                        AbstractArray, infer_shape, MatMul)
from needle.utils import OrderedSet
from needle.backend_ndarray import NDArray
from needle.backend_selection import array_api
import needle as ndl
import torch

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
                input = [map_or_cached(i) for i in t.inputs]
                tmp =  t.op.compute(*input)
                t.cached_data = tmp
                val_map[t] = tmp
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
        self.users[new] = OrderedSet(self.users[origin])
        self.nodes.add(new)
        self.users[origin].clear()
        if self.root == origin:
            self.root = new

    def remove_node(self, node: Value):
        assert self.root != node
        self.users.pop(node)
        self.nodes.remove(node)
        for invar in node.inputs:
            if invar in self.users:
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
    def pp_param(self, param):
        return param.name() + f" {pp_shape(param.cached_data)}"

    def pp_outv(self, var):
        return var.name()

    def pp_inv(self, var):
        return var.name()

    def pp_graph(self):
        param_str = get_indent() + "params: "
        for param in self.params:
            param_str += self.pp_param(param) + ", "
        if len(self.params) > 0:
            param_str = param_str[:-1]
        param_dict = {p: idx for idx, p in enumerate(self.params)}

        eqn_str = get_indent() + "eqns:"
        for node in self.topo_order():
            cur_eqn_str = f"\n{get_indent() * 2}"
            cur_eqn_str += self.pp_outv(node) + " = "
            if node in param_dict:
                cur_eqn_str += f"Parameter({param_dict[node]})"
                eqn_str += cur_eqn_str
                continue

            cur_eqn_str += op_name(node.op) + " ("
            for inv in node.inputs:
                cur_eqn_str += self.pp_inv(inv) + ", "
            if len(node.inputs) > 0:
                cur_eqn_str = cur_eqn_str[:-2]
            cur_eqn_str += ")"
            eqn_str += cur_eqn_str

        root_str = get_indent() + "root var: " + self.root.name()
        ret_str = "{\n" + param_str + "\n" + eqn_str + "\n" + root_str + "\n}"
        return ret_str

    def __str__(self):
        return self.pp_graph()


class AbstractValue(Value):
    aval: Union[AbstractArray, Tuple[AbstractArray]] = None

    def __init__(self, val: Value, inputs, is_param):
        if is_param:
            self._init(None, [],
                       num_outputs=val.num_outputs,
                       cached_data=None,
                       requires_grad=False)
        else:
            self._init(val.op,
                       inputs,
                       num_outputs=val.num_outputs,
                       cached_data=None,
                       requires_grad=False)

    def realized_cached_data(self):
        raise NotImplementedError()

    def __str__(self):
        type_name = "AbsValue"
        return f"{type_name}({self.name()})"


class FreezedGraph(Graph):
    """A freezed graph which should behave as a closed function."""
    def __init__(self, graph: Graph):
        var_map: Dict[Value, AbstractValue] = {}

        def map_or_create(val):
            if val in var_map:
                return var_map[val]
            is_param = val in graph.params
            inputs = ([] if is_param else
                      [map_or_create(inv) for inv in val.inputs])
            var_map[val] = AbstractValue(val, inputs, is_param)
            return var_map[val]

        self.params = tuple([map_or_create(var) for var in graph.params])
        self.nodes = tuple([map_or_create(var) for var in graph.nodes])
        self.root = map_or_create(graph.root)
        self.users = {}
        for var in graph.users:
            self.users[var_map[var]] = set(
                [var_map[u] for u in list(graph.users[var])])
        self._topo_order = super().topo_order()

        # Infer the shape and dtype of each node in the graph
        for node in graph.topo_order():
            mapped = var_map[node]
            assert mapped.aval is None
            if node in graph.params:
                mapped.aval = AbstractArray(node.cached_data.shape,
                                            node.cached_data.dtype)
            else:
                mapped.aval = infer_shape(mapped.op, *mapped.inputs)

    def topo_order(self):
        return self._topo_order

    def _pp_var_with_shape(self, var):
        return var.name() + f" {pp_shape(var.aval)}"

    def pp_param(self, param):
        return self._pp_var_with_shape(param)

    def pp_outv(self, var):
        return self._pp_var_with_shape(var)

    def pp_inv(self, var):
        return self._pp_var_with_shape(var)


class Fused(Op):

    def __init__(self, graph: Graph):
        self.graph = graph
        self.freezed_topo_order = None
        
    def compute(self, *args):
        import time
        if not isinstance(self.graph, FreezedGraph):
            self.graph = FreezedGraph(self.graph)
            self.freezed_topo_order = self.graph.topo_order()
        ewise_op_names = []
        is_scalar_op = []
        tensor_input_topo_order = []
        scalar_input_topo_order = []
        val_map = {}
        assert len(args) == len(self.graph.params)
        def map_or_cached(val: Value):
            if val in self.graph.params:
                return val_map[val]
            return None

        for p, v in zip(self.graph.params, args):
            val_map[p] = v        
        device = ndl.cuda()
        cnt = 0
        for  freeze_t in self.freezed_topo_order:
            if freeze_t.op is None:
                continue
            if cnt ==0:
                assert isinstance(freeze_t.op, MatMul)
                m,n = freeze_t.inputs[0].aval.shape
                _, p = freeze_t.inputs[1].aval.shape
                matmul_a = map_or_cached(freeze_t.inputs[0])
                matmul_b = map_or_cached(freeze_t.inputs[1])
                out = NDArray.make((m, p), device=device)
            else:
                if len(freeze_t.inputs) == 2:
                    additional_input= map_or_cached(freeze_t.inputs[1])
                    s1 = freeze_t.inputs[0].aval.shape
                    s2 = freeze_t.inputs[1].aval.shape
                    if _should_broadcast_to(s1, s2):
                        raise ValueError("broadcasting to second input is not supported")
                    elif _should_broadcast_to(s2, s1):
                        if (len(s2) < len(s1)):
                            additional_input = additional_input.reshape((1,) * (len(s1) - len(s2)) + s2)
                        additional_input = array_api.broadcast_to(additional_input, s1).compact()
                    tensor_input_topo_order.append(additional_input._handle)
                    scalar_input_topo_order.append(0.)
                elif len(freeze_t.inputs)==1:
                    dummy = NDArray.make((1,1), device=device)
                    tensor_input_topo_order.append(dummy._handle)
                    scalar_input_topo_order.append(0.)
                else:
                    raise ValueError("ewise func with 2 or more inputs is not supported")
                ewise_op_names.append(op_name(freeze_t.op))
                is_scalar_op.append(int(is_ewise_unary(freeze_t.op)))
            cnt+=1
        if ewise_op_names == ["add", "relu"]:
            device.matmul_fused_bias_relu(matmul_a.compact()._handle, matmul_b.compact()._handle, tensor_input_topo_order[0], out._handle, m, n, p)
            # device.matmul_fused(matmul_a.compact()._handle, matmul_b.compact()._handle, out._handle, m, n, p, ewise_op_names,  tensor_input_topo_order, scalar_input_topo_order, is_scalar_op,)
            # device.matmul_fused_two_ewise(matmul_a.compact()._handle, matmul_b.compact()._handle, out._handle, m, n, p, ewise_op_names,  tensor_input_topo_order, scalar_input_topo_order, is_scalar_op,)
        elif ewise_op_names == ["add", "add", "tanh"]:
            device.matmul_fused_bias_bias_tanh(matmul_a.compact()._handle, matmul_b.compact()._handle, tensor_input_topo_order[0], tensor_input_topo_order[1], out._handle, m, n, p)
        return out
        # return self.graph.exec(*args)

register_op(Fused, "fused")


def build_graph_from_tensor(root: Tensor, is_leaf_fn=None):
    queue = Queue()
    queue.put(root)
    visited = set()
    graph = Graph(root)
    # handle custom`` is_leaf(for subgraph)
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
            if input_tensor in visited:
                graph.add_user(input_tensor, node)
                continue
            elif is_leaf(input_tensor):
                graph.add_param(input_tensor, node)
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
    assert len(nodes) > 1, len(nodes)
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


# TODO: to handle c = neg(a) + relu(a), we need to build a dep tree, and check
# whether all ops between a node and its parent on the tree is unary or binary.
def look_up_from(root: Value, graph: Graph):
    visited = set()
    fused = OrderedSet()
    q = Queue()

    q.put(root)
    visited.add(root)
    while not q.empty():
        node = q.get()
        for inv in node.inputs:
            if inv in visited:
                continue
            if (len(graph.users[inv]) == 1 and
                (is_ewise_unary(inv.op) or is_ewise_binary(inv.op) or
                 is_broadcast(inv.op))):
                fused.add(inv)
                q.put(inv)
            visited.add(inv)
    return fused

def look_down_from(root: Value, graph: Graph):
    visited = set()
    fused = OrderedSet()
    final_root = OrderedSet()
    q = Queue()

    q.put(root)
    fused.add(root)
    visited.add(root)
    final_root.add(root)
    while not q.empty():
        node = q.get()
        if len(graph.users[node]) > 1:
            # TODO(hongyi): this condition may be removed, but i don't think you
            # can do so complicated optimization
            pass
        else:
            added_users = []
            for user in graph.users[node]:
                if user in visited:
                    continue
                elif (is_ewise_unary(user.op) or is_ewise_binary(user.op) or
                      is_broadcast(user.op)):
                    fused.add(user)
                    q.put(user)
                    added_users.append(user)
            if node in final_root and len(added_users) > 0:
                final_root.remove(node)
                final_root.update(added_users)
        visited.add(node)
    if len(final_root) == 1:
        final_root = list(final_root)[0]
    else:
        final_root = _collect_multi_roots(tuple(final_root), graph)
    return fused, final_root


def up_down_look(node: Value, graph: Graph):
    fused = OrderedSet()

    # look up from node
    # fused.update(look_up_from(node, graph))

    # look down
    new_fused, root = look_down_from(node, graph)
    fused.update(new_fused)

    # look up from the new root: root = node + neg(param), we need to involve
    # the neg operand into the fused graph
    # fused.update(look_up_from(root, graph))
    return fused, root


# version 2: All elementwise operators around an op
def pattern_matching_elementwise(graph: Graph):
    all_fused = set()
    for node in graph.topo_order():
        if node in graph.params or node in all_fused:
            continue
        if (is_ewise_unary(node.op) or is_ewise_binary(node.op) or
                is_broadcast(node.op)):
            continue
        fused, root = up_down_look(node, graph)
        if len(fused) > 1:
            graph.replace_nodes_by_fused_op(list(fused), root)
            all_fused.update(fused)
