"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


###### Begin my tool functions
def _reduce_grad_axes(x: Tensor, grad_x: Tensor):
    """Reduce axes expanded in grad_x"""
    nlead = len(grad_x.shape) - len(x.shape)
    diff, = numpy.nonzero(
            tuple(s != d for s, d in zip(grad_x.shape[nlead:], x.shape)))
    reduce_axes = tuple(range(nlead)) + tuple(nlead + diff)
    if (reduce_axes):
        return reshape(summation(grad_x, reduce_axes), x.shape)
    return grad_x

def keepdim_shape(shape, axes):
    if isinstance(axes, int):
        expanded_shape = list(shape)
        expanded_shape[axes] = 1
    else:
        axes_set = set(axes or range(len(shape)))
        expanded_shape = numpy.where(
            [i in axes_set for i in range(len(shape))],
            numpy.ones_like(shape), shape)
    return expanded_shape

def _should_broadcast_to(s1, s2):
    # Return whether s1 should be broadcasted to s2
    return len(s1) < len(s2) or any(a1 == 1 != a2 for a1, a2 in zip(s1, s2))

def maybe_broadcast(ary_1, ary_2):
    s1 = ary_1.shape
    s2 = ary_2.shape
    if (len(s1) < len(s2)):
        ary_1 = ary_1.reshape((1,) * (len(s2) - len(s1)) + s1)
    if (len(s2) < len(s1)):
        ary_2 = ary_2.reshape((1,) * (len(s1) - len(s2)) + s2)
    s1 = ary_1.shape
    s2 = ary_2.shape
    if any(a1 == 1 != a2 for a1, a2 in zip(s1, s2)):
        return array_api.broadcast_to(ary_1, s2).compact(), ary_2
    if any(a2 == 1 != a1 for a2, a1 in zip(s2, s1)):
        return ary_1, array_api.broadcast_to(ary_2, s1).compact()
    return ary_1, ary_2
###### End my tool functions


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        a, b = maybe_broadcast(a, b)
        return a + b


    def gradient(self, out_grad: Tensor, node: Tensor):
        a, b = node.inputs
        return _reduce_grad_axes(a, out_grad), _reduce_grad_axes(b, out_grad)


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        a, b = maybe_broadcast(a, b)
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        grad_lhs = _reduce_grad_axes(lhs, out_grad * rhs)
        grad_rhs = _reduce_grad_axes(rhs, out_grad * lhs)
        return grad_lhs, grad_rhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return multiply(
            out_grad, mul_scalar(power_scalar(a, self.scalar - 1),
                                              self.scalar))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        a, b = maybe_broadcast(a, b)
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = divide(out_grad, b)
        grad_b = negate(divide(multiply(out_grad, a), power_scalar(b, 2)))
        return _reduce_grad_axes(a, grad_a), _reduce_grad_axes(b, grad_b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        ndim = len(a.shape)
        new_np_axes = list(range(ndim))
        if self.axes is None:
            new_np_axes[-1] = ndim - 2
            new_np_axes[-2] = ndim - 1
        else:
            new_np_axes[self.axes[0]] = self.axes[1]
            new_np_axes[self.axes[1]] = self.axes[0]
        return a.permute(new_np_axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if len(a.shape) < len(self.shape):
            a = a.reshape((1,) * (len(self.shape) - len(a.shape)) + a.shape)
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        dst_shape = node.inputs[0].shape
        src_shape = out_grad.shape
        nlead = len(src_shape) - len(dst_shape)
        diff, = numpy.nonzero(
            tuple(s != d for s, d in zip(src_shape[nlead:], dst_shape)))
        reduce_axes = tuple(range(nlead)) + tuple(nlead + diff)
        return reshape(summation(out_grad, reduce_axes), dst_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return a.sum(self.axes)
        # Use this pattern because the stupid ta does not support reducing multi
        # axes
        axes = self.axes
        if axes is None:
            axes = list(range(len(a.shape)))
        for i, axis in enumerate(sorted(axes)):
            a = a.sum(axis - i)
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # This api does not have keep_dim option, so it will always not keep_dim
        src_shape = node.inputs[0].shape
        axes = self.axes
        if axes is None:
            axes = list(range(len(src_shape)))
        expanded_shape = keepdim_shape(src_shape, axes)
        return broadcast_to(reshape(out_grad, expanded_shape), src_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        maybe_s1 = b.shape[:-2] + a.shape[-2:]
        maybe_s2 = a.shape[:-2] + b.shape[-2:]
        if _should_broadcast_to(a.shape, maybe_s1):
            a = array_api.broadcast_to(a, maybe_s1).compact()
        if _should_broadcast_to(b.shape, maybe_s2):
            b = array_api.broadcast_to(b, maybe_s2).compact()
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        return _reduce_grad_axes(a, grad_a), _reduce_grad_axes(b, grad_b)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, exp(node.inputs[0]))
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        mask = (x.realize_cached_data() >= 0)
        if hasattr(mask, "astype"):
            mask = mask.astype(x.dtype)
        return multiply(out_grad, Tensor.make_const(mask))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes, )
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z_max = array_api.broadcast_to(Z.max(self.axes, keepdims=True),
                                       Z.shape)
        return (array_api.log(array_api.exp(Z - z_max).sum(self.axes)) +
                Z.max(self.axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        z_max = Tensor.make_const(
            Z.realize_cached_data().max(self.axes, keepdims=True))
        tmp = exp(Z - z_max)
        tmp_2 = summation(tmp, axes=self.axes)
        return multiply(
            broadcast_to(reshape(divide(out_grad, tmp_2), z_max.shape),
                         Z.shape), tmp)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = tanh(node.inputs[0])**2
        x = 1 - x
        x = out_grad * x
        return out_grad * (1 - tanh(node.inputs[0])**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        shape = shape[:self.axis] + (len(args), ) + shape[self.axis:]
        ret = array_api.empty(shape, device=args[0].device)
        for i in range(0, len(args)):
            slices = [slice(None)] * len(args[0].shape)
            slices.insert(self.axis, i)
            ret[tuple(slices)] = args[i]
        return ret
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        ret = []
        shape = list(A.shape)
        shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            ret.append(A[tuple(slices)].compact().reshape(shape))
        return tuple(ret)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        slices = [slice(None)] * len(shape)
        for i in self.axes:
            if i < len(shape):
                shape[i] *= (self.dilation + 1)
                slices[i] = slice(None, None, self.dilation + 1)
        new_buffer = array_api.full(shape, 0, device=a.device)
        new_buffer[tuple(slices)] = a
        return new_buffer

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        new_stride = list(a.strides)
        for i in self.axes:
            if i < len(shape):
                shape[i] //= (self.dilation + 1)
                new_stride[i] *= (self.dilation + 1)
        return NDArray.make(shape,
                            strides=new_stride,
                            device=a._device,
                            handle=a._handle).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # pad input
        p = self.padding
        padded_A = A.pad(((0, 0),) + ((p,) * 2,) * 2 + ((0, 0),))
        # img2col
        kh, kw, ci, co = B.shape
        n, h, w, c = padded_A.shape
        assert c == ci
        ns, hs, ws, cs = padded_A.strides
        out_h = (h - kh) // self.stride + 1
        out_w = (w - kw) // self.stride + 1
        batch_dims = (n, out_h, out_w,)
        col_shape = batch_dims + (kw, kh, ci)
        col_strides = (ns, hs * self.stride, ws * self.stride, hs, ws, cs,)
        col_acc = kw * kh * ci
        col = array_api.NDArray.make(col_shape, col_strides, padded_A._device,
                                     padded_A._handle).compact()
        col = col.reshape((numpy.prod(batch_dims), col_acc))
        # dot
        out_shape = batch_dims + (co, )
        B = B.reshape((col_acc, co))
        return array_api.reshape(col @ B, out_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        img, weight = node.inputs
        n, h, w, ci = img.cached_data.shape
        kh, kw, ci, co = weight.cached_data.shape
        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        # Conv((n, h', w', co), (kh, kw, co, ci)) = (n, h, w, ci)
        pad = kh - self.padding + (
            (h - kh + 2 * self.padding) % self.stride - self.stride - 1) // 2
        img_grad = conv(out_grad,
                        flip(transpose(weight, (2, 3)), (0, 1)),
                        padding=pad)

        # n, h', w', co -> h', w', n, co
        n, hh, ww, co = out_grad.shape
        out_grad_kernel_view = reshape(out_grad, (n, hh * ww, co))
        out_grad_kernel_view = transpose(out_grad_kernel_view, (0, 1))
        out_grad_kernel_view = reshape(out_grad_kernel_view, (hh, ww, n, co))
        # n, h, w, ci -> ci, h, w, n
        img_transposed = transpose(img, (0, 3))
        # Conv((ci, h, w, n), (h', w', n, co)) = (ci, kh, kw, co)
        pad = (kh - 1 + hh - h) // 2
        weight_grad = conv(img_transposed, out_grad_kernel_view, padding=pad)
        # output: ci, kh, kw, co -> kh, kw, ci, co
        weight_grad = reshape(weight_grad, (ci, kh * kw, co))
        weight_grad = reshape(transpose(weight_grad, (0, 1)), (kh, kw, ci, co))
        return (img_grad, weight_grad)
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

###### Begin my ops
def mean(a, axes, keepdims=False):
    if isinstance(axes, int):
        num = a.shape[axes]
    else:
        ax_set = set(axes)
        num = numpy.prod([v for idx, v in enumerate(a.shape) if idx in ax_set])
    out = divide_scalar(summation(a, axes), num)
    if keepdims:
        out = reshape(out, keepdim_shape(a.shape, axes))
    return out
###### End my ops

_unary_elementwise_ops = set()


_binary_elementwise_ops = set()


_op_names = {}


def register_op(cls, name):
    _op_names[cls] = name


def op_name(op):
    cls = type(op)
    if cls in _op_names:
        return _op_names[cls]
    return str(cls)[len("<class 'needle.ops."):-2]


def register_ewise_unary(cls, name):
    _unary_elementwise_ops.add(cls)
    register_op(cls, name)


def register_ewise_binary(cls, name):
    _binary_elementwise_ops.add(cls)
    register_op(cls, name)


def is_ewise_unary(op):
    return type(op) in _unary_elementwise_ops


def is_ewise_binary(op):
    return type(op) in _binary_elementwise_ops


register_ewise_unary(AddScalar, "add_scalar")
register_ewise_unary(MulScalar, "mul_scalar")
register_ewise_unary(PowerScalar, "pow_scalar")
register_ewise_unary(DivScalar, "div_scalar")
register_ewise_unary(Negate, "neg")
register_ewise_unary(Log, "log")
register_ewise_unary(Exp, "exp")
register_ewise_unary(ReLU, "relu")
register_ewise_unary(Tanh, "tanh")

register_ewise_binary(EWiseAdd, "add")
register_ewise_binary(EWiseMul, "mul")
register_ewise_binary(EWiseDiv, "div")

register_op(Transpose, "transpose")
register_op(Reshape, "reshape")
register_op(BroadcastTo, "broadcast_to")
register_op(Summation, "sum")
register_op(MatMul, "dot")
register_op(LogSumExp, "logsumexp")
register_op(Stack, "stack")
register_op(Split, "split")
register_op(Flip, "flip")
register_op(Dilate, "dilate")
register_op(UnDilate, "undilate")
register_op(Conv, "conv")
