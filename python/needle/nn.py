"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features,
                                                     out_features,
                                                     device=device,
                                                     dtype=dtype),
                                device=device,
                                dtype=dtype)
        if bias:
            # The homework requires the shape (1, out) instead of (out,)
            self.bias = Parameter(init.kaiming_uniform(out_features,
                                                       1,
                                                       device=device,
                                                       dtype=dtype).reshape(
                                                           (1, out_features)),
                                  device=device,
                                  dtype=dtype)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out = ops.add(out, self.bias)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(1 + ops.exp(-x), -1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = logits.shape[1]
        one_hot = init.one_hot(n, y, device=logits.device, dtype=logits.dtype)
        def sum_keep_dim(x, axes):
            axes_set = set(axes)
            expanded_shape = np.where([i in axes_set for i in range(len(x.shape))],
                                    np.ones_like(x.shape), x.shape)
            return ops.reshape(ops.summation(x, axes=axes), expanded_shape)

        batched = ops.log(
            ops.summation(ops.exp(
                logits -
                sum_keep_dim(ops.multiply(logits, one_hot), axes=(1, ))),
                          axes=1))
        return ops.mean(batched, axes=0)
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype), device=device)
        self.bias = Parameter(init.zeros(dim, dtype=dtype), device=device)
        self.running_mean = init.zeros(dim, dtype=dtype, device=device)
        self.running_var = init.ones(dim, dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        subtract_axis = 0
        if self.training:
            mean = ops.mean(x, axes=subtract_axis)
            # var = (ops.mean(
            #     ops.power_scalar(x, 2), axes=subtract_axis) -
            #     ops.power_scalar(mean, 2))
            var = ops.mean(ops.power_scalar(x - mean, 2),
                           axes=subtract_axis,
                           keepdims=True)
            # update running mean and var
            momentum = self.momentum
            shape = (self.dim,)
            self.running_mean = ops.reshape(
                (1 - momentum) * self.running_mean + momentum * mean,
                shape).detach()
            self.running_var = ops.reshape(
                (1 - momentum) * self.running_var + momentum * var,
                shape).detach()
        else:
            mean = self.running_mean
            var = self.running_var
        return ops.add(
            self.bias,
            ops.multiply(
                self.weight,
                ops.divide(x - mean, ops.power_scalar(var + self.eps, 0.5))))
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype), device=device)
        self.bias = Parameter(init.zeros(dim, dtype=dtype), device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Both with shape (batch_size, 1)
        subtract_axis = 1
        mean = ops.mean(x, axes=subtract_axis, keepdims=True)
        # var = (ops.mean(
        #     ops.power_scalar(x, 2), axes=subtract_axis, keepdims=True) -
        #        ops.power_scalar(mean, 2))
        var = ops.mean(ops.power_scalar(x - mean, 2),
                       axes=subtract_axis,
                       keepdims=True)
        return ops.add(
            self.bias,
            ops.multiply(
                self.weight,
                ops.divide(x - mean, ops.power_scalar(var + self.eps, 0.5))))
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training or self.p == 0:
            return x
        else:
            non_zero = ops.divide_scalar(x, 1 - self.p)
            # Notice that this is not differentiable!
            rn = init.randb(*x.shape,
                            p=1 - self.p,
                            device=x.device,
                            dtype=x.dtype)
            return ops.multiply(non_zero, rn)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        fan_in = in_channels * kernel_size**2
        fan_out = out_channels * kernel_size**2
        self.weight = Parameter(init.kaiming_uniform(
            fan_in,
            fan_out, (kernel_size, kernel_size, in_channels, out_channels),
            device=device,
            dtype=dtype),
                                device=device,
                                dtype=dtype)
        if bias:
            interval = 1.0 / fan_in**0.5
            self.bias = Parameter(
                init.rand(out_channels,
                          low=-interval,
                          high=interval,
                          device=device,
                          dtype=dtype))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        n, c, h, w = x.shape
        assert c == self.in_channels
        x = x.reshape((n, c, h * w)).transpose((1, 2)).reshape((n, h, w, c))
        # get padding value
        if self.stride == 1:
            assert (self.kernel_size - 1) % 2 == 0
        padding  = (self.kernel_size - 1) // 2
        # run conv
        out = ops.conv(x, self.weight, self.stride, padding)
        # add bias
        if self.bias is not None:
            out = ops.add(out, self.bias)
        # transpose
        n, h, w, o = out.shape
        out = out.reshape((n, h * w, o)).transpose((1, 2)).reshape((n, o, h, w))
        return out
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        val = (1 / hidden_size)**0.5
        kwargs = {
            "low": -val,
            "high": val,
            "device": device,
            "dtype": dtype,
            "requires_grad": True
        }
        self.W_ih = Parameter(init.rand(input_size, hidden_size, **kwargs))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, **kwargs))
        self.bias_ih = (Parameter(init.rand(hidden_size, **kwargs))
                     if bias else None)
        self.bias_hh = (Parameter(init.rand(hidden_size, **kwargs))
                     if bias else None)
        if nonlinearity == "relu":
            self.act = ReLU()
        elif nonlinearity == "tanh":
            self.act = Tanh()
        else:
            raise ValueError(f"unsupported nonlinearity: {nonlinearity}")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0],
                           self.hidden_size,
                           device=X.device,
                           dtype=X.dtype)
        ret = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih is not None:
            assert self.bias_hh is not None
            ret += self.bias_ih + self.bias_hh
        return self.act(ret)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = tuple(
            RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias,
                    nonlinearity, device, dtype) for i in range(num_layers))
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        # tuple(seq_len) of (bs, input_size)
        xs = ops.split(X, 0)
        if h0 is None:
            h0 = init.zeros(self.num_layers,
                            bs,
                            self.hidden_size,
                            device=self.device,
                            dtype=self.dtype)
        # tuple(num_layers) of (bs, hidden_size)
        hs = ops.split(h0, 0)
        final_hiddens = []

        # x_t^l = h_t^{l-1}
        # h_t^l = cell(x_t, h_{t-1}^l)
        # xs[t] = x_i^t. Update for each i
        # hs[i] = h_0^i
        # final_hiddens[i] = h_T value for layer i
        for i, cell in enumerate(self.rnn_cells):
            h_t = hs[i]
            new_xs = []
            for j in range(seq_len):
                h_t = cell(xs[j], h_t)
                new_xs.append(h_t)
            final_hiddens.append(h_t)
            xs = new_xs
        output = ops.stack(xs, axis=0)
        final_hiddens = ops.stack(final_hiddens, axis=0)
        return output, final_hiddens
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        val = (1 / hidden_size)**0.5
        kwargs = {
            "low": -val,
            "high": val,
            "device": device,
            "dtype": dtype,
            "requires_grad": True
        }
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, **kwargs))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, **kwargs))
        self.bias_ih = (Parameter(init.rand(4 * hidden_size, **kwargs))
                     if bias else None)
        self.bias_hh = (Parameter(init.rand(4 * hidden_size, **kwargs))
                     if bias else None)
        self.act = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        def reshape_and_split(x):
            new_shape = x.shape[:-1] + (4, x.shape[-1] // 4)
            return ops.split(ops.reshape(x, new_shape), len(x.shape) - 1)
        def linear_may_biased(x, h, w_ih, w_hh, b_ih, b_hh):
            ret = x @ w_ih + h @ w_hh
            if b_ih is not None:
                assert b_hh is not None
                ret += b_ih + b_hh
            return ret
        bs = X.shape[0]

        # init h with zeros if not input
        if h is None:
            kwargs = {"device": X.device, "dtype": X.dtype}
            h0 = init.zeros(bs, self.hidden_size, **kwargs)
            c0 = init.zeros(bs, self.hidden_size, **kwargs)
        else:
            h0, c0 = h
        # split to different weights
        split_W_ih = reshape_and_split(self.W_ih)
        split_W_hh = reshape_and_split(self.W_hh)
        if self.bias_ih is not None:
            assert self.bias_hh is not None
            split_bias_ih = reshape_and_split(self.bias_ih)
            split_bias_hh = reshape_and_split(self.bias_hh)
        else:
            split_bias_ih = split_bias_hh = [None] * 4
        # repeated pattern
        i = self.act(
            linear_may_biased(X, h0, split_W_ih[0], split_W_hh[0],
                              split_bias_ih[0], split_bias_hh[0]))
        f = self.act(
            linear_may_biased(X, h0, split_W_ih[1], split_W_hh[1],
                              split_bias_ih[1], split_bias_hh[1]))
        g = ops.tanh(
            linear_may_biased(X, h0, split_W_ih[2], split_W_hh[2],
                              split_bias_ih[2], split_bias_hh[2]))
        o = self.act(
            linear_may_biased(X, h0, split_W_ih[3], split_W_hh[3],
                              split_bias_ih[3], split_bias_hh[3]))

        c = f * c0 + i * g
        hh = o * ops.tanh(c)
        return hh, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = tuple(
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias,
                     device, dtype) for i in range(num_layers))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h is None:
            kwargs = {"device": self.device, "dtype": self.dtype}
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, **kwargs)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, **kwargs)
        else:
            h0, c0 = h
        # tuple(seq_len) of (bs, input_size)
        xs = ops.split(X, 0)
        # tuple(num_layers) of (bs, hidden_size)
        hs = ops.split(h0, 0)
        cs = ops.split(c0, 0)
        final_h = []
        final_c = []
        for i, cell in enumerate(self.lstm_cells):
            h_t, c_t = hs[i], cs[i]
            new_xs = []
            for j in range(seq_len):
                h_t, c_t = cell(xs[j], (h_t, c_t))
                new_xs.append(h_t)
            final_h.append(h_t)
            final_c.append(c_t)
            xs = new_xs
        output = ops.stack(xs, 0)
        final_h = ops.stack(final_h, 0)
        final_c = ops.stack(final_c, 0)
        return output, (final_h, final_c)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim),
                                device=device,
                                dtype=dtype)
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        return init.one_hot(self.num_embeddings, x, self.device,
                            self.dtype) @ self.weight
        ### END YOUR SOLUTION
