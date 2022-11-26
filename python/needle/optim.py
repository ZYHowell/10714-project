"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = []
        self.weight_decay = weight_decay

        for param in params:
            self.u.append(ndl.init.zeros_like(param))

    def step(self):
        ### BEGIN YOUR SOLUTION
        for idx in range(len(self.params)):
            param: ndl.nn.Parameter = self.params[idx]
            u = self.u[idx]
            grad = param.grad
            grad = grad + self.weight_decay * param
            new_u = (self.momentum * u + (1 - self.momentum) * grad).detach()
            new_param = (param - self.lr * new_u).detach()
            param.cached_data = new_param.cached_data
            self.u[idx] = new_u
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = []
        self.v = []

        for param in params:
            self.m.append(ndl.init.zeros_like(param))
            self.v.append(ndl.init.zeros_like(param))

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for idx in range(len(self.params)):
            param = self.params[idx]
            m = self.m[idx]
            v = self.v[idx]
            grad = param.grad
            grad = grad + self.weight_decay * param
            new_m = (self.beta1 * m + (1 - self.beta1) * grad).detach()
            new_v = (self.beta2 * v + (1 - self.beta2) * grad**2).detach()

            bias_cor_m = new_m / (1 - self.beta1**self.t)
            bias_cor_v = new_v / (1 - self.beta2**self.t)
            new_param = (param - self.lr * bias_cor_m /
                         (bias_cor_v**0.5 + self.eps)).detach()

            param.cached_data = new_param.cached_data
            self.m[idx] = new_m
            self.v[idx] = new_v
        ### END YOUR SOLUTION
