import torch
import torch.nn.functional as F
from torch.autograd import Function
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)


class ReLU(ActivationFunction):
    def forward(self, x):
        return F.relu(x)


class TopKReLU(ActivationFunction):
    def __init__(self, k=1000):
        self.k = k

    def forward(self, x):
        k_values, _ = torch.topk(x, k=self.k, sorted=False)
        x_threshold = k_values.min(dim=-1, keepdim=True)[0]
        output = torch.where(x < x_threshold, torch.tensor(0.0, device=x.device), x)
        output = F.relu(output)
        return output


class RectangleFunction(Function):
    @staticmethod
    def forward(ctx, x):
        output = ((x > -0.5) & (x < 0.5)).to(x.dtype)
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = torch.zeros_like(x)
        return grad_input


class JumpReLUFunction(Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        out = x * (x > threshold).to(x.dtype)
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        x_grad = (x > threshold).to(x.dtype) * grad_output

        rectangle = RectangleFunction.apply
        threshold_grad = (
            - (threshold / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output
        )

        return x_grad, threshold_grad, None


class JumpReLU(ActivationFunction):
    def __init__(self):
        self.bandwidth = 0.001
        self.jumprelu_function = JumpReLUFunction.apply

    def forward(self, x, theta):
        out = self.jumprelu_function(x, theta, self.bandwidth)
        return out

    def __call__(self, x, theta):
        return self.forward(x, theta)
