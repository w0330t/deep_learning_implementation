import torch
from torch import nn
from typing import Any, Dict, List, Tuple


class ReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        r"""ReLU的前向传播，执行激活函数。公式为：
        $$
        y = \max(0, x)
        $$
        Args:
            ctx (Any): 上下文对象。
            x (torch.Tensor): 需要预测的特征

        Returns:
            torch.Tensor: 经过激活函数后的结果
        """

        # 保存输入张量，以便在反向传播中使用
        ctx.save_for_backward(x)

        return torch.maximum(x, torch.zeros_like(x))


    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        r"""ReLU的反向传播，计算梯度。
        当x大于0时，梯度为1，否则为0。
        公式为：
        $$
        \frac{\partial y}{\partial x} = \begin{cases}
            1, & \text{if } x > 0 \\
            0, & \text{otherwise}
        \end{cases}
        $$
        Args:
            ctx (Any): 上下文对象。
            grad_output (torch.Tensor): 反向传播的梯度。

        Returns:
            torch.Tensor: 反向传播的梯度。
        """


        x = ctx.saved_tensors[0]
        mask = x > 0

        # 计算梯度，当x大于0时，梯度为1，否则为0
        return grad_output * mask.float()


class ReLULayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ReLUFunction.apply(x)


l = ReLULayer()
# requires_grad=True 以便后续测试反向传播
x = torch.tensor([[1, 1, 1], [2, -1, 2]], dtype=torch.float, requires_grad=True)

# 前向传播
out = l.forward(x)
print("Forward output:\n", out)

# 反向传播测试
out.sum().backward()
print("Gradients of x:\n", x.grad)