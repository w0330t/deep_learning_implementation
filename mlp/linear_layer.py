import torch
from torch import nn
from typing import Any, Dict, List, Tuple


class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                x: torch.Tensor,
                w: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        """前向传播，执行线性变换。
        $$
        y = xW^T + b
        $$
        Args:
            ctx (Any): 上下文对象，用于保存反向传播需要的变量。
            x (torch.Tensor): 需要预测的特征，形状为 (batch_size, input_dim)
            w (torch.Tensor): 线性变换的权重，形状为 (output_dim, input_dim)
            b (torch.Tensor): 线性变换的偏置，形状为 (output_dim, )

        Returns:
            torch.Tensor: 线性变换后的结果。
        """
        # 在反向传播求导的时候，我们需要用到 X 和 W，所以保存它们
        ctx.save_for_backward(x, w)

        # 执行正向计算
        y = x @ w.T + b
        return y



    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) \
                -> Tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None]:
        """反向传播求梯度

        Args:
            ctx (Any): 上下文对象，用于保存反向传播需要的变量。
            grad_output (torch.Tensor): 上一层返回的梯度的值

        Returns:
            Tuple[torch.Tensor|None, torch.Tensor|None, torch.Tensor|None]:
            返回的梯度顺序必须和 forward 方法的输入参数顺序 (x, w, b) 严格对应，如果不需要计算则返回 None
        """

        # 取出正向传播时保存的张量
        x, w = ctx.saved_tensors

        # 初始化梯度
        grad_x = grad_w = grad_b = None

        # ctx.needs_input_grad[n] 会返回一个布尔值，当它为True才会计算当前梯度。
        # x 是 forward 的第一个参数

        # 1. 计算对 x 的梯度: $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W$

        if ctx.needs_input_grad[0]:
            grad_x = grad_output @ w

        # 2. 计算对 w 的梯度: $\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Y}\right)^T X$

        if ctx.needs_input_grad[1]:
            grad_w = grad_output.T @ x

        # 3. 计算对 b 的梯度: 将 grad_output 沿着 batch 维度 (dim=0) 求和

        # 公式为：$\frac{\partial L}{\partial b} = \sum_{i} \left(\frac{\partial L}{\partial Y}\right)_i$

        if ctx.needs_input_grad[2]:
            grad_b = grad_output.sum(dim=0)

        return grad_x, grad_w, grad_b



class LinearLayer(nn.Module):
    """
    自定义线性层，用于实现矩阵乘法和偏置加法。
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        初始化权重和偏置。

        Args:
            input_dim (int): 输入特征维度。
            output_dim (int): 输出维度。
        """
        super().__init__()
        # 初始化权重和偏置
        self.w = nn.Parameter(torch.randn(output_dim, input_dim))
        self.b = nn.Parameter(torch.randn(output_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行线性变换。
        $$
        y = xW^T + b
        $$
        Args:
            x (torch.Tensor): 形状为 (batch_size, input_dim) 的输入张量。

        Returns:
            torch.Tensor: 形状为 (batch_size, output_dim) 的输出张量。
        """
        # 传统方法
        # y = x @ self.w.T + self.b

        # 硬核方法
        y = LinearFunction.apply(x, self.w, self.b)
        return y


# 固定随机数
torch.manual_seed(28)
l = LinearLayer(3, 2)
x = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float)
print(l.forward(x))