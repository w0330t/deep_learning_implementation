import torch
from torch import nn
from typing import Any, Dict, List, Tuple


class MLPModel(nn.Module):
    """
    自定义多层感知机 (MLP) 架构。
    旨在定义网络层并处理前向传播。
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], **kwargs: Any) -> None:
        """
        初始化网络层。在此处你可以自由使用 nn.Linear, nn.ReLU, nn.BatchNorm1d 等。

        Args:
            input_dim (int): 输入特征维度。
            output_dim (int): 输出维度（回归通常为 1）。
            hidden_layers (List[int]): 隐藏层神经元数量列表，例如 [128, 64]。
            **kwargs (Any): 接收配置中的自定义模型参数。
        """
        super().__init__()
        # TODO: 构建你的层结构
        pass


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播计算。

        Args:
            x (torch.Tensor): 形状为 (batch_size, input_dim) 的输入张量。

        Returns:
            torch.Tensor: 形状为 (batch_size, output_dim) 的预测张量。
        """
        # TODO: 实现数据流经各层的逻辑
        pass