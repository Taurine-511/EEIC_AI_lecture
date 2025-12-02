import torch
import torch.nn as nn


class DQNModel(nn.Module):
    """
    1次元ベクトルからQ値を返すQ関数モデル。
    Args:
        in_size: 入力される次元数 (例えば、EazyMazeの場合はx, y座標の2つ)
        out_size: 出力される次元数（actionの数 例えば、EazyMazeの場合は上下左右の4つ）
    """

    def __init__(self, in_size, out_size):
        """
        初期値を代入。
        """
        unit_sizes = [in_size, 5, 5, 5, 5]
        super(DQNModel, self).__init__()
        self.l_1 = nn.Linear(unit_sizes[0], unit_sizes[1])
        self.l_2 = nn.Linear(unit_sizes[1], unit_sizes[2])
        self.l_3 = nn.Linear(unit_sizes[2], unit_sizes[3])
        self.l_out = nn.Linear(unit_sizes[2], out_size)
        self.activation_func = torch.nn.functional.silu

    def forward(self, x):
        """
        順伝搬の処理を行う
        """
        h = x
        h = self.l_1(h)
        h = self.activation_func(h)
        h = self.l_2(h)
        h = self.activation_func(h)
        h = self.l_3(h)
        h = self.activation_func(h)
        h = self.l_out(h)
        return h
