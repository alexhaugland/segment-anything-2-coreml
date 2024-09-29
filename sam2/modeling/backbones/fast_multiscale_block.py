from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.modeling.sam2_utils import DropPath

# New MLP class for FastMultiScaleBlock
class LinearMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    x = pool(x)
    if norm:
        x = norm(x)
    return x

class FastMultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_pool = q_pool

        self.q_proj = nn.Linear(dim, dim_out, bias=True)
        self.k_proj = nn.Linear(dim, dim_out, bias=True)
        self.v_proj = nn.Linear(dim, dim_out, bias=True)

        self.proj = nn.Linear(dim_out, dim_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)

        if self.q_pool:
            # Q pooling (for downsample at stage changes)
            q = q.transpose(1, 2).view(B, H, W, self.dim_out)
            q = self.q_pool(q.permute(0, 3, 1, 2))
            new_H, new_W = q.shape[-2:]
            q = q.permute(0, 2, 3, 1).view(B, new_H * new_W, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            new_H, new_W = H, W

        # Scaled Dot-Product Attention
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, new_H, new_W, self.dim_out)
        x = self.proj(attn_output)

        return x

class FastMultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.ReLU,
        window_size: int = 0,
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = FastMultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = LinearMLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        shortcut = x  # B, H, W, C

        # Apply norm1
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = self.proj(shortcut)
            if self.pool:
                shortcut = shortcut.permute(0, 3, 1, 2)
                shortcut = self.pool(shortcut)
                shortcut = shortcut.permute(0, 2, 3, 1)

        # Window partition
        if self.window_size > 0:
            # Inline window_partition function
            window_size = self.window_size
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w

            x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
            pad_hw = (Hp, Wp)
        else:
            pad_hw = None

        # Attention
        x = self.attn(x)

        # Reverse window partition
        if self.window_size > 0:
            # Inline window_unpartition function
            if self.q_stride:
                window_size = self.window_size // self.q_stride[0]
            else:
                window_size = self.window_size
            Hp, Wp = pad_hw
            B = x.shape[0] // (Hp * Wp // window_size // window_size)
            x = x.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

            if Hp > H or Wp > W:
                x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x