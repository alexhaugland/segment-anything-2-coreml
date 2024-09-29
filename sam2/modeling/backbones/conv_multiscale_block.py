import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from sam2.modeling.sam2_utils import DropPath, MLP
from sam2.modeling.backbones.utils import window_partition, window_unpartition

class ConvMultiScaleAttention(nn.Module):
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
        # Replace Conv2d with Linear equivalent (1x1 Conv)
        self.qkv = nn.Conv2d(dim, dim_out * 3, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already in NCHW format
        B, C, H, W = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)

        if self.q_pool:
            q = self.q_pool(qkv[:, 0].reshape(B, -1, H, W))
            H, W = q.shape[2:]
            q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        else:
            q = q.reshape(B, self.num_heads, self.head_dim, H * W)

        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, H, W)
        x = self.proj(x)
        return x

def window_partition_nchw(x: torch.Tensor, window_size: int):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, C, H, W].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, C, window_size, window_size].
        (Hp, Wp): padded height and width before partition
    """
    B, C, H, W = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, C, Hp // window_size, window_size, -1)
    windows = x.permute(0, 2, 4, 1, 3).contiguous()
    windows = windows.view(-1, C, window_size, window_size)
    return windows, (Hp, Wp)

def window_unpartition_nchw(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, C, window_size, window_size].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, C, H, W].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    C = windows.shape[1]

    x = windows.view(B, Hp // window_size, Wp // window_size, C, window_size * window_size)
    x = x.permute(0, 3, 1, 2, 4).contiguous()
    x = x.view(B, C, Hp, Wp)

    if Hp > H or Wp > W:
        x = x[:, :, :H, :W].contiguous()
    return x

class ConvMultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        q_stride: Tuple[int, int] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        
        # Change LayerNorm to BatchNorm2d for NCHW format
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim_out)
        
        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = ConvMultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP with Conv2d layers
        self.mlp = nn.Sequential(
            nn.Conv2d(dim_out, int(dim_out * mlp_ratio), kernel_size=1),
            act_layer(),
            nn.Conv2d(int(dim_out * mlp_ratio), dim_out, kernel_size=1),
        )

        if dim != dim_out:
            self.proj = nn.Conv2d(dim, dim_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        shortcut = x
        
        x = self.norm1(x)

        if self.dim != self.dim_out:
            shortcut = self.proj(shortcut)
            if self.pool:
                shortcut = self.pool(shortcut)

        if self.window_size > 0:
            x, (Hp, Wp) = window_partition_nchw(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition_nchw(x, self.window_size, (Hp, Wp), shortcut.shape[2:])

        # Apply drop path and add shortcut
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x