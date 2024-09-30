import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.modeling.vision_transformers.model import _build_model
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

class MOATAdapter(nn.Module):
    def __init__(self, moat_config):
        super().__init__()
        self.adapter = nn.Conv2d(moat_config.hidden_size[-1], 256, kernel_size=1)

    def forward(self, x):
        return self.adapter(x)

class CustomFpnNeck(FpnNeck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the convs with new ones that output 256 channels
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1),
                nn.GroupNorm(32, 256)
            )
            for in_channels in kwargs['backbone_channel_list']
        ])

    def forward(self, xs):
        out = []
        for i, x in enumerate(xs):
            out.append(self.convs[i](x))
        
        # Apply top-down pathway
        for i in range(len(out)-1, 0, -1):
            out[i-1] = out[i-1] + F.interpolate(out[i], size=out[i-1].shape[-2:], mode='nearest')
        
        pos = [self.position_encoding(x) for x in out]
        return out, pos

class CustomImageEncoder(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp

    def forward(self, sample: torch.Tensor):
        x = self.trunk(sample)
        features, pos = self.neck(x)
        
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        
        src = features[-1]
        
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output

def build_moat_image_encoder(image_size: int = 1024) -> nn.Module:
    moat_config, moat_trunk = _build_model(
        shape=(1, 3, image_size, image_size),
        base_arch="tiny-moat-2",
        attention_mode="global",
        split_head=True,
        output_stride=32,
    )
    
    # Add channel_list attribute to moat_trunk
    moat_trunk.channel_list = moat_config.hidden_size
    
    # Create FpnNeck
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=128,  # Half of 256
        normalize=True,
        scale=None,
        temperature=10000
    )
    neck = CustomFpnNeck(
        position_encoding=position_encoding,
        d_model=256,
        backbone_channel_list=moat_config.hidden_size,
        fpn_top_down_levels=list(range(len(moat_config.hidden_size))),  # Use all levels
        fpn_interp_model="nearest"
    )
    
    image_encoder = CustomImageEncoder(moat_trunk, neck, scalp=1)
    
    return image_encoder