import argparse
import os
import time
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import torch.backends.cudnn as cudnn

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.vision_transformers.model import _build_model, MOATConfig
from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from sam2.modeling.position_encoding import PositionEmbeddingSine

# Reuse dataset loading code from transfermer.py
from transfermer import SAVTorchDataset, create_sam_model


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
        # Forward through backbone (trunk)
        x = self.trunk(sample)
        print(f"MOAT trunk output shapes: {[feat.shape for feat in x]}")
        
        # Forward through neck
        features, pos = self.neck(x)
        print(f"Neck output shapes: {[feat.shape for feat in features]}")
        
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        
        src = features[-1]
        print(f"Final feature shape: {src.shape}")
        
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Distill SAM2 image encoder into MOAT blocks")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml", help="SAM2 model configuration file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    # Set environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
    
    cudnn.benchmark = False

    # Create SAM model
    sam_predictor: SAM2ImagePredictor = create_sam_model(args.model_cfg, args.checkpoint)
    moat_predictor: SAM2ImagePredictor = create_sam_model(args.model_cfg, args.checkpoint)

    # Create MOAT image encoder
    moat_image_encoder: nn.Module = build_moat_image_encoder().cuda()
    moat_predictor.model.image_encoder = moat_image_encoder
    # Ensure both models are in the same precision (float32)
    sam_predictor.model.float()
    moat_image_encoder.float()
    
    # Optimize only the parameters of moat_image_encoder
    optimizer = torch.optim.Adam(moat_image_encoder.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Resume from checkpoint if specified
    start_epoch: int = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            moat_image_encoder.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    wandb.init(
        project="moat_sam_image_encoder_distillation",
        config=vars(args),
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Initialize the SAV dataset
    sav_dir: str = os.path.expanduser("~/mldata/sav_000")  # Update this path if necessary
    dataset: SAVTorchDataset = SAVTorchDataset(sav_dir)
    dataloader: DataLoader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        start_time: float = time.time()
        epoch_loss: float = 0.0
        num_batches: int = 0

        for batch_images in dataloader:            
            optimizer.zero_grad()
            batch_images = [img.numpy() for img in batch_images]
            
            print("Running SAM image encoder...")
            with torch.no_grad():
                sam_predictor.set_image_batch(batch_images)
            sam_features = sam_predictor._features["image_embed"]
            print(f"SAM features shape: {sam_features.shape}")            
            print("Running MOAT image encoder...")
            moat_predictor.set_image_batch(batch_images)
            moat_features = moat_predictor._features["image_embed"]
            print(f"MOAT features shape: {moat_features.shape}")            
            # Compute loss (now the shapes should match)
            loss: torch.Tensor = criterion(moat_features, sam_features)
            
            print(f"Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            batch_duration: float = time.time() - start_time
            wandb.log({"epoch": epoch + 1, "num_batches": num_batches, "loss": loss.item(), "batch_duration": batch_duration})

        avg_loss: float = epoch_loss / num_batches
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        checkpoint: dict = {
            'epoch': epoch + 1,
            'state_dict': moat_image_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"moat_image_encoder_checkpoint_epoch_{epoch+1}.pt"))
    
    print("Training completed.")
    
    final_checkpoint: dict = {
        'epoch': args.epochs,
        'state_dict': moat_image_encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, "trained_moat_image_encoder.pt"))
    print(f"Trained MOAT image encoder saved to {os.path.join(args.save_dir, 'trained_moat_image_encoder.pt')}")

if __name__ == "__main__":
    main()