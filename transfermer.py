from datetime import datetime
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.vision_transformers.moat import MOATBlock, MOATBlockConfig
from sam2.modeling.backbones.hieradet import ConvMultiScaleBlock
import numpy as np
import argparse
import wandb
import os
from torch.utils.data import DataLoader
from sav_dataset.utils.sav_utils import SAVDataset
import cv2
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Train a MOAT block to match SAM2 output")
parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml", help="SAM2 model configuration file")
parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint file")
parser.add_argument("--input_layer", type=str, default="image_encoder.trunk.blocks.0", help="Input layer for feature collection")
parser.add_argument("--output_layer", type=str, default="image_encoder.trunk.blocks.1", help="Output layer for feature collection")
parser.add_argument("--block_type", type=str, default="moat_block", choices=["moat_block", "conv_block"], help="Type of block to create")
parser.add_argument("--resume", type=str, default=None, help="Path to saved MOAT block snapshot to resume training")

args = parser.parse_args()

class FeatureCollector(nn.Module):
    def __init__(self, predictor, input_layer, output_layer):
        super().__init__()
        self.predictor = predictor
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.features = {}
        
        for layer in [input_layer, output_layer]:
            layer_parts = layer.split('.')
            module = self.predictor.model
            for part in layer_parts[:-1]:
                module = getattr(module, part)
            layer_module = getattr(module, layer_parts[-1])
            layer_module.register_forward_hook(self.save_features_hook(layer))
    
    def save_features_hook(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = input[0].detach()
        return hook
    
    def collect_features(self, image):
        self.features = {layer: [] for layer in [self.input_layer, self.output_layer]}
        self.predictor.set_image_batch(image)
        return self.features[self.input_layer], self.features[self.output_layer]

def create_sam_model(model_cfg, checkpoint_path):
    sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def create_feature_collector(predictor, input_layer, output_layer):
    feature_collector = FeatureCollector(predictor, input_layer, output_layer)
    return feature_collector

def read_video(video_id, sav_dataset):
    frames, manual_annot, _ = sav_dataset.get_frames_and_annotations(video_id)

    if frames is None or manual_annot is None:
        return None, None, None

    # Randomly select a frame
    frame_idx = np.random.randint(len(frames))
    Img = frames[frame_idx]
    r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scaling factor
    Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    orig_shape = Img.shape

    # Pad the image to 1024x1024
    if Img.shape[0] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([1024 - Img.shape[0], Img.shape[1], 3], dtype=np.uint8)],
            axis=0,
        )
    if Img.shape[1] < 1024:
        Img = np.concatenate(
            [Img, np.zeros([Img.shape[0], 1024 - Img.shape[1], 3], dtype=np.uint8)],
            axis=1,
        )

    return Img

class SAVTorchDataset(Dataset):
    def __init__(self, sav_dir):
        video_ids = [f.split(".")[0] for f in os.listdir(sav_dir) if f.endswith(".mp4")]
        self.video_ids = video_ids
        self.dataset = SAVDataset(sav_dir)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video = read_video(video_id, self.dataset)
        return video

def reduce_precision(module):
    for param in module.parameters():
        param.data = param.data.half()  # Convert to half precision
    return module

def main():
    # Create SAM model
    sam_predictor = create_sam_model(args.model_cfg, args.checkpoint)
    
    # Create feature collector
    feature_collector = create_feature_collector(sam_predictor, args.input_layer, args.output_layer)
    
    # Get the configuration of the original Hiera block
    original_block = sam_predictor.model.image_encoder.trunk.blocks[0]
    
    if args.block_type == "moat_block":
        config = {
            "block_name": args.block_type,
            "window_size": [16, 16],
            "attn_norm_class": original_block.norm1.__class__,
            "head_dim": 16,
            "activation": nn.GELU(),
            "kernel_size": 3,
            "stride": 1,
            "input_filters": original_block.dim,
            "output_filters": original_block.dim_out,
            "expand_ratio": 1,
            "id_skip": True,
            "se_ratio": None,
            "attention_mode": "local",
            "split_head": True
        }
        # Create a MOATBlockConfig
        moat_config = MOATBlockConfig(**config)
        block = MOATBlock(moat_config).cuda()
    elif args.block_type == "conv_block":
        block = ConvMultiScaleBlock(
            dim=original_block.dim,
            dim_out=original_block.dim_out,
            num_heads=original_block.attn.num_heads,
            mlp_ratio=4.0,
            drop_path=0.0,
            norm_layer=original_block.norm1.__class__,
            q_stride=original_block.q_stride,
            act_layer=nn.GELU,
            window_size=original_block.window_size
        ).cuda()
        config = {"block_type": "conv_block"}
    else:
        raise ValueError(f"Unsupported block type: {args.block_type}")
    
    # Resume from snapshot if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            block.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    optimizer = torch.optim.Adam(block.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    wandb.init(
        project="transfermer",
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    # Initialize the SAV dataset
    sav_dir = os.path.expanduser("~/mldata/sav_000")  # Update this path if necessary
    
    dataset = SAVTorchDataset(sav_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_images in dataloader:
            batch_images = [img.numpy() for img in batch_images]
            
            # Collect features from the original model
            original_input, original_output = feature_collector.collect_features(batch_images)
            
            # Pass the same input through our block
            if args.block_type == "moat_block":
                block_output = block(original_input.permute(0, 3, 1, 2))
            else:  # conv_block
                block_output = block(original_input)
            
            # Compute loss
            if args.block_type == "moat_block":
                loss = criterion(block_output, original_output.permute(0, 3, 1, 2))
            else:  # conv_block
                loss = criterion(block_output, original_output)
            
            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            batch_duration = time.time() - start_time
            wandb.log({"num_batches": num_batches, "loss": loss.item(), "batch_duration": batch_duration})


        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every epoch, overwriting the previous one
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': block.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
        }
        torch.save(checkpoint, f"{args.block_type}_checkpoint_latest.pt")
    
    print("Training completed.")
    
    # Save the final trained MOAT block
    final_checkpoint = {
        'epoch': epoch,
        'state_dict': block.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    torch.save(final_checkpoint, f"trained_{args.block_type}.pt")
    print(f"Trained {args.block_type} saved to trained_{args.block_type}.pt")

if __name__ == "__main__":
    main()