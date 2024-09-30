import argparse
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import torch.nn.functional as F

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.backbones.moat_hiera import build_moat_image_encoder
from sam2.modeling.vision_transformers.model import PEType
# Reuse dataset loading code from transfermer.py
from transfermer import SAVTorchDataset, create_sam_model

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def generate_and_log_debug_images(epoch, sam_predictor, moat_predictor, truck_image, truck_points, truck_labels):
    with torch.no_grad():
        sam_predictor.set_image(truck_image)
        sam_masks, sam_scores, _ = sam_predictor.predict(
            point_coords=truck_points,
            point_labels=truck_labels,
            multimask_output=True,
        )
        
        moat_predictor.set_image(truck_image)
        moat_masks, moat_scores, _ = moat_predictor.predict(
            point_coords=truck_points,
            point_labels=truck_labels,
            multimask_output=True,
        )

    # Select the best mask for each model
    sam_best_mask = sam_masks[np.argmax(sam_scores)]
    moat_best_mask = moat_masks[np.argmax(moat_scores)]

    # Create separate figures for SAM and MOAT masks
    fig_sam, ax_sam = plt.subplots(figsize=(10, 10))
    fig_moat, ax_moat = plt.subplots(figsize=(10, 10))

    # Plot SAM mask
    ax_sam.imshow(truck_image)
    show_mask(sam_best_mask, ax_sam)
    show_points(truck_points, truck_labels, ax_sam)
    ax_sam.set_title("SAM Mask")

    # Plot MOAT mask
    ax_moat.imshow(truck_image)
    show_mask(moat_best_mask, ax_moat)
    show_points(truck_points, truck_labels, ax_moat)
    ax_moat.set_title("MOAT Mask")

    # Save the figures and log to wandb
    sam_image_path = f"debug_images/sam_mask_epoch_{epoch+1}.png"
    moat_image_path = f"debug_images/moat_mask_epoch_{epoch+1}.png"
    
    fig_sam.savefig(sam_image_path)
    fig_moat.savefig(moat_image_path)
    
    wandb.log({
        "SAM_mask": wandb.Image(sam_image_path),
        "MOAT_mask": wandb.Image(moat_image_path),
        "epoch": epoch + 1,
    })
    
    plt.close(fig_sam)
    plt.close(fig_moat)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    
    images, masks, points, labels = zip(*batch)
    
    # Stack images (they should all be the same size)
    images = torch.stack(images)
    
    # Pad masks to the maximum number of masks in the batch
    max_masks = max(m.shape[0] for m in masks)
    padded_masks = []
    for mask in masks:
        pad_size = max_masks - mask.shape[0]
        padded_mask = F.pad(mask, (0, 0, 0, 0, 0, pad_size))
        padded_masks.append(padded_mask)
    masks = torch.stack(padded_masks)
    
    # Pad points and labels similarly
    padded_points = []
    padded_labels = []
    for p, l in zip(points, labels):
        pad_size = max_masks - p.shape[0]
        padded_point = F.pad(p, (0, 0, 0, pad_size))
        padded_points.append(padded_point)
        padded_label = F.pad(l, (0, pad_size), value=-1)  # Use -1 as padding value for labels
        padded_labels.append(padded_label)
    points = torch.stack(padded_points)
    labels = torch.stack(padded_labels)
    
    return images, masks, points, labels

def main() -> None:
    parser = argparse.ArgumentParser(description="Distill SAM2 image encoder into MOAT blocks")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml", help="SAM2 model configuration file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    # MOAT hyperparameters
    parser.add_argument("--image_size", type=int, default=1024, help="Input image size")
    parser.add_argument("--base_arch", type=str, default="tiny-moat-2", choices=["tiny-moat-0", "tiny-moat-1", "tiny-moat-2"], help="Base architecture for MOAT")
    parser.add_argument("--attention_mode", type=str, default="global", choices=["global", "local"], help="Attention mode for MOAT")
    parser.add_argument("--split_head", type=bool, default=True, help="Whether to split head in attention")
    parser.add_argument("--output_stride", type=int, default=32, choices=[16, 32], help="Output stride for MOAT")
    parser.add_argument("--num_blocks", nargs='+', type=int, default=[2, 2, 2, 2], help="Number of blocks for each stage in MOAT")
    parser.add_argument("--mbconv_block_expand_ratio", type=int, default=4, help="Expansion ratio for MBConv blocks")
    parser.add_argument("--moat_block_expand_ratio", type=int, default=4, help="Expansion ratio for MOAT blocks")
    
    parser.add_argument("--optimizer", type=str, default="adam", choices=['adam', 'adamw', 'sgd'], help="Optimizer to use")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")

    args = parser.parse_args()

    # Initialize wandb
    run = wandb.init(project="moat_sam_image_encoder_distillation", config=vars(args))

    # Override args with wandb config
    config = wandb.config
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)

    cudnn.benchmark = False
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Create SAM model
    sam_predictor: SAM2ImagePredictor = create_sam_model(args.model_cfg, args.checkpoint)
    sam_predictor.model.image_encoder.float()

    moat_predictor: SAM2ImagePredictor = create_sam_model(args.model_cfg, args.checkpoint)

    # Create MOAT image encoder with new hyperparameters
    moat_image_encoder: nn.Module = build_moat_image_encoder(
        image_size=args.image_size,
        base_arch=args.base_arch,
        attention_mode=args.attention_mode,
        split_head=args.split_head,
        output_stride=args.output_stride,
        num_blocks=args.num_blocks,
        mbconv_block_expand_ratio=args.mbconv_block_expand_ratio,
        moat_block_expand_ratio=args.moat_block_expand_ratio,
        pe_type=PEType.LePE_ADD
    ).cuda()
    moat_image_encoder.float()
    moat_predictor.model.image_encoder = moat_image_encoder
    
    # Set up optimizer based on the chosen type
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(moat_image_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(moat_image_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(moat_image_encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Modify the loss function
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    
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
    
    run_id = wandb.util.generate_id()
    wandb.init(
        project="moat_sam_image_encoder_distillation",
        config=vars(args),
        name=f"run_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        id=run_id,
    )
    
    # Initialize the SAV dataset
    sav_dir: str = os.path.expanduser("~/mldata/sav_000")  # Update this path if necessary
    dataset: SAVTorchDataset = SAVTorchDataset(sav_dir)
    dataloader: DataLoader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, collate_fn=collate_fn)

    os.makedirs(args.save_dir, exist_ok=True)

    # Load the truck image
    truck_image = np.array(Image.open("notebooks/images/truck.jpg"))
    
    # Define point prompts for the truck image
    truck_points = np.array([[[500, 375]]])
    truck_labels = np.array([[1]])

    latest_checkpoint_path = None

    for epoch in range(start_epoch, args.epochs):
        start_time: float = time.time()
        epoch_loss: float = 0.0
        num_batches: int = 0

        for batch in dataloader:
            if batch is None:
                continue

            batch_images, batch_masks, batch_points, batch_labels = batch

            # Convert batch_images from tensor to list of numpy arrays
            batch_images_list = [img.permute(1, 2, 0).cpu().numpy() for img in batch_images]

            batch_masks = batch_masks.cuda()
            batch_points = batch_points.cuda()
            batch_labels = batch_labels.cuda()

            optimizer.zero_grad()
            
            with torch.no_grad():
                sam_predictor.set_image_batch(batch_images_list)
            sam_features = sam_predictor._features["image_embed"]
            
            moat_predictor.set_image_batch(batch_images_list)
            moat_features = moat_predictor._features["image_embed"]
            
            # Calculate embedding loss
            embedding_loss: torch.Tensor = mse_criterion(moat_features, sam_features)

            # Decode masks for MOAT using predict_batch
            with torch.no_grad():
                moat_masks, moat_scores, _ = moat_predictor.predict_batch(
                    point_coords_batch=batch_points.cpu().numpy(),
                    point_labels_batch=batch_labels.cpu().numpy(),
                    multimask_output=True,
                )

            # Convert moat_masks and moat_scores to PyTorch tensors
            moat_masks = torch.as_tensor(np.stack(moat_masks)).float().cuda()
            moat_scores = torch.as_tensor(np.stack(moat_scores)).float().cuda()

            # Get the number of masks returned by the model and in the batch
            num_moat_masks = moat_masks.shape[1]
            num_batch_masks = batch_masks.shape[1]
            batch_size = batch_masks.shape[0]

            # Determine the number of masks to use
            num_masks_to_use = min(num_moat_masks, num_batch_masks)

            # Select the top N masks from batch_masks and MOAT masks
            batch_masks_top_n = batch_masks[:, :num_masks_to_use]
            moat_masks_top_n = moat_masks[:, :num_masks_to_use]

            # Calculate mask loss (comparing MOAT masks with top N ground truth masks)
            mask_loss = bce_criterion(moat_masks_top_n, batch_masks_top_n)

            # Calculate IoU loss
            inter = (batch_masks_top_n * (moat_masks_top_n > 0.5)).sum((2, 3))
            union = batch_masks_top_n.sum((2, 3)) + (moat_masks_top_n > 0.5).sum((2, 3)) - inter
            iou = inter / (union + 1e-6)  # Add small epsilon to avoid division by zero
            iou_loss = torch.abs(moat_scores[:, :num_masks_to_use] - iou).mean()

            # Combine losses
            loss = embedding_loss + mask_loss + 0.05 * iou_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            batch_duration: float = time.time() - start_time
            wandb.log({
                "epoch": epoch + 1,
                "num_batches": num_batches,
                "total_loss": loss.item(),
                "embedding_loss": embedding_loss.item(),
                "mask_loss": mask_loss.item(),
                "iou_loss": iou_loss.item(),
                "batch_duration": batch_duration
            })

        avg_loss: float = epoch_loss / num_batches
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save the latest checkpoint, overwriting the previous one
        latest_checkpoint_path = os.path.join(args.save_dir, f"moat_image_encoder_checkpoint_{run_id}.pt")
        checkpoint: dict = {
            'epoch': epoch + 1,
            'state_dict': moat_image_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, latest_checkpoint_path)
        
        generate_and_log_debug_images(epoch, sam_predictor, moat_predictor, truck_image, truck_points, truck_labels)

    print("Training completed.")

if __name__ == "__main__":
    main()