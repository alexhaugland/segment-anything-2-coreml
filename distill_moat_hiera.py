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

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.backbones.moat_hiera import build_moat_image_encoder

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

    # Create MOAT image encoder
    moat_image_encoder: nn.Module = build_moat_image_encoder().cuda()
    moat_image_encoder.float()
    moat_predictor.model.image_encoder = moat_image_encoder
    
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

    # Load the truck image
    truck_image = np.array(Image.open("notebooks/images/truck.jpg"))
    
    # Define point prompts for the truck image
    truck_points = np.array([[[500, 375]]])
    truck_labels = np.array([[1]])

    for epoch in range(start_epoch, args.epochs):
        start_time: float = time.time()
        epoch_loss: float = 0.0
        num_batches: int = 0

        for batch_images in dataloader:            
            optimizer.zero_grad()
            batch_images = [img.numpy() for img in batch_images]
            
            with torch.no_grad():
                sam_predictor.set_image_batch(batch_images)
            sam_features = sam_predictor._features["image_embed"]
            
            moat_predictor.set_image_batch(batch_images)
            moat_features = moat_predictor._features["image_embed"]
            
            loss: torch.Tensor = criterion(moat_features, sam_features)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            batch_duration: float = time.time() - start_time
            wandb.log({"epoch": epoch + 1, "num_batches": num_batches, "loss": loss.item(), "batch_duration": batch_duration})
            break

        avg_loss: float = epoch_loss / num_batches
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
        checkpoint: dict = {
            'epoch': epoch + 1,
            'state_dict': moat_image_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.save_dir, f"moat_image_encoder_checkpoint_epoch_{epoch+1}.pt"))
        
        # Generate and log debug images
        generate_and_log_debug_images(epoch, sam_predictor, moat_predictor, truck_image, truck_points, truck_labels)

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