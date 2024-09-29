import argparse
import torch
import numpy as np
from PIL import Image
import os
import torch.backends.cudnn as cudnn
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from moat_hiera import build_moat_image_encoder, MOATWithMerge

def main():
    parser = argparse.ArgumentParser(description="Test SAM2 with MOAT-Hiera image encoder")
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml", help="SAM2 model configuration file")
    parser.add_argument("--sam2_checkpoint", type=str, default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint file")
    parser.add_argument("--moat_checkpoint", type=str, default="checkpoints/moat_image_encoder_checkpoint_epoch_1.pt", help="MOAT-Hiera checkpoint file")
    parser.add_argument("--image_path", type=str, default="./notebooks/images/truck.jpg", help="Path to the input image")
    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '0'
    os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
    
    cudnn.benchmark = False

    if not os.path.exists(args.image_path):
        print(f"Default image not found at {args.image_path}. Please provide a valid image path.")
        return

    # Load SAM2 model
    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Load MOAT-Hiera image encoder
    moat_image_encoder = build_moat_image_encoder().cuda()
    moat_image_encoder.float()
    checkpoint = torch.load(args.moat_checkpoint)
    moat_image_encoder.load_state_dict(checkpoint['state_dict'])

    # Replace SAM2's image encoder with MOAT-Hiera
    predictor.model.image_encoder = moat_image_encoder

    # Load and preprocess the image
    image = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)

    # Set the image in the predictor
    predictor.set_image(image_np)

    # Generate a sample input point
    input_point = np.array([[image_np.shape[1] // 2, image_np.shape[0] // 2]])
    input_label = np.array([1])

    # Run the model
    with torch.no_grad():
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

    print(f"Generated {len(masks)} masks")
    print(f"Scores: {scores}")
    print(f"Logits shape: {logits.shape}")

if __name__ == "__main__":
    main()