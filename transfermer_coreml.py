import torch
import coremltools as ct
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.vision_transformers.moat import MOATBlock, MOATBlockConfig
import argparse

parser = argparse.ArgumentParser(description="Export SAM2 block and MOAT block to Core ML")
parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml", help="SAM2 model configuration file")
parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint file")
parser.add_argument("--moat_checkpoint", type=str, required=True, help="Path to trained MOAT block checkpoint")
args = parser.parse_args()



def export_sam2_block(model_cfg, checkpoint_path):
    sam2_model = build_sam2(model_cfg, checkpoint_path, device="cpu")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Extract the first block of the image encoder
    sam2_block = predictor.model.image_encoder.trunk.blocks[0]
    
    # note that these are DIFFERENT
    # Create a dummy input (batch size 1, channels 96, height 256, width 256)
    dummy_input = torch.randn(1, 256, 256, 96)
    
    # Trace the model
    traced_model = torch.jit.trace(sam2_block, dummy_input)
    
    # Convert to Core ML
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Save the model
    model.save("sam2_block.mlpackage")
    print("SAM2 block exported to sam2_block.mlpackage")

def export_moat_block(moat_checkpoint_path):
    # Load the trained MOAT block
    checkpoint = torch.load(moat_checkpoint_path, map_location="cpu")
    config = checkpoint['config']
    moat_config = MOATBlockConfig(**config)
    moat_block = MOATBlock(moat_config)
    moat_block.load_state_dict(checkpoint['state_dict'])
    moat_block.eval()
    
    # note that these are DIFFERENT
    # Create a dummy input (batch size 1, channels 96, height 256, width 256)
    dummy_input = torch.randn(1, 96, 256, 256)
    
    # Trace the model
    traced_model = torch.jit.trace(moat_block, dummy_input)
    
    # Convert to Core ML
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Save the model
    model.save("moat_block.mlpackage")
    print("MOAT block exported to moat_block.mlpackage")

def main():
    # Export SAM2 block
    export_sam2_block(args.model_cfg, args.checkpoint)
    
    # Export MOAT block
    export_moat_block(args.moat_checkpoint)

if __name__ == "__main__":
    main()
