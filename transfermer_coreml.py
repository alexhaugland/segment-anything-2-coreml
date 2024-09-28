import torch
import torch.nn as nn
import coremltools as ct
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.vision_transformers.moat import MOATBlock, MOATBlockConfig
import argparse
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil.passes.defs.quantization import ComputePrecision


parser = argparse.ArgumentParser(description="Export SAM2 block and MOAT block to Core ML")
parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml", help="SAM2 model configuration file")
parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint file")
parser.add_argument("--moat_checkpoint", type=str, default=None, help="Path to trained MOAT block checkpoint (optional)")
args = parser.parse_args()



def export_sam2_block(model_cfg, checkpoint_path):
    sam2_model = build_sam2(model_cfg, checkpoint_path, device="cpu")
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Extract the first block of the image encoder
    sam2_block = predictor.model.image_encoder.trunk.blocks[0]
    
    # note that these are DIFFERENT
    # Create a dummy input (batch size 1, height 256, width 256, channels 96)
    dummy_input = torch.randn(1, 256, 256, 96)
    
    # Trace the model
    traced_model = torch.jit.trace(sam2_block, dummy_input)
    
    # Convert to Core ML
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=AvailableTarget.iOS18,
        compute_precision=ComputePrecision.FLOAT16,
    )
    
    # Save the model
    model.save("sam2_block.mlpackage")
    print("SAM2 block exported to sam2_block.mlpackage")

def export_moat_block(moat_checkpoint_path=None):
    if moat_checkpoint_path:
        # Load the trained MOAT block
        checkpoint = torch.load(moat_checkpoint_path, map_location="cpu")
        config = checkpoint['config']
        moat_config = MOATBlockConfig(**config)
        moat_block = MOATBlock(moat_config)
        moat_block.load_state_dict(checkpoint['state_dict'])
    else:
        # Use default configuration from transfermer.py
        cfg = compose(config_name=args.model_cfg)
        OmegaConf.resolve(cfg)
        
        original_block_dim = cfg.model.image_encoder.trunk.embed_dim
        original_block_dim_out = cfg.model.image_encoder.trunk.embed_dim
        
        config = {
            "block_name": "moat_block",
            "window_size": [4, 4],
            "attn_norm_class": nn.LayerNorm,
            "head_dim": 16,
            "activation": nn.GELU(),
            "kernel_size": 3,
            "stride": 1,
            "input_filters": original_block_dim,
            "output_filters": original_block_dim_out,
            "expand_ratio": 1,
            "id_skip": False,
            "se_ratio": None,
            "attention_mode": "local",
            "split_head": True
        }
        moat_config = MOATBlockConfig(**config)
        moat_block = MOATBlock(moat_config)
    
    moat_block.eval()
    
    # Create a dummy input (batch size 1, channels 96, height 256, width 256)
    dummy_input = torch.randn(1, moat_config.input_filters, 256, 256)
    
    # Trace the model
    traced_model = torch.jit.trace(moat_block, dummy_input)
    
    # Convert to Core ML
    model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=AvailableTarget.iOS18,
        compute_precision=ComputePrecision.FLOAT16,
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
