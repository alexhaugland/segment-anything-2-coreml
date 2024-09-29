import torch
import torch.nn as nn
import coremltools as ct
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.vision_transformers.moat import MOATBlock, MOATBlockConfig
from sam2.modeling.backbones.hieradet import ConvMultiScaleBlock
import argparse
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil.passes.defs.quantization import ComputePrecision
import os
import copy

parser = argparse.ArgumentParser(description="Export SAM2 block and custom block to Core ML")
parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml", help="SAM2 model configuration file")
parser.add_argument("--checkpoint", type=str, default="checkpoints/sam2_hiera_tiny.pt", help="SAM2 checkpoint file")
parser.add_argument("--custom_checkpoint", type=str, default=None, help="Path to trained custom block checkpoint (optional)")
parser.add_argument("--block_type", type=str, default="conv_block", choices=["moat_block", "conv_block"], help="Type of block to export")
parser.add_argument("--iterate_layers", action="store_true", help="Iterate over layers, removing one at a time from the end")
parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the exported models")
args = parser.parse_args()


def export_sam2_block(sam2_model):
    # Extract the first block of the image encoder
    sam2_block = sam2_model.image_encoder.trunk.blocks[0]
    
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

def export_custom_block(block_type, sam2_model, custom_checkpoint_path=None, iterate_layers=False, output_dir="output"):
    original_block = sam2_model.image_encoder.trunk.blocks[0]
    
    if custom_checkpoint_path:
        # Load the trained custom block
        checkpoint = torch.load(custom_checkpoint_path, map_location="cpu")
        config = checkpoint['config']
        if block_type == "moat_block":
            moat_config = MOATBlockConfig(**config)
            block = MOATBlock(moat_config)
        elif block_type == "conv_block":
            block = ConvMultiScaleBlock(**config)
        else:
            raise ValueError(f"Unsupported block type: {block_type}")
        block.load_state_dict(checkpoint['state_dict'])
    else:
        # Use configuration from the SAM2 model
        if block_type == "moat_block":
            config = {
                "block_name": "moat_block",
                "window_size": [4, 4],
                "attn_norm_class": original_block.norm1.__class__,
                "head_dim": 16,
                "activation": nn.GELU(),
                "kernel_size": 3,
                "stride": 1,
                "input_filters": original_block.dim,
                "output_filters": original_block.dim_out,
                "expand_ratio": 1,
                "id_skip": False,
                "se_ratio": None,
                "attention_mode": "local",
                "split_head": True
            }
            moat_config = MOATBlockConfig(**config)
            block = MOATBlock(moat_config)
        elif block_type == "conv_block":
            config = {
                "dim": original_block.dim,
                "dim_out": original_block.dim_out,
                "num_heads": original_block.attn.num_heads,
                "mlp_ratio": 4.0,
                "drop_path": 0.0,
                "norm_layer": nn.LayerNorm,
                "q_stride": original_block.q_stride,
                "act_layer": nn.GELU,
                "window_size": original_block.window_size,
            }
            block = ConvMultiScaleBlock(**config)
        else:
            raise ValueError(f"Unsupported block type: {block_type}")
    
    block.eval()
    
    # Create a dummy input in NCHW format
    dummy_input = torch.randn(1, config["dim" if "dim" in config else "input_filters"], 256, 256)
    
    if iterate_layers:
        os.makedirs(output_dir, exist_ok=True)
        
        # Define a list of components to remove iteratively
        components_to_remove = [
            'mlp',
            'norm2',
            'attn',
            'norm1',
        ]
        
        for i, component in enumerate(components_to_remove):
            # Create a deep copy of the original block
            truncated_block = copy.deepcopy(block)
            
            # Remove components
            for j in range(i + 1):
                setattr(truncated_block, components_to_remove[j], nn.Identity())
            
            # Trace the model
            traced_model = torch.jit.trace(truncated_block, dummy_input)
            
            # Convert to Core ML
            model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
                compute_units=ct.ComputeUnit.ALL,
                minimum_deployment_target=AvailableTarget.iOS18,
                compute_precision=ComputePrecision.FLOAT16,
            )
            
            # Save the model
            model_name = f"{block_type}_removed_{i+1}.mlpackage"
            model_path = os.path.join(output_dir, model_name)
            model.save(model_path)
            print(f"Exported {model_name} to {model_path}")
    else:
        # Trace the model
        traced_model = torch.jit.trace(block, dummy_input)
        
        # Convert to Core ML
        model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=AvailableTarget.iOS18,
            compute_precision=ComputePrecision.FLOAT16,
        )
        
        # Save the model
        model_path = os.path.join(output_dir, f"{block_type}.mlpackage")
        model.save(model_path)
        print(f"{block_type} exported to {model_path}")

def main():
    # Load the SAM2 model
    sam2_model = build_sam2(args.model_cfg, args.checkpoint, device="cpu")
    
    # Export SAM2 block
    export_sam2_block(sam2_model)
    
    # Export custom block
    export_custom_block(args.block_type, sam2_model, args.custom_checkpoint, args.iterate_layers, args.output_dir)

if __name__ == "__main__":
    main()