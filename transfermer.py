from datetime import datetime
import torch
import torch.nn as nn
import time
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.vision_transformers.moat import MOATBlock, MOATBlockConfig
import numpy as np
import argparse
import wandb

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

def create_feature_collector(model_cfg, checkpoint_path, input_layer, output_layer):
    sam2_model = build_sam2(model_cfg, checkpoint_path, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    feature_collector = FeatureCollector(predictor, input_layer, output_layer)
    return feature_collector

def main():
    parser = argparse.ArgumentParser(description="Train a MOAT block to match SAM2 output")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    args = parser.parse_args()

    model_cfg = "sam2_hiera_t.yaml"
    sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
    input_layer = 'image_encoder.trunk.blocks.0'
    output_layer = 'image_encoder.trunk.blocks.1'
    
    feature_collector = create_feature_collector(model_cfg, sam2_checkpoint, input_layer, output_layer)
    
    # Get the configuration of the original Hiera block
    original_block = feature_collector.predictor.model.image_encoder.trunk.blocks[0]
    
    config = {
        "block_name": "moat_block",
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
        "attention_mode": "local", # somehow this adds a list of size window * window
        "split_head": True
    }
    # Create a MOATBlockConfig
    moat_config = MOATBlockConfig(
        **config
    )
    
    # Instantiate a new MOAT block with the configuration
    moat_block = MOATBlock(moat_config).cuda()
    
    optimizer = torch.optim.Adam(moat_block.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    

    wandb.init(
        project="transfermer",
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    for epoch in range(args.epochs):
        start_time = time.time()
        # Generate a batch of random images
        batch_size = 2
        dummy_images = [np.random.rand(1024, 1024, 3).astype(np.float32) for _ in range(batch_size)]
        
        # Collect features from the original model
        original_input, original_output = feature_collector.collect_features(dummy_images)
        
        # Pass the same input through our MOAT block
        moat_output = moat_block(original_input.permute(0, 3, 1, 2))
        
        # Compute loss
        loss = criterion(moat_output, original_output.permute(0, 3, 1, 2))
        loss_numeric = loss.item()

        
        
        # Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss_numeric:.4f}")
        
        epoch_duration = time.time() - start_time
        wandb.log({"epoch": epoch, "loss": loss_numeric, "epoch_duration": epoch_duration})
    
    print("Training completed.")
    
    # Save the trained MOAT block
    torch.save(moat_block.state_dict(), "trained_moat_block.pt")
    print("Trained MOAT block saved to trained_moat_block.pt")

if __name__ == "__main__":
    main()
