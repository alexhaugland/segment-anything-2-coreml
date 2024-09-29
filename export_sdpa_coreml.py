import torch
import torch.nn.functional as F
import coremltools as ct
import argparse


class SDPA(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

def export_sdpa_to_coreml(batch_size, num_heads, seq_length, head_dim, output_path):
    # Create dummy inputs
    q = torch.randn(batch_size, num_heads, seq_length, head_dim)
    k = torch.randn(batch_size, num_heads, seq_length, head_dim)
    v = torch.randn(batch_size, num_heads, seq_length, head_dim)

    # Trace the function
    traced_model = torch.jit.trace(SDPA(), (q, k, v))

    # Convert to Core ML
    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[
            ct.TensorType(name="query", shape=q.shape),
            ct.TensorType(name="key", shape=k.shape),
            ct.TensorType(name="value", shape=v.shape)
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS18,
    )

    # Save the model
    mlmodel.save(output_path)
    print(f"Core ML model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SDPA to Core ML")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--head_dim", type=int, default=48, help="Dimension per head")
    parser.add_argument("--output", type=str, default="sdpa_model.mlpackage", help="Output file path")

    args = parser.parse_args()

    export_sdpa_to_coreml(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        seq_length=args.seq_length,
        head_dim=args.head_dim,
        output_path=args.output
    )