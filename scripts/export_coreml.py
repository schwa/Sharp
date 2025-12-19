#!/usr/bin/env python3
"""
Export SHARP PyTorch model to CoreML format.
"""

import sys
from pathlib import Path

# Add ml-sharp to path
ml_sharp_path = Path(__file__).parent.parent.parent / "ml-sharp" / "src"
sys.path.insert(0, str(ml_sharp_path))

import torch
import torch.nn as nn
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import numpy as np

def main():
    checkpoint_path = Path(__file__).parent.parent / "Models" / "sharp_2572gikvuh.pt"
    output_path = Path(__file__).parent.parent / "Models" / "SharpPredictor.mlpackage"
    
    print(f"🔮 SHARP CoreML Export")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_path}")
    print()
    
    from sharp.models import PredictorParams, create_predictor
    
    print("Loading PyTorch model...")
    device = "cpu"
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = create_predictor(PredictorParams())
    model.load_state_dict(state_dict)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters")
    
    # Create wrapper that returns flat tensors
    class SharpWrapper(nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor
            
        def forward(self, image, disparity_factor):
            gaussians = self.predictor(image, disparity_factor)
            # Return flat tensors instead of NamedTuple
            return (
                gaussians.mean_vectors,
                gaussians.singular_values, 
                gaussians.quaternions,
                gaussians.colors,
                gaussians.opacities,
            )
    
    wrapper = SharpWrapper(model)
    wrapper.eval()
    
    # Example inputs
    print("Creating example inputs...")
    example_image = torch.randn(1, 3, 1536, 1536)
    example_disparity = torch.tensor([1.0])
    
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_image, example_disparity))
    
    print("Converting to CoreML...")
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="image", shape=(1, 3, 1536, 1536)),
                ct.TensorType(name="disparity_factor", shape=(1,)),
            ],
            outputs=[
                ct.TensorType(name="mean_vectors"),
                ct.TensorType(name="singular_values"),
                ct.TensorType(name="quaternions"),
                ct.TensorType(name="colors"),
                ct.TensorType(name="opacities"),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
        )
        
        print(f"Saving to {output_path}...")
        mlmodel.save(str(output_path))
        print(f"✅ Success! CoreML model saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ CoreML conversion failed: {e}")
        print()
        print("Trying with more permissive settings...")
        
        try:
            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="image", shape=(1, 3, 1536, 1536)),
                    ct.TensorType(name="disparity_factor", shape=(1,)),
                ],
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.macOS14,
                compute_precision=ct.precision.FLOAT32,
            )
            mlmodel.save(str(output_path))
            print(f"✅ Success! CoreML model saved to: {output_path}")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
