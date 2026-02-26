
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from model.brainmodule import BrainModule

def count_module_params(module):
    if module is None: return 0
    return sum(p.numel() for p in module.parameters())

def main():
    print("🚀 Analyzing New BrainModule Architecture implementation...")
    
    # 1. Model Configuration
    # We use in_channels=271 because sensor_positions_P1.npy has 271 sensors.
    n_subjects = 4
    time_len = 181
    in_channels = 271 
    target_dim=768
    
    model = BrainModule(
        n_subjects=n_subjects,
        n_time_steps=time_len,
        in_channels=in_channels,
        target_dim=target_dim,
        use_clip_head=True,
        use_mse_head=True,
    )
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ==========================================
    # 2. Register Hooks
    # ==========================================
    layer_stats = []

    def get_info_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0: 
                in_shape = list(input[0].shape)
            else: 
                in_shape = "Unknown"
            
            if isinstance(output, torch.Tensor): 
                out_shape = list(output.shape)
            elif isinstance(output, dict):
                out_shape = {k: list(v.shape) for k, v in output.items()}
            else: 
                out_shape = "Unknown"
            
            total_params = count_module_params(module)

            layer_stats.append({
                "Layer Name": name, 
                "Input Shape": str(in_shape), 
                "Output Shape": str(out_shape), 
                "Params": f"{total_params:,}"
            })
        return hook

    hooks = []
    
    # Register hooks for key components
    hooks.append(model.spatial_attention.register_forward_hook(get_info_hook("1. Spatial Attention")))
    hooks.append(model.mixing_layer.register_forward_hook(get_info_hook("1.5. Mixing Layer (1x1)")))
    hooks.append(model.subject_layer.register_forward_hook(get_info_hook("2. Subject Layer")))
    hooks.append(model.backbone.register_forward_hook(get_info_hook("3. Backbone (ConvSeq)")))
    hooks.append(model.feature_projection.register_forward_hook(get_info_hook("4. Feature Proj (->F')")))
    hooks.append(model.temporal_aggregator.register_forward_hook(get_info_hook("5. Temporal Agg")))
    
    if 'clip' in model.heads:
        hooks.append(model.heads['clip'].register_forward_hook(get_info_hook("Head: CLIP")))
    if 'mse' in model.heads:
        hooks.append(model.heads['mse'].register_forward_hook(get_info_hook("Head: MSE")))

    # --- Execution ---
    # Input: (B, 271, T)
    dummy_x = torch.randn(1, 271, time_len).to(device)
    dummy_subj = torch.zeros(1, dtype=torch.long).to(device)

    try:
        with torch.no_grad():
            _ = model(dummy_x, dummy_subj)
        
        total = sum(p.numel() for p in model.parameters())
        print("=" * 110)
        print(f"📊 Total Parameters: {total:,}")
        print("=" * 110)
        
        df = pd.DataFrame(layer_stats)
        header = f"{'Layer Name':<30} | {'Input Shape':<22} | {'Output Shape':<30} | {'Params':>15}"
        print(header)
        print("-" * 110)
        for _, row in df.iterrows():
            print(f"{row['Layer Name']:<30} | {row['Input Shape']:<22} | {row['Output Shape']:<30} | {row['Params']:>15}")
        print("-" * 110)

    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        for h in hooks: h.remove()

if __name__ == "__main__":
    main()