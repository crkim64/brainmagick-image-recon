import h5py
import numpy as np
import os
from tqdm import tqdm

def preprocess_autokl(src_path, dst_path):
    print(f"Processing AutoKL: {src_path} -> {dst_path}")
    with h5py.File(src_path, 'r') as f_src:
        features = f_src['features'] # Shape: (N, 4, 64, 64)
        n_samples = features.shape[0]
        
        # We want to flatten the result: (N, 4096)
        # Process in chunks to avoid OOM
        chunk_size = 1000
        
        with h5py.File(dst_path, 'w') as f_dst:
            dset = f_dst.create_dataset('features', (n_samples, 4096), dtype='float32')
            
            for i in tqdm(range(0, n_samples, chunk_size), desc="AutoKL"):
                chunk = features[i:i+chunk_size] # (B, 4, 64, 64)
                # Channel Mean: (B, 64, 64)
                chunk_mean = np.mean(chunk, axis=1)
                # Flatten: (B, 4096)
                chunk_flat = chunk_mean.reshape(chunk_mean.shape[0], -1)
                dset[i:i+chunk_size] = chunk_flat

def preprocess_clip_text(src_path, dst_path):
    print(f"Processing CLIP Text: {src_path} -> {dst_path}")
    with h5py.File(src_path, 'r') as f_src:
        features = f_src['features'] # Shape: (N, 77, 768)
        n_samples = features.shape[0]
        
        # We want (N, 768)
        chunk_size = 1000
        
        with h5py.File(dst_path, 'w') as f_dst:
            dset = f_dst.create_dataset('features', (n_samples, 768), dtype='float32')
            
            for i in tqdm(range(0, n_samples, chunk_size), desc="CLIP Text"):
                chunk = features[i:i+chunk_size] # (B, 77, 768)
                # Mean over tokens (axis 1)
                chunk_mean = np.mean(chunk, axis=1) # (B, 768)
                dset[i:i+chunk_size] = chunk_mean

def preprocess_clip_vision(src_path, dst_path):
    print(f"Processing CLIP Vision: {src_path} -> {dst_path}")
    with h5py.File(src_path, 'r') as f_src:
        features = f_src['features'] # Shape: (N, 257, 768)
        n_samples = features.shape[0]
        
        # We want CLS token (index 0): (N, 768)
        chunk_size = 1000
        
        with h5py.File(dst_path, 'w') as f_dst:
            dset = f_dst.create_dataset('features', (n_samples, 768), dtype='float32')
            
            for i in tqdm(range(0, n_samples, chunk_size), desc="CLIP Vision"):
                chunk = features[i:i+chunk_size] # (B, 257, 768)
                # CLS token is at index 0
                chunk_cls = chunk[:, 0, :] # (B, 768)
                dset[i:i+chunk_size] = chunk_cls

if __name__ == "__main__":
    base_dir = "./data/extracted_features"
    
    # 1. AutoKL
    preprocess_autokl(
        os.path.join(base_dir, "combined_autokl_train.h5"),
        os.path.join(base_dir, "combined_autokl_train_mean.h5")
    )
    
    # 2. CLIP Text
    preprocess_clip_text(
        os.path.join(base_dir, "combined_clip_text_train.h5"),
        os.path.join(base_dir, "combined_clip_text_train_mean.h5")
    )
    
    # 3. CLIP Vision
    preprocess_clip_vision(
        os.path.join(base_dir, "combined_clip_vision_train.h5"),
        os.path.join(base_dir, "combined_clip_vision_train_cls.h5")
    )
