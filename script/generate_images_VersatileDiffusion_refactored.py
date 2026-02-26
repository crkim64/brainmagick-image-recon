import sys
import os
import argparse
import numpy as np
import torch
import h5py
import PIL
from PIL import Image
from torchvision import transforms as tvtrans

# Versatile Diffusion setup
sys.path.append('versatile_diffusion')

# Brain Module Setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD

# ==========================================
# 1. Setup & Config
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1', help="Subject Name")
parser.add_argument("--diff_strength", type=float, default=0.75, help="Denoising Strength")
parser.add_argument("--scale", type=float, default=7.5, help="Guidance Scale")
parser.add_argument("--mixing", type=float, default=0.4, help="Mixing Ratio (0.0=TextOnly, 1.0=VisOnly)")
parser.add_argument("--device", default='cuda:0', help="Main Device")
parser.add_argument("--vd_device", default='cuda:0', help="Device for Versatile Diffusion")
parser.add_argument("--bm_device", default='cuda:0', help="Device for Brain Modules")
parser.add_argument("--target_normalization", type=lambda x: (str(x).lower() == 'true'), default=True, help="Apply denormalization to predicted features")
args = parser.parse_args()

# Paths
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
POS_DIR = os.path.join(DATA_DIR, 'sensor_positions')
OUT_DIR = f'results/image_generation/{args.subject}_all_refactored'
os.makedirs(OUT_DIR, exist_ok=True)

# Checkpoints (Full Sequence / Full Latent Models)
CKPT_AUTOKL = './checkpoints/autokl_hybrid/best_model.pth' 
CKPT_VISION = './checkpoints/clip_vision_hybrid/best_model.pth'  
CKPT_TEXT   = './checkpoints/clip_text_hybrid/best_model.pth'     

# Statistics Paths (New)
STATS_AUTOKL = './checkpoints/autokl_hybrid/autokl_stats.npz'
STATS_VISION = './checkpoints/clip_vision_hybrid/clip_vision_stats.npz'
STATS_TEXT   = './checkpoints/clip_text_hybrid/clip_text_stats.npz'

# ==========================================
# 2. Loading Utilities
# ==========================================
print(f"📚 Loading Image Dictionary...")
IMAGE_MAP = {}
if os.path.exists(DICT_PATH):
    with h5py.File(DICT_PATH, 'r') as f:
        cats = f['category_nr'][:]
        exs = f['exemplar_nr'][:]
        paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
        for c, e, p in zip(cats, exs, paths):
            IMAGE_MAP[(c, e)] = p

# Load Statistics for Denormalization
if args.target_normalization:
    print(f"📊 Loading Statistics for Denormalization (Stats files found? {os.path.exists(STATS_AUTOKL)})")
    try:
        stats_autokl = np.load(STATS_AUTOKL)
        stats_vision = np.load(STATS_VISION)
        stats_text = np.load(STATS_TEXT)
        
        # Prepare for broadcasting on bm_device
        mean_autokl = torch.from_numpy(stats_autokl['mean']).float().to(args.bm_device).reshape(1, 1, 64, 64)
        std_autokl = torch.from_numpy(stats_autokl['std']).float().to(args.bm_device).reshape(1, 1, 64, 64)
        
        mean_vision = torch.from_numpy(stats_vision['mean']).float().to(args.bm_device).reshape(1, 768)
        std_vision = torch.from_numpy(stats_vision['std']).float().to(args.bm_device).reshape(1, 768)
        
        mean_text = torch.from_numpy(stats_text['mean']).float().to(args.bm_device).reshape(1, 768)
        std_text = torch.from_numpy(stats_text['std']).float().to(args.bm_device).reshape(1, 768)
        
        print("   ✅ Stats Loaded and moved to device.")
    except Exception as e:
        print(f"   ⚠️ Failed to load stats: {e}")
        print("   ⚠️ Proceeding without denormalization (might produce garbage results if model was trained with normalization).")
        args.target_normalization = False
else:
    print("📊 Target Normalization Disabled (Using Raw Predictions)")
# Stats loading removed as per user request (predicting raw features)

def load_brain_module(ckpt_path, out_dim_clip, out_dim_mse, device):
    print(f"Loading BrainModule from {ckpt_path}...")
    
    # Updated to match model/brainmodule.py signature
    # Note: target_dim applies to both heads in current implementation.
    # We assume out_dim_clip == out_dim_mse or use one of them.
    # For AutoKL: 4096. For CLIP: 768.
    
    model = BrainModule(
        n_time_steps=180,      # Matches the exact downsampled dimension (120Hz)
        in_channels=271,       # MEG Channels
        n_subjects=4,
        spatial_out=270,
        backbone_channels=[270, 320, 320, 320, 320],
        feature_dim=2048,
        target_dim=out_dim_mse, # Use MSE dim as primary target
        use_clip_head=True,
        use_mse_head=True,
    ).to(device)

    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

# BatchObject removed (handled inside BrainModule)

# ==========================================
# 3. Model Initialization
# ==========================================
print("🚀 Loading Versatile Diffusion...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)

net.clip.to(args.vd_device).half()
net.autokl.to(args.vd_device).half()
net.model.to(args.vd_device).half()
if hasattr(net.model, 'diffusion_model'):
    net.model.diffusion_model.device = args.vd_device
net.model.device = args.vd_device
sampler = DDIMSampler_VD(net)

print("🚀 Loading Brain Modules...")
# Load Full Models
# AutoKL: Checkpoint (Hybrid) has head_clip=4096, head_mse=4096 (Channel Mean)
bm_autokl = load_brain_module(CKPT_AUTOKL, out_dim_clip=4096, out_dim_mse=4096, device=args.bm_device)
bm_vision = load_brain_module(CKPT_VISION, out_dim_clip=768, out_dim_mse=768, device=args.bm_device)
bm_text = load_brain_module(CKPT_TEXT, out_dim_clip=768, out_dim_mse=768, device=args.bm_device)

# Stats loading skipped
print("📊 Preprocessing: RAW Predictions (No Denormalization)")

# ==========================================
# 4. Refactoring Functions
# ==========================================
def refactor_autokl(z):
    # z: (1, 1, 64, 64) -> Already Mean
    # Repeat to (1, 4, 64, 64)
    z_new = z.repeat(1, 4, 1, 1)
    return z_new

def refactor_clip_vision(c):
    # c: (1, 768) or (1, 1, 768) -> Predicted CLS Token
    # Goal: (1, 257, 768) with CLS at idx 0, others 0
    if c.dim() == 2:
        c = c.unsqueeze(1) # (1, 1, 768)
    
    # Create empty tensor (1, 257, 768)
    c_new = torch.zeros(c.shape[0], 257, 768).to(c.device).type(c.dtype)
    c_new[:, 0:1, :] = c
    return c_new

def refactor_clip_text(c):
    # c: (1, 768) or (1, 1, 768) -> Predicted Mean Token
    # Goal: (1, 77, 768) replicated
    if c.dim() == 2:
        c = c.unsqueeze(1) # (1, 1, 768)
        
    c_new = c.repeat(1, 77, 1) # (1, 77, 768)
    return c_new

# ==========================================
# ==========================================
# 5. Generation Loop
# ==========================================
# Sensor positions are loaded internally by BrainModule/ChannelMerger (via common.py)
# so we don't need to load them here.

sub_map = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3}
sub_idx = torch.tensor([sub_map[args.subject]]).long().to(args.bm_device)

print(f"📂 Loading Test Data: {TEST_MEG_PATH}")
with h5py.File(TEST_MEG_PATH, 'r') as f:
    meg_data = f['meg'][:]      
    test_cats = f['category_nr'][:]
    test_exs = f['exemplar_nr'][:]
    
print(f"✨ Start Generation for 10 Unique Images (Averaged over trials)...")

# Group indices by (category_nr, exemplar_nr)
from collections import defaultdict
grouped_indices = defaultdict(list)
for idx in range(len(meg_data)):
    cat_id = test_cats[idx]
    ex_id = test_exs[idx]
    grouped_indices[(cat_id, ex_id)].append(idx)

# Select first 50 unique images
unique_keys = list(grouped_indices.keys())
selected_keys = unique_keys[:50]

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    for i, key in enumerate(selected_keys):
        indices = grouped_indices[key]
        print(f"[{i+1}/50] Processing Image Key {key}: Found {len(indices)} repeated trials.")
        
        # Lists to store predictions for averaging
        preds_autokl = []
        preds_vision = []
        preds_text = []
        
        # --- 1. Predict for each trial ---
        for idx in indices:
            # Load MEG
            meg_tensor = torch.from_numpy(meg_data[idx]).unsqueeze(0).float().to(args.bm_device)
            # BatchObject not needed for updated BrainModule (internal mock)
            
            # (A) AutoKL (Predicted is Raw)
            # Forward: model(x, sub_idx) -> {'clip': ..., 'mse': ...}
            out_autokl = bm_autokl(meg_tensor, sub_idx)
            pred_latent_flat = out_autokl['mse'] # Use MSE head output (Raw Feature)
            pred_latent = pred_latent_flat.reshape(1, 1, 64, 64) 
            preds_autokl.append(pred_latent)
            
            # (B) CLIP Vision (Predicted is Raw)
            out_vision = bm_vision(meg_tensor, sub_idx)
            pred_vision_flat = out_vision['mse'] # Use MSE head output
            pred_vision = pred_vision_flat.reshape(1, 768)
            preds_vision.append(pred_vision)
            
            # (C) CLIP Text (Predicted is Raw)
            out_text = bm_text(meg_tensor, sub_idx)
            pred_text_flat = out_text['mse'] # Use MSE head output
            pred_text = pred_text_flat.reshape(1, 768)
            preds_text.append(pred_text)
            
        # --- 2. Average Predictions ---
        # Stack and Mean over 0-th dimension. 
        # For AutoKL: (N, 1, 1, 64, 64) -> (1, 1, 64, 64)
        avg_autokl = torch.stack(preds_autokl).mean(dim=0)
        
        # For Vision/Text: (N, 1, 768) -> (1, 768)
        avg_vision = torch.stack(preds_vision).mean(dim=0)
        avg_text = torch.stack(preds_text).mean(dim=0)
        
        # --- [NEW] Denormalization Logic & Variance Rescaling ---
        if args.target_normalization:
            # 1. Z-Score the model's prediction to recover lost variance
            # AutoKL:
            avg_a_mean = avg_autokl.mean()
            avg_a_std = avg_autokl.std()
            avg_autokl = (avg_autokl - avg_a_mean) / (avg_a_std + 1e-6)
            
            # Vision:
            avg_v_mean = avg_vision.mean()
            avg_v_std = avg_vision.std()
            avg_vision = (avg_vision - avg_v_mean) / (avg_v_std + 1e-6)
            
            # Text:
            avg_t_mean = avg_text.mean()
            avg_t_std = avg_text.std()
            avg_text = (avg_text - avg_t_mean) / (avg_t_std + 1e-6)

            # 2. Denormalize using Dataset Statistics
            avg_autokl = avg_autokl * (std_autokl + 1e-6) + mean_autokl
            avg_vision = avg_vision * (std_vision + 1e-6) + mean_vision
            avg_text = avg_text * (std_text + 1e-6) + mean_text
        
        # --- 3. Refactor Averaged Features ---
        
        # (A) AutoKL Refactor
        # No Normalization (User Request)
        init_latent = refactor_autokl(avg_autokl).to(args.vd_device)

        # (B) CLIP Vision Refactor
        # Input (1, 768) -> Output (1, 257, 768)
        cond_vision = refactor_clip_vision(avg_vision).to(args.vd_device)

        # (C) CLIP Text Refactor
        # Input (1, 768) -> Output (1, 77, 768)
        cond_text = refactor_clip_text(avg_text).to(args.vd_device)
        
        # --- 4. Prepare Paths (using first index info) ---
        first_idx = indices[0]
        cat_id = test_cats[first_idx]
        ex_id = test_exs[first_idx]
        
        original_relative_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
        # Fix path if it starts with 'data/' to avoid duplication with DATA_DIR
        clean_path = original_relative_path.replace('data/', '', 1) if original_relative_path.startswith('data/') else original_relative_path
        full_gt_path = os.path.join(DATA_DIR, clean_path)
        
        base_name = os.path.splitext(os.path.basename(original_relative_path))[0]
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])

        # Update the naming conventions as requested
        # Format: [object_name]_[index]s_avg[N]_gen.png or ..._gt.png
        
        save_filename = f"{safe_name}_{i+1:02d}s_avg{len(indices)}_gen.png"
        save_path = os.path.join(OUT_DIR, save_filename)
        
        # [Mod] Save GT Image
        gt_filename = f"{safe_name}_{i+1:02d}s_avg{len(indices)}_gt.png"
        gt_save_path = os.path.join(OUT_DIR, gt_filename)
        
        if os.path.exists(full_gt_path):
            try:
                gt_img = Image.open(full_gt_path).convert('RGB')
                gt_img.save(gt_save_path)
            except Exception as e:
                print(f"[Warning] Failed to save GT image {full_gt_path}: {e}")
        else:
            # Try alternative path logic for test set
            fname = os.path.basename(clean_path)
            parts = fname.split('_')
            if len(parts) > 1:
                category = "_".join(parts[:-1])
                alt_path = os.path.join(DATA_DIR, 'images_meg', category, fname)
            else:
                alt_path = os.path.join(DATA_DIR, 'images_meg', fname) # Fallback

            if os.path.exists(alt_path):
                try:
                    gt_img = Image.open(alt_path).convert('RGB')
                    gt_img.save(gt_save_path)
                except:
                    print(f"[Warning] GT Image found but failed to load at {alt_path}")
            else:
                 print(f"[Warning] GT Image not found at {full_gt_path} or {alt_path}")



        # 3. VD Sampling
        dummy_text = ''
        utx = net.clip_encode_text(dummy_text).to(args.vd_device).half()
        dummy_img = torch.zeros((1, 3, 224, 224)).to(args.vd_device).half()
        uim = net.clip_encode_vision(dummy_img).to(args.vd_device).half()

        sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
        t_enc = int(args.diff_strength * 50)
        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(args.vd_device))
        
        z = sampler.decode_dc(
            x_latent=z_enc,
            first_conditioning=[uim, cond_vision], 
            second_conditioning=[utx, cond_text],  
            t_start=t_enc,
            unconditional_guidance_scale=args.scale,
            xtype='image',
            first_ctype='vision',
            second_ctype='prompt',
            mixed_ratio=(1 - args.mixing) 
        )
        
        z = z.float()
        net.autokl.to(args.vd_device).float() 
        x = net.autokl_decode(z)
        
        if i == 0:
            print(f"[DEBUG] Latent Mean: {init_latent.mean().item():.4f}")
            print(f"[DEBUG] Vision Mean: {cond_vision.mean().item():.4f} (Non-zero elements)")
        
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        img = tvtrans.ToPILImage()(x[0].cpu())
        img.save(save_path)
        
        print(f"Generated {i+1}/50: {save_filename}")

print(f"🎉 Generation Finished! Saved to {OUT_DIR}")
