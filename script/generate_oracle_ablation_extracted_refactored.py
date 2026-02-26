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
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD

# ==========================================
# 1. Setup & Config
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1', help="Subject Name")
parser.add_argument("--vd_device", default='cuda', help="Device for Versatile Diffusion")
parser.add_argument("--num_samples", type=int, default=2, help="Number of samples to generate")
args = parser.parse_args()

# Paths
DATA_DIR = './data'
OUT_DIR = f'results/image_generation/{args.subject}_oracle_ablation_extracted_refactored'
os.makedirs(OUT_DIR, exist_ok=True)

EXTRACTED_DIR = os.path.join(DATA_DIR, 'extracted_features', args.subject)

FEATURES_AUTOKL = os.path.join(EXTRACTED_DIR, f'{args.subject}_autokl_test.h5')
FEATURES_VISION = os.path.join(EXTRACTED_DIR, f'{args.subject}_clip_vision_test.h5')
FEATURES_TEXT   = os.path.join(EXTRACTED_DIR, f'{args.subject}_clip_text_test.h5')

DICT_PATH = os.path.join(DATA_DIR, 'image_path_dictionary.h5')
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')

# ==========================================
# 2. Loading Utilities
# ==========================================
print(f"📚 Loading Image Dictionary...")
IMAGE_MAP = {}
with h5py.File(DICT_PATH, 'r') as f:
    cats = f['category_nr'][:]
    exs = f['exemplar_nr'][:]
    paths = [p.decode('utf-8') if isinstance(p, bytes) else p for p in f['image_path'][:]]
    for c, e, p in zip(cats, exs, paths):
        IMAGE_MAP[(c, e)] = p

print("🚀 Loading Versatile Diffusion...")
cfgm_name = 'vd_noema'
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
sd = {k: v for k, v in sd.items() if k in net.state_dict()}
net.load_state_dict(sd, strict=False)

net.clip.to(args.vd_device)
net.autokl.to(args.vd_device)
net.model.to(args.vd_device)
if hasattr(net.model, 'diffusion_model'):
    net.model.diffusion_model.device = args.vd_device
net.model.device = args.vd_device
sampler = DDIMSampler_VD(net)

# ==========================================
# 3. Load Features & Generate
# ==========================================
print("📂 Loading Test Metadata & Extracted Features...")

with h5py.File(TEST_MEG_PATH, 'r') as f:
    test_cats = f['category_nr'][:]
    test_exs = f['exemplar_nr'][:]

# Open Feature Files
f_kl = h5py.File(FEATURES_AUTOKL, 'r')
f_vis = h5py.File(FEATURES_VISION, 'r')
f_txt = h5py.File(FEATURES_TEXT, 'r')

# Helper to refactor features
def refactor_autokl(z):
    # z: (1, 4, 64, 64)
    # Mean over channel dim (1) -> (1, 1, 64, 64) -> Repeat -> (1, 4, 64, 64)
    z_mean = z.mean(dim=1, keepdim=True)
    z_new = z_mean.repeat(1, 4, 1, 1)
    return z_new

def refactor_clip_vision(c):
    # c: (1, 257, 768)
    # Take CLS (idx 0), Zero Pad others
    try:
        cls_token = c[:, 0:1, :].clone() # (1, 1, 768)
        c_new = torch.zeros_like(c)
        c_new[:, 0:1, :] = cls_token
        return c_new
    except IndexError:
        print(f"Warning: CLIP Vision shape mismatch {c.shape}")
        return c

def refactor_clip_text(c):
    # c: (1, 77, 768)
    # Mean Token -> Replicate
    c_mean = c.mean(dim=1, keepdim=True) # (1, 1, 768)
    c_new = c_mean.repeat(1, 77, 1) # (1, 77, 768)
    return c_new

# Indices to test
indices = range(args.num_samples) 
print(f"✨ Start Refactored Feature Ablation for {len(indices)} samples...")

# Dummy Placeholders for Condition Mixing
dummy_text = ''
utx = net.clip_encode_text(dummy_text).to(args.vd_device)
dummy_img = torch.zeros((1, 3, 224, 224)).to(args.vd_device)
uim = net.clip_encode_vision(dummy_img).to(args.vd_device)

with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    for idx_cnt, idx in enumerate(indices):
        if idx >= len(test_cats): break
        
        cat_id = test_cats[idx]
        ex_id = test_exs[idx]
        
        rel_path = IMAGE_MAP.get((cat_id, ex_id), "unknown")
        if rel_path == "unknown": continue
        
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        safe_name = "".join([c if c.isalnum() or c in ('_','-') else '_' for c in base_name])

        # Load GT Image for reference
        full_path = os.path.join(DATA_DIR, rel_path)
        if os.path.exists(full_path):
            gt_img = Image.open(full_path).convert('RGB')
            gt_img.save(os.path.join(OUT_DIR, f"{idx:04d}_{safe_name}_GT.png"))
        else:
             # Fallback logic
            fname = os.path.basename(rel_path)
            parts = fname.split('_')
            for i in range(1, len(parts)):
                sub_cat = "_".join(parts[:i])
                fallback_path = os.path.join(DATA_DIR, 'images_meg', sub_cat, fname)
                if os.path.exists(fallback_path):
                    gt_img = Image.open(fallback_path).convert('RGB')
                    gt_img.save(os.path.join(OUT_DIR, f"{idx:04d}_{safe_name}_GT.png"))
                    break
        
        print(f"Processing {idx} ({idx_cnt+1}/{len(indices)}): {safe_name}")

        # ---------------------------
        # Get Features & Refactor
        # ---------------------------
        # AutoKL: (1, 4, 64, 64)
        z_gt_np = f_kl['features'][idx]
        z_gt = torch.from_numpy(z_gt_np).float().unsqueeze(0).to(args.vd_device)
        z_refactored = refactor_autokl(z_gt)
        
        # CLIP Vision: (1, 257, 768)
        c_vis_np = f_vis['features'][idx]
        c_vis = torch.from_numpy(c_vis_np).float().unsqueeze(0).to(args.vd_device)
        c_vis_refactored = refactor_clip_vision(c_vis)
        
        # CLIP Text: (1, 77, 768)
        c_text_np = f_txt['features'][idx]
        c_text = torch.from_numpy(c_text_np).float().unsqueeze(0).to(args.vd_device)
        c_text_refactored = refactor_clip_text(c_text)

        # ---------------------------
        # Generation Loop
        # ---------------------------
        # We perform ablation to see contribution of each Refactored component
        conditions = {
            'Refactored_All': (z_refactored, c_vis_refactored, c_text_refactored, 0.5), 
            'Refactored_AutoKL': (z_refactored, uim, utx, 0.5),
            'Refactored_Vision': (None, c_vis_refactored, utx, 1.0),
            'Refactored_Text':   (None, uim, c_text_refactored, 0.0), # Text Only uses mix 0.0
            'Refactored_Vision+Text': (None,  c_vis_refactored, c_text_refactored, 0.5),
        }
        
        sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)

        for mode, (z_in, c_v, c_t, mix) in conditions.items():
            
            if z_in is not None:
                # Reconstruction / Editing Mode (AutoKL guided)
                t_enc = int(0.75 * 50) # strength 0.75
                # Note: Stochastic encode adds noise to z_in
                # Since z_in is now "channel averaged", it lacks spatial details.
                # Encoding might help add some texture back, or it might just blur it more.
                z_enc = sampler.stochastic_encode(z_in, torch.tensor([t_enc]).to(args.vd_device))
            else:
                # Generation Mode (From Scratch)
                t_enc = 50 
                z_enc = torch.randn(1, 4, 64, 64).to(args.vd_device)

            # Decode
            z = sampler.decode_dc(
                x_latent=z_enc,
                first_conditioning=[uim, c_v],
                second_conditioning=[utx, c_t],
                t_start=t_enc,
                unconditional_guidance_scale=7.5,
                xtype='image',
                first_ctype='vision',
                second_ctype='prompt',
                mixed_ratio=(1.0 - mix) 
            )
            
            # VAE Decode
            z = z.float()
            net.autokl.to(args.vd_device).float() 
            x = net.autokl_decode(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            
            img = tvtrans.ToPILImage()(x[0].cpu())
            img.save(os.path.join(OUT_DIR, f"{idx:04d}_{safe_name}_{mode}.png"))

f_kl.close()
f_vis.close()
f_txt.close()
print(f"🎉 Refactored Generation Finished! Results saved to {OUT_DIR}")
