
import sys
import os
import argparse
import numpy as np
import torch
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add paths
sys.path.append('versatile_diffusion')
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(PROJ_DIR))

try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

# ==========================================
# Configuration
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--subject", default='P1')
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

DEVICE = args.device
print(f"🚀 Evaluating Feature Alignment for {args.subject} on {DEVICE}...")

# Paths
DATA_DIR = './data'
TEST_MEG_PATH = os.path.join(DATA_DIR, 'test', f'{args.subject}_test.h5')
GT_VISION_PATH = os.path.join(DATA_DIR, 'extracted_features', args.subject, f'{args.subject}_clipvision_test.h5')
GT_TEXT_PATH = os.path.join(DATA_DIR, 'extracted_features', args.subject, f'{args.subject}_cliptext_test.h5')

if not os.path.exists(GT_VISION_PATH):
    GT_VISION_PATH = os.path.join(DATA_DIR, 'extracted_features', args.subject, f'{args.subject}_clip_vision_test.h5')
if not os.path.exists(GT_TEXT_PATH):
    GT_TEXT_PATH = os.path.join(DATA_DIR, 'extracted_features', args.subject, f'{args.subject}_clip_text_test.h5')

CKPT_VISION = './checkpoints/clip_vision_hybrid/best_model.pth'
CKPT_TEXT = './checkpoints/clip_text_hybrid/best_model.pth'

# ==========================================
# Helper Classes
# ==========================================
class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions
    def __len__(self):
        return self.meg.shape[0]

def load_brain_module(ckpt_path, target_dim, device):
    print(f"⌛ Loading BrainModule: {ckpt_path}")
    model = BrainModule(
        n_time_steps=180,
        in_channels=271,
        n_subjects=4,
        target_dim=target_dim,
        use_clip_head=True,
        use_mse_head=True,
    ).to(device)
    
    # Checkpoint Strict Loading
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        model.eval()
        return model
    else:
        print(f"⚠️ Checkpoint NOT FOUND: {ckpt_path}")
        return None

# ==========================================
# Main
# ==========================================
def main():
    # 1. Load Data
    print("📂 Loading Data...")
    
    # MEG
    with h5py.File(TEST_MEG_PATH, 'r') as f:
        meg_data = f['meg'][:]
        # Need category & exemplar to match ground-truth features
        test_cats = f['category_nr'][:]
        test_exs = f['exemplar_nr'][:]
        
    # GT Vision & Text from smallset
    SMALL_GT_PATH = './data/retrieval_extracted_features/extracted_features_smallset.h5'
    
    gt_vision = None
    gt_text = None
    
    if os.path.exists(SMALL_GT_PATH):
        print(f"✅ Loading Ground-Truth Features from {SMALL_GT_PATH}")
        with h5py.File(SMALL_GT_PATH, 'r') as f:
            gt_cats = f['category_nr'][:]
            gt_exs = f['exemplar_nr'][:]
            
            # (200, 768)
            feat_vis = f['clipvision'][:] 
            feat_txt = f['cliptext'][:]
            
            # Map index dictionary (cat, ex) -> dict idx
            gt_map = {(c, e): i for i, (c, e) in enumerate(zip(gt_cats, gt_exs))}
            
            # Reconstruct full matched arrays (N, 768)
            N = len(meg_data)
            gt_vision = np.zeros((N, 768), dtype=np.float32)
            gt_text = np.zeros((N, 768), dtype=np.float32)
            
            found_count = 0
            for i in range(N):
                key = (test_cats[i], test_exs[i])
                if key in gt_map:
                    idx = gt_map[key]
                    gt_vision[i] = feat_vis[idx]
                    gt_text[i] = feat_txt[idx]
                    found_count += 1
                    
            print(f"   Matched {found_count}/{N} trials with Ground-Truth Features.")
    else:
        print(f"⚠️ GT path not found: {SMALL_GT_PATH}")

    print(f"   MEG Shape: {meg_data.shape}")
    if gt_vision is not None: print(f"   GT Vision Shape: {gt_vision.shape}")
    if gt_text is not None: print(f"   GT Text Shape: {gt_text.shape}")

    sub_idx = torch.tensor([{'P1':0, 'P2':1, 'P3':2, 'P4':3}[args.subject]]).long().to(DEVICE)

    # 2. Load Models & Stats
    bm_vision = load_brain_module(CKPT_VISION, 768, DEVICE)
    bm_text = load_brain_module(CKPT_TEXT, 768, DEVICE)

    print("📊 Loading Normalization Statistics (for Target Denormalization)...")
    stats_vision, stats_text = None, None
    if os.path.exists('./checkpoints/clip_vision_hybrid/clip_vision_stats.npz'):
        s_v = np.load('./checkpoints/clip_vision_hybrid/clip_vision_stats.npz')
        stats_vision = {
            'mean': torch.from_numpy(s_v['mean']).float().to(DEVICE),
            'std': torch.from_numpy(s_v['std']).float().to(DEVICE)
        }
    if os.path.exists('./checkpoints/clip_text_hybrid/clip_text_stats.npz'):
        s_t = np.load('./checkpoints/clip_text_hybrid/clip_text_stats.npz')
        stats_text = {
            'mean': torch.from_numpy(s_t['mean']).float().to(DEVICE),
            'std': torch.from_numpy(s_t['std']).float().to(DEVICE)
        }

    # 3. Evaluate Loop
    if gt_vision is not None and gt_text is not None:
        dataset = TensorDataset(
            torch.from_numpy(meg_data).float(),
            torch.from_numpy(gt_vision).float(),
            torch.from_numpy(gt_text).float()
        )
    else:
        dataset = TensorDataset(torch.from_numpy(meg_data).float())
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = {
        'vis_mse': [], 'vis_cos_global': [],
        'txt_mse': [], 'txt_cos_global': []
    }
    
    # Stats for Distribution Check
    stats = {
        'pred_vis_mean': [], 'pred_vis_std': [],
        'gt_vis_mean': [], 'gt_vis_std': [],
        'pred_txt_mean': [], 'pred_txt_std': [],
        'gt_txt_mean': [], 'gt_txt_std': [],
    }

    print("⚡ Running Inference & Evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if len(batch) == 3:
                meg, gt_vis, gt_txt = batch
                gt_vis = gt_vis.to(DEVICE)
                gt_txt = gt_txt.to(DEVICE)
            else:
                meg = batch[0]
                gt_vis, gt_txt = None, None
                
            meg = meg.to(DEVICE)
            batch_sub_idx = sub_idx.expand(meg.shape[0])

            # --- VISION ---
            if bm_vision:
                out_v = bm_vision(meg, batch_sub_idx)
                pred_vis = out_v['mse'] # (B, 768)
                
                # Denormalize & Z-score
                if stats_vision is not None:
                    p_mean, p_std = pred_vis.mean(dim=-1, keepdim=True), pred_vis.std(dim=-1, keepdim=True)
                    pred_vis = (pred_vis - p_mean) / (p_std + 1e-6)
                    pred_vis = pred_vis * (stats_vision['std'] + 1e-6) + stats_vision['mean']
                
                if gt_vis is not None:
                    # Depending on how GT is structured, old structure was (B, 257, 768)
                    # We only compare against CLS token (index 0)
                    tgt_v = gt_vis[:, 0, :] if gt_vis.dim() == 3 else gt_vis
                    
                    # MSE
                    metrics['vis_mse'].append(F.mse_loss(pred_vis, tgt_v).item())
                    # Cosine
                    metrics['vis_cos_global'].append(F.cosine_similarity(pred_vis, tgt_v).mean().item())
                    
                    stats['gt_vis_mean'].append(tgt_v.mean().item())
                    stats['gt_vis_std'].append(tgt_v.std().item())
                    
                stats['pred_vis_mean'].append(pred_vis.mean().item())
                stats['pred_vis_std'].append(pred_vis.std().item())

            # --- TEXT ---
            if bm_text:
                out_t = bm_text(meg, batch_sub_idx)
                pred_txt = out_t['mse'] # (B, 768)
                
                # Denormalize & Z-score
                if stats_text is not None:
                    p_mean, p_std = pred_txt.mean(dim=-1, keepdim=True), pred_txt.std(dim=-1, keepdim=True)
                    pred_txt = (pred_txt - p_mean) / (p_std + 1e-6)
                    pred_txt = pred_txt * (stats_text['std'] + 1e-6) + stats_text['mean']
                
                if gt_txt is not None:
                    # Old structure was (B, 77, 768). Mean over tokens.
                    tgt_t = gt_txt.mean(1) if gt_txt.dim() == 3 else gt_txt
                    
                    metrics['txt_mse'].append(F.mse_loss(pred_txt, tgt_t).item())
                    metrics['txt_cos_global'].append(F.cosine_similarity(pred_txt, tgt_t).mean().item())
                    
                    stats['gt_txt_mean'].append(tgt_t.mean().item())
                    stats['gt_txt_std'].append(tgt_t.std().item())
                    
                stats['pred_txt_mean'].append(pred_txt.mean().item())
                stats['pred_txt_std'].append(pred_txt.std().item())

    # 4. Report
    print("\n📊 Evaluation Results:")
    print("="*40)
    
    if bm_vision:
        print(f"👁️ CLIP VISION:")
        if metrics['vis_mse']:
            print(f"   MSE: {np.mean(metrics['vis_mse']):.5f}")
            print(f"   Cosine Sim (CLS Token):   {np.mean(metrics['vis_cos_global']):.4f}")
            print(f"   GT   Stats | Mean: {np.mean(stats['gt_vis_mean']):.4f}, Std: {np.mean(stats['gt_vis_std']):.4f}")
        print("-" * 20)
        print(f"   Pred Stats | Mean: {np.mean(stats['pred_vis_mean']):.4f}, Std: {np.mean(stats['pred_vis_std']):.4f}")
        print("="*40)

    if bm_text:
        print(f"📝 CLIP TEXT:")
        if metrics['txt_mse']:
            print(f"   MSE: {np.mean(metrics['txt_mse']):.5f}")
            print(f"   Cosine Sim (Mean Token):  {np.mean(metrics['txt_cos_global']):.4f}")
            print(f"   GT   Stats | Mean: {np.mean(stats['gt_txt_mean']):.4f}, Std: {np.mean(stats['gt_txt_std']):.4f}")
        print("-" * 20)
        print(f"   Pred Stats | Mean: {np.mean(stats['pred_txt_mean']):.4f}, Std: {np.mean(stats['pred_txt_std']):.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
