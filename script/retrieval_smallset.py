import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from retrieval_utils import ensure_features

# Import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
try:
    from model.brainmodule import BrainModule
except ImportError:
    # try one directory up
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.brainmodule import BrainModule

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Default checkpoint paths (adjust if needed)
CKPT_AUTOKL = './checkpoints/autokl_hybrid/best_model.pth'
CKPT_CLIP_TEXT = './checkpoints/clip_text_hybrid/best_model.pth'
CKPT_CLIP_VISION = './checkpoints/clip_vision_hybrid/best_model.pth'

def load_brain_module(ckpt_path, out_dim):
    # Matches the training spec
    model = BrainModule(
        n_time_steps=180,
        in_channels=271,
        n_subjects=4,
        target_dim=out_dim,
        use_clip_head=True,
        use_mse_head=True,
    ).to(DEVICE)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"✅ Loaded checkpoint: {ckpt_path}")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {ckpt_path}. Using random weights.")
    model.eval()
    return model

def calculate_top5_accuracy(pred_vectors, gt_vectors):
    """
    Given (N, D) predictions and (N, D) ground truths,
    computes average Top-5 retrieval accuracy using cosine similarity.
    """
    preds = F.normalize(pred_vectors, p=2, dim=-1)
    gts = F.normalize(gt_vectors, p=2, dim=-1)
    
    # Cosine Similarity Matrix: (N_preds, N_gts)
    sim_matrix = torch.matmul(preds, gts.T)
    
    # Get Top-5 indices for each prediction
    top5_indices = sim_matrix.topk(5, dim=-1).indices # (N, 5)
    
    # Check if the correct index is in the Top-5
    correct_indices = torch.arange(len(preds), device=preds.device).unsqueeze(1) # (N, 1)
    matches = (top5_indices == correct_indices).float()
    
    top5_acc = matches.sum(dim=-1).mean().item() * 100
    return top5_acc

def main():
    print("="*50)
    print("🧠 MEG Brain Decoding: Small Set Retrieval (Top-5)")
    print("="*50)
    
    # 1. Ensure features are extracted!
    path_small, _ = ensure_features(device=DEVICE)
    
    # 2. Load the ground truth small-set features
    print(f"\n📥 Loading Small Set ground-truth features...")
    with h5py.File(path_small, 'r') as f:
        gt_autokl = f['autokl'][:]
        gt_cliptext = f['cliptext'][:]
        gt_clipvision = f['clipvision'][:]
        gt_cats = f['category_nr'][:]
        gt_exs = f['exemplar_nr'][:]
        
    num_small = len(gt_cats)
    print(f"   Identified {num_small} unique images in Small Set.")
    
    gt_images = list(zip(gt_cats, gt_exs))
    
    # 3. Load Models
    print("\n🚀 Loading Brain Modules...")
    model_autokl = load_brain_module(CKPT_AUTOKL, out_dim=4096)
    model_cliptext = load_brain_module(CKPT_CLIP_TEXT, out_dim=768)
    model_clipvision = load_brain_module(CKPT_CLIP_VISION, out_dim=768)
    
    # We evaluate sequentially for each subject, or combined?
    # The paper says: "accuracies are based on MEG responses to the averaged subsets... for a single participant"
    # We will compute accuracies PER participant for P1, P2, P3, P4.
    
    subjects = ['P1', 'P2', 'P3', 'P4']
    
    results_dict = {m: [] for m in ['AutoKL', 'CLIP_Text', 'CLIP_Vision']}
    
    for sub_idx, sub in enumerate(subjects):
        test_path = f'./data/test/{sub}_test.h5'
        if not os.path.exists(test_path):
            print(f"⚠️ {test_path} missing. Skipping {sub}.")
            continue
            
        print(f"\n📊 --- Evaluating Subect: {sub} ---")
        
        # Dictionary to accumulate predicted vectors for each unique image
        pred_accums = {
            'autokl': {img: [] for img in gt_images},
            'cliptext': {img: [] for img in gt_images},
            'clipvision': {img: [] for img in gt_images},
        }
        
        with h5py.File(test_path, 'r') as f:
            meg_data = f['meg'][:]
            cats = f['category_nr'][:]
            exs = f['exemplar_nr'][:]
            
        N = len(meg_data)
        batch_size = 128
        
        # Inference
        with torch.no_grad():
            for i in tqdm(range(0, N, batch_size), desc=f"Predicting {sub}"):
                batch_meg = torch.from_numpy(meg_data[i:i+batch_size]).float().to(DEVICE)
                batch_cats = cats[i:i+batch_size]
                batch_exs = exs[i:i+batch_size]
                subj_ids = torch.full((len(batch_meg),), sub_idx, dtype=torch.long).to(DEVICE)
                
                # Forward Pass
                out_a = model_autokl(batch_meg, subj_ids)['clip'].cpu()
                out_t = model_cliptext(batch_meg, subj_ids)['clip'].cpu()
                out_v = model_clipvision(batch_meg, subj_ids)['clip'].cpu()
                
                for b in range(len(batch_meg)):
                    img_pair = (batch_cats[b], batch_exs[b])
                    if img_pair in gt_images:
                        pred_accums['autokl'][img_pair].append(out_a[b])
                        pred_accums['cliptext'][img_pair].append(out_t[b])
                        pred_accums['clipvision'][img_pair].append(out_v[b])
        
        # Calculate Averaged Predictions and Matches
        print(f"   Computing mean predictions & calculating Top-5 Acc...")
        final_preds = {'autokl': [], 'cliptext': [], 'clipvision': []}
        ordered_gts = {'autokl': [], 'cliptext': [], 'clipvision': []}
        
        for idx_gt, img_pair in enumerate(gt_images):
            # If the subject saw this image
            if len(pred_accums['autokl'][img_pair]) > 0:
                # Average the Predicted Vectors ("값을 평균해서")
                final_preds['autokl'].append(torch.stack(pred_accums['autokl'][img_pair]).mean(dim=0))
                final_preds['cliptext'].append(torch.stack(pred_accums['cliptext'][img_pair]).mean(dim=0))
                final_preds['clipvision'].append(torch.stack(pred_accums['clipvision'][img_pair]).mean(dim=0))
                
                ordered_gts['autokl'].append(torch.from_numpy(gt_autokl[idx_gt]))
                ordered_gts['cliptext'].append(torch.from_numpy(gt_cliptext[idx_gt]))
                ordered_gts['clipvision'].append(torch.from_numpy(gt_clipvision[idx_gt]))
                
        if len(final_preds['autokl']) == 0:
            print(f"   No small set images found for {sub}.")
            continue
            
        t_preds_a = torch.stack(final_preds['autokl']).to(DEVICE)
        t_preds_t = torch.stack(final_preds['cliptext']).to(DEVICE)
        t_preds_v = torch.stack(final_preds['clipvision']).to(DEVICE)
        
        t_gts_a = torch.stack(ordered_gts['autokl']).to(DEVICE)
        t_gts_t = torch.stack(ordered_gts['cliptext']).to(DEVICE)
        t_gts_v = torch.stack(ordered_gts['clipvision']).to(DEVICE)
        
        acc_a = calculate_top5_accuracy(t_preds_a, t_gts_a)
        acc_t = calculate_top5_accuracy(t_preds_t, t_gts_t)
        acc_v = calculate_top5_accuracy(t_preds_v, t_gts_v)
        
        print(f"   [Top-5 Accuracy {sub}] AutoKL: {acc_a:.2f}% | CLIP Text: {acc_t:.2f}% | CLIP Vision: {acc_v:.2f}%")
        results_dict['AutoKL'].append(acc_a)
        results_dict['CLIP_Text'].append(acc_t)
        results_dict['CLIP_Vision'].append(acc_v)
        
    print("\n" + "="*50)
    print("🏆 FINAL AVERAGE RESULTS (Across Participants)")
    print("="*50)
    print(f"AutoKL     : {np.mean(results_dict['AutoKL']):.2f}%")
    print(f"CLIP Text  : {np.mean(results_dict['CLIP_Text']):.2f}%")
    print(f"CLIP Vision: {np.mean(results_dict['CLIP_Vision']):.2f}%")

if __name__ == "__main__":
    main()
