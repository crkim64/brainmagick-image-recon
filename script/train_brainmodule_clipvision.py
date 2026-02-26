import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from tqdm import tqdm
import logging
import time
import h5py

# 상위 폴더를 path에 추가하여 model 모듈을 불러옴
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from model.brainmodule import BrainModule
except ImportError:
    sys.path.append('./')
    from model.brainmodule import BrainModule

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
CONFIG = {
    'batch_size': 128,      
    'lr': 3e-4,             
    'epochs': 100,
    'patience': 10,         
    'device': 'cuda',
    'subjects': ['P1', 'P2', 'P3', 'P4'],
    
    # 데이터 스펙
    'in_channels': 271,     
    'time_len': 180,        
    
    # [수정] Vision Feature Dimensions
    # Target: CLS Token (768) for both Heads
    
    # 1. CLIP Head (Semantic): CLS Token (768)
    'out_dim_clip': 768,    
    
    # 2. MSE Head (Reconstruction): CLS Token (768)
    'out_dim_mse': 768, 
    
    'lambda_loss': 0.5,     
    'target_normalization': True, # [NEW] User Request
    
    # 경로 설정
    'data_dir': './data',   
    'train_meg_path': './data/train/combined_train.h5',
    'train_feat_path': './data/extracted_features/combined_clip_vision_train_cls.h5', 
    'pos_dir': './data/sensor_positions',
    'ckpt_dir': './checkpoints/clip_vision_hybrid', 
    'log_dir': './logs'                 
}

# ==========================================
# 2. 로깅 설정
# ==========================================
os.makedirs(CONFIG['log_dir'], exist_ok=True)
os.makedirs(CONFIG['ckpt_dir'], exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(os.path.join(CONFIG['log_dir'], 'train_clip_fullseq.log'), mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def log_print(msg):
    logger.info(msg)

# ==========================================
# 3. 유틸리티
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            log_print(f'🛑 EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        log_print(f'✅ Validation loss decreased. Saving best model to {self.path}')

class BatchObject:
    def __init__(self, meg, subject_index, positions):
        self.meg = meg
        self.subject_index = subject_index
        self.meg_positions = positions 
    def __len__(self):
        return self.meg.shape[0]

# ==========================================
# 4. 데이터셋 정의 (Full CLIP Vision Sequence + Stats)
# ==========================================
class FullCLIPDataset(Dataset):
    def __init__(self, meg_path, feat_path, pos_dir, subjects, stats_save_path, target_normalization=True):
        self.meg_path = meg_path
        self.feat_path = feat_path
        self.subjects = subjects
        
        with h5py.File(meg_path, 'r') as f:
            self.length = f['meg'].shape[0]
            self.categories = f['category_nr'][:]
            self.exemplars = f['exemplar_nr'][:]
            
        self.subject_positions_list = []
        for sub in subjects:
            pos_path = os.path.join(pos_dir, f"sensor_positions_{sub}.npy")
            if os.path.exists(pos_path):
                pos = np.load(pos_path)
                if np.isnan(pos).any(): pos = np.nan_to_num(pos)
            else:
                pos = np.random.rand(271, 2)
            self.subject_positions_list.append(torch.tensor(pos, dtype=torch.float32))
        
        self.target_normalization = target_normalization

        # [수정] MSE Normalization
        # Load Mean/Std and apply (target - mean) / std
        if not os.path.exists(stats_save_path):
            log_print("📊 Calculating Full CLIP Vision Mean/Std...")
            self._compute_and_save_stats(feat_path, stats_save_path)
        else:
            log_print(f"📊 Full CLIP Vision stats found at {stats_save_path}")

        stats = np.load(stats_save_path)
        self.target_mean = torch.from_numpy(stats['mean']).float()
        self.target_std = torch.from_numpy(stats['std']).float()

        self.meg_hf = None
        self.feat_hf = None

    def _compute_and_save_stats(self, feat_path, save_path, chunk_size=1000):
        with h5py.File(feat_path, 'r') as f:
            dset = f['features'] 
            n_samples = dset.shape[0]
            out_dim = 768 
            
            sum_features = np.zeros(out_dim, dtype=np.float64)
            sum_sq_features = np.zeros(out_dim, dtype=np.float64)
            
            for i in tqdm(range(0, n_samples, chunk_size), desc="Stats"):
                chunk = dset[i : i + chunk_size] 
                # [Optimized] Already extracted CLS token in preprocessed file
                # Shape: (B, 768)
                
                sum_features += np.sum(chunk, axis=0)
                sum_sq_features += np.sum(chunk ** 2, axis=0)
            
            mean = sum_features / n_samples
            variance = (sum_sq_features / n_samples) - (mean ** 2)
            std = np.sqrt(np.maximum(variance, 1e-6))
            
            np.savez(save_path, mean=mean, std=std)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.meg_hf is None:
            self.meg_hf = h5py.File(self.meg_path, 'r')
            self.feat_hf = h5py.File(self.feat_path, 'r')
            
        meg_data = self.meg_hf['meg'][idx]
        subj_idx = self.meg_hf['subject_idx'][idx]
        
        cat_id = self.categories[idx]
        ex_id = self.exemplars[idx]
        unique_img_id = cat_id * 100 + ex_id
        
        # [Optimized] CLIP Vision Feature Load (CLS Token)
        # Shape: (768,)
        feat_raw = self.feat_hf['features'][idx]
        
        target_raw = torch.from_numpy(feat_raw).float()
        
        if self.target_normalization:
            target_raw = (target_raw - self.target_mean) / (self.target_std + 1e-6)
        
        # 1. MSE Target: Raw
        target_mse = target_raw
        
        # 2. CLIP Target: Raw
        target_clip = target_raw.clone()
        
        meg_tensor = torch.from_numpy(meg_data).float()
        subj_tensor = torch.tensor(subj_idx, dtype=torch.long)
        pos_tensor = self.subject_positions_list[subj_idx]
        
        return meg_tensor, target_clip, target_mse, subj_tensor, pos_tensor, unique_img_id
        
    def __del__(self):
        if self.meg_hf is not None: self.meg_hf.close()
        if self.feat_hf is not None: self.feat_hf.close()

# ==========================================
# 5. Loss Function (SoftCLIP + MSE)
# ==========================================
class BrainDecodingLoss(nn.Module):
    def __init__(self, lambda_loss=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_loss = lambda_loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def soft_clip_loss(self, preds, targets, img_ids):
        preds = preds / (preds.norm(dim=-1, keepdim=True) + 1e-6)
        targets = targets / (targets.norm(dim=-1, keepdim=True) + 1e-6)
        
        logits = torch.matmul(preds, targets.T) * self.logit_scale.exp()
        
        labels_mask = (img_ids.unsqueeze(0) == img_ids.unsqueeze(1)).float()
        soft_labels = labels_mask / labels_mask.sum(dim=1, keepdim=True)
        
        log_probs = F.log_softmax(logits, dim=1)
        # Standard Cross Entropy Loss
        loss = -torch.sum(soft_labels * log_probs, dim=1).mean()
        
        return loss

    def forward(self, pred_clip, pred_mse, target_clip, target_mse, img_ids):
        # MSE: Full Sequence Raw Values
        l_mse = self.mse(pred_mse, target_mse)
        
        # CLIP: Global Vector SoftCLIP
        l_clip = self.soft_clip_loss(pred_clip, target_clip, img_ids)
        
        loss = self.lambda_loss * l_clip + (1 - self.lambda_loss) * l_mse
        return loss, l_clip, l_mse

# ==========================================
# 6. Training Main
# ==========================================
def main():
    log_print("\n" + "="*50)
    log_print(f"🚀 Initializing Training (Target: Full CLIP Sequence)")
    log_print(f"   CLIP Dim: {CONFIG['out_dim_clip']}")
    log_print(f"   MSE Dim:  {CONFIG['out_dim_mse']}")
    log_print("="*50)
    
    # Dataset Load
    stats_path = os.path.join(CONFIG['ckpt_dir'], 'clip_vision_stats.npz')
    full_dataset = FullCLIPDataset(
        meg_path=CONFIG['train_meg_path'],
        feat_path=CONFIG['train_feat_path'],
        pos_dir=CONFIG['pos_dir'],
        subjects=CONFIG['subjects'],
        stats_save_path=stats_path, # 저장만 함
        target_normalization=CONFIG['target_normalization']
    )
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))
    
    log_print(f"📊 Dataset Split -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # Model Init
    # Model Init
    model = BrainModule(
        n_time_steps=CONFIG['time_len'],
        in_channels=CONFIG['in_channels'],
        n_subjects=len(CONFIG['subjects']),
        target_dim=CONFIG['out_dim_clip'], # 768
        use_clip_head=True,
        use_mse_head=True,
    ).to(CONFIG['device'])

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], betas=(0.9, 0.999))
    criterion = BrainDecodingLoss(lambda_loss=CONFIG['lambda_loss']).to(CONFIG['device'])
    
    best_model_path = os.path.join(CONFIG['ckpt_dir'], 'best_model.pth')
    early_stopping = EarlyStopping(patience=CONFIG['patience'], path=best_model_path)

    log_print("🚀 Start Training Loop (Standard FP32)...")

    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        model.train()
        train_loss, train_clip, train_mse = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
        
        for i, (meg, t_clip, t_mse, subj_idx, pos, img_ids) in enumerate(pbar):
            meg = meg.to(CONFIG['device'])
            t_clip = t_clip.to(CONFIG['device'])
            t_mse = t_mse.to(CONFIG['device'])
            subj_idx = subj_idx.to(CONFIG['device'])
            img_ids = img_ids.to(CONFIG['device'])
            
            # 1. Forward
            outputs = model(meg, subj_idx)
            out_clip = outputs['clip']
            out_mse = outputs['mse']
            
            # 2. Loss
            loss, l_clip, l_mse = criterion(out_clip, out_mse, t_clip, t_mse, img_ids)
            
            # 3. Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_clip += l_clip.item()
            train_mse += l_mse.item()
            pbar.set_postfix({'L': f"{loss.item():.6f}", 'CLIP': f"{l_clip.item():.4f}", 'MSE': f"{l_mse.item():.6f}"})
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_clip, val_mse = 0, 0, 0
        with torch.no_grad():
            for meg, t_clip, t_mse, subj_idx, pos, img_ids in val_loader:
                meg = meg.to(CONFIG['device'])
                t_clip = t_clip.to(CONFIG['device'])
                t_mse = t_mse.to(CONFIG['device'])
                subj_idx = subj_idx.to(CONFIG['device'])
                img_ids = img_ids.to(CONFIG['device'])
                
                outputs = model(meg, subj_idx)
                out_clip = outputs['clip']
                out_mse = outputs['mse']
                
                loss, l_clip, l_mse = criterion(out_clip, out_mse, t_clip, t_mse, img_ids)
                
                val_loss += loss.item()
                val_clip += l_clip.item()
                val_mse += l_mse.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_clip = val_clip / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        
        elapsed = time.time() - start_time
        
        log_msg = (f"Epoch {epoch+1} | Time: {elapsed:.1f}s | "
                   f"Train Loss: {avg_train_loss:.6f} | "
                   f"Val Loss: {avg_val_loss:.6f} (CLIP: {avg_val_clip:.4f}, MSE: {avg_val_mse:.6f})")
        log_print(log_msg)
        
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            log_print("🛑 Early stopping triggered!")
            break

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(CONFIG['ckpt_dir'], f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            log_print(f"💾 Checkpoint saved: {ckpt_path}")

    log_print("🎉 Training Finished!")

if __name__ == "__main__":
    main()