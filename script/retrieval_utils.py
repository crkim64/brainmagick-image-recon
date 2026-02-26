import os
import sys
import h5py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Versatile Diffusion Paths
sys.path.append('versatile_diffusion')
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

DICT_PATH = './data/image_path_dictionary.h5'
FEAT_DIR = './data/retrieval_extracted_features'

def get_unique_images_from_test(test_files):
    small_set_imgs = set()
    large_set_imgs = set()
    
    for fpath in test_files:
        with h5py.File(fpath, 'r') as f:
            cats = f['category_nr'][:]
            exs = f['exemplar_nr'][:]
            
            pairs = list(zip(cats, exs))
            unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
            
            for (cat, ex), count in zip(unique_pairs, counts):
                if count > 1:
                    small_set_imgs.add((cat, ex))
                else:
                    large_set_imgs.add((cat, ex))
                    
    return list(small_set_imgs), list(large_set_imgs)

def load_image_map():
    IMAGE_MAP = {}
    with h5py.File(DICT_PATH, 'r') as f:
        cats = f['category_nr'][:]
        exs = f['exemplar_nr'][:]
        paths = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in f['image_path'][:]]
        for c, e, p in zip(cats, exs, paths):
            IMAGE_MAP[(c, e)] = p
    return IMAGE_MAP

def get_full_path(rel_path, image_root='./data'):
    if not rel_path: return None
    full_path = os.path.join(image_root, rel_path)
    if os.path.exists(full_path): return full_path
    fixed_path = rel_path.replace('images_test_meg', 'images_meg')
    full_path = os.path.join(image_root, fixed_path)
    if os.path.exists(full_path): return full_path
    
    filename = os.path.basename(fixed_path)
    if '_' in filename:
        cat_guess = filename.rsplit('_', 1)[0]
        try_path = os.path.join(image_root, 'images_meg', cat_guess, filename)
        if os.path.exists(try_path): return try_path
        parts = filename.split('_')
        for i in range(1, len(parts)):
            sub_cat = "_".join(parts[:i])
            try_path = os.path.join(image_root, 'images_meg', sub_cat, filename)
            if os.path.exists(try_path): return try_path
    return None

def extract_category_name(path_str):
    if not path_str: return "object"
    filename = os.path.basename(path_str)
    parent_dir = os.path.basename(os.path.dirname(path_str))
    if "images" not in parent_dir and parent_dir != ".":
        category = parent_dir
    else:
        if '_' in filename: category = filename.rsplit('_', 1)[0]
        else: category = os.path.splitext(filename)[0]
    return category.replace('_', ' ')

def extract_features_for_set(img_list, set_name, device):
    save_path = os.path.join(FEAT_DIR, f'extracted_features_{set_name}.h5')
    if os.path.exists(save_path):
        print(f"✅ Features for {set_name} already exist at {save_path}")
        return save_path

    os.makedirs(FEAT_DIR, exist_ok=True)
    IMAGE_MAP = load_image_map()
    
    print(f"🚀 Loading Versatile Diffusion Model for {set_name} Extraction...")
    cfgm_name = 'vd_noema'
    pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth' 
    cfgm = model_cfg_bank()(cfgm_name)
    net = get_model()(cfgm)
    sd = torch.load(pth, map_location='cpu')
    net.load_state_dict(sd, strict=False)
    net.model = None

    net.autokl = net.autokl.to(device).half()
    net.autokl.eval()
    net.clip = net.clip.to(device).float()
    net.clip.eval()

    transform_autokl = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    transform_clip = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    results = {'autokl': [], 'clipvision': [], 'cliptext': [], 'cats': [], 'exs': []}
    
    with torch.no_grad():
        for cat, ex in tqdm(img_list, desc=f"Extracting {set_name} features"):
            path_str = IMAGE_MAP.get((cat, ex), "")
            full_path = get_full_path(path_str)
            
            try:
                img = Image.open(full_path).convert('RGB')
            except:
                img = Image.new('RGB', (512, 512))
            
            # --- AutoKL ---
            img_autokl = transform_autokl(img) * 2 - 1
            img_autokl = img_autokl.unsqueeze(0).to(device).half()
            
            z_autokl = net.autokl_encode(img_autokl) # [1, 4, 64, 64]
            # Channel averaging
            z_autokl = z_autokl.mean(dim=1).flatten().cpu().numpy().astype(np.float32)
            
            # --- CLIP Vision ---
            img_clip = transform_clip(img) * 2 - 1
            img_clip = img_clip.unsqueeze(0).to(device).float()
            # Extract CLS token from [1, 257, 768]
            z_cv = net.clip_encode_vision(img_clip) # [1, 257, 768]
            z_clipvision = z_cv[:, 0, :].flatten().cpu().numpy().astype(np.float32) # CLS Token

            # --- CLIP Text ---
            category_name = extract_category_name(path_str)
            z_ct = net.clip_encode_text([category_name]) # [1, 77, 768]
            # User requirement: Mean Token for clip text. Wait, paper says "mean pooling across the 77 tokens" for CLIP Text FCR?
            # Let's mean pool over seq length
            z_cliptext = z_ct.mean(dim=1).flatten().cpu().numpy().astype(np.float32)

            results['autokl'].append(z_autokl)
            results['clipvision'].append(z_clipvision)
            results['cliptext'].append(z_cliptext)
            results['cats'].append(cat)
            results['exs'].append(ex)

    print(f"💾 Saving {set_name} features to HDF5...")
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('autokl', data=np.stack(results['autokl']))
        f.create_dataset('clipvision', data=np.stack(results['clipvision']))
        f.create_dataset('cliptext', data=np.stack(results['cliptext']))
        f.create_dataset('category_nr', data=np.array(results['cats'], dtype=np.int32))
        f.create_dataset('exemplar_nr', data=np.array(results['exs'], dtype=np.int32))
    
    return save_path

def ensure_features(device='cuda:0'):
    test_files = [f'./data/test/P{i}_test.h5' for i in range(1, 5)]
    small_set, large_set = get_unique_images_from_test(test_files)
    
    p1 = os.path.join(FEAT_DIR, 'extracted_features_smallset.h5')
    p2 = os.path.join(FEAT_DIR, 'extracted_features_largeset.h5')
    
    if os.path.exists(p1) and os.path.exists(p2):
        print("✅ Features already extracted for both sets.")
        return p1, p2
        
    p1 = extract_features_for_set(small_set, 'smallset', device)
    p2 = extract_features_for_set(large_set, 'largeset', device)
    return p1, p2

if __name__ == "__main__":
    ensure_features()
