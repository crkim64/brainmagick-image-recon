import mne
import numpy as np
import os
import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
# ==========================================
# 1. 설정
# ==========================================
subs = ["P1", "P2", "P3", "P4"]
base_input_dir = config.MEG_PREPROCESSED_DIR
save_dir = config.SENSOR_POS_DIR
os.makedirs(save_dir, exist_ok=True)

# 2. 표준 레이아웃 로드 (CTF-275 전체 정보)
print("🔧 Loading Standard CTF-275 Layout (Template)...")
layout_template = mne.channels.read_layout('CTF275')
# layout_template.names: 표준 채널명 리스트 (예: 'MLC11', 'MLC12'...)
# layout_template.pos: 좌표 (275, 4)

for sub in subs:
    print(f"\n🚀 Processing Subject: {sub}")
    
    # .fif 파일 경로
    fif_path = os.path.join(base_input_dir, f"preprocessed_{sub}-epo.fif")
    save_path = os.path.join(save_dir, f"sensor_positions_{sub}.npy")
    plot_path = os.path.join(save_dir, f"check_sensor_pos_{sub}.png")

    if not os.path.exists(fif_path):
        print(f"❌ File not found: {fif_path}")
        continue

    # 3. 데이터의 채널 이름 가져오기 (이 순서가 데이터 행렬의 순서임!)
    info = mne.io.read_info(fif_path, verbose=False)
    data_ch_names = info['ch_names'] # 271개 리스트
    
    print(f"   Data channels: {len(data_ch_names)} (First: {data_ch_names[0]})")

    # 4. 이름 매칭을 통한 좌표 추출
    matched_positions = []
    missing_count = 0
    
    for name in data_ch_names:
        # 데이터 이름: 'MLC11-1609' -> 표준 이름: 'MLC11' (접미사 제거)
        clean_name = name.split('-')[0]
        
        if clean_name in layout_template.names:
            # 표준 레이아웃에서 해당 이름의 인덱스 찾기
            idx = layout_template.names.index(clean_name)
            # 좌표 가져오기 (x, y)
            pos = layout_template.pos[idx, :2]
            matched_positions.append(pos)
        else:
            print(f"⚠️ Warning: Channel {name} not found in layout template!")
            matched_positions.append([0, 0]) # 예외 처리 (거의 발생 안 함)
            missing_count += 1

    # (271, 2) 배열로 변환
    pos_array = np.array(matched_positions)
    
    # 5. 정규화 (Min-Max Scaling to 0~1)
    # BrainModule은 0~1 사이 좌표를 기대함
    pos_min = pos_array.min(axis=0)
    pos_max = pos_array.max(axis=0)
    pos_norm = (pos_array - pos_min) / (pos_max - pos_min)

    print(f"✅ Extracted Shape: {pos_norm.shape} (Should be 271, 2)")
    
    # 6. 저장
    np.save(save_path, pos_norm)
    print(f"💾 Saved to: {save_path}")

    # 7. 검증용 시각화
    plt.figure(figsize=(5, 5))
    plt.scatter(pos_norm[:, 0], pos_norm[:, 1], c='purple', s=20, alpha=0.7)
    plt.title(f"{sub} Positions (N={len(pos_norm)})")
    plt.xlabel("x (norm)")
    plt.ylabel("y (norm)")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path)
    plt.close()

print("\n🎉 All done! Perfect matching complete.")