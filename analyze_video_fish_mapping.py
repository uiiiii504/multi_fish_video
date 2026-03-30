"""
分析视频与鱼的对应关系
假设：一个视频有4条鱼，每组一条
数据结构：control(0-18), negative(19-37), Trp(38-56), Tyr(57-75)
所以视频i对应的鱼是：i, i+19, i+38, i+57
"""

import os
import joblib

def analyze_video_fish_mapping(working_dir, prefix):
    predictions_path = os.path.join(working_dir, f"{prefix}_predictions.sav")
    
    with open(predictions_path, 'rb') as f:
        predictions = joblib.load(f)
    
    filepaths = predictions[2] if len(predictions) > 2 else []
    
    print("=== 分析视频与鱼的对应关系 ===\n")
    
    groups = ['control', 'negative', 'Trp', 'Tyr']
    fish_per_group = 19
    
    print("假设：一个视频有4条鱼，每组一条")
    print(f"组顺序：{groups}")
    print(f"每组鱼数：{fish_per_group}")
    print()
    
    print("=== 视频与鱼的对应关系 ===")
    for video_idx in range(fish_per_group):
        print(f"\n视频 {video_idx}:")
        for g, group_name in enumerate(groups):
            fish_idx = video_idx + g * fish_per_group
            if fish_idx < len(filepaths):
                filepath = filepaths[fish_idx]
                filename = os.path.basename(filepath)
                print(f"  {group_name}: 鱼_{fish_idx} -> {filename}")
    
    print("\n=== 验证文件名模式 ===")
    for video_idx in [0, 1, 2]:
        print(f"\n视频 {video_idx} 的文件名:")
        for g, group_name in enumerate(groups):
            fish_idx = video_idx + g * fish_per_group
            if fish_idx < len(filepaths):
                filepath = filepaths[fish_idx]
                filename = os.path.basename(filepath)
                print(f"  {group_name}: {filename}")

if __name__ == "__main__":
    working_dir = r"D:\study\gradu_pro\data\6dpf\output_6dpf_24h_corrected"
    prefix = "6dpf_24h"
    analyze_video_fish_mapping(working_dir, prefix)
