import joblib
import numpy as np
import pandas as pd

# 加载sav文件
data_path = 'D:\\study\\gradu_pro\\tongji\\test\\video\\Mar-28-2026_data.sav'
predictions_path = 'D:\\study\\gradu_pro\\tongji\\test\\video\\Mar-28-2026_predictions.sav'

print("=" * 60)
print("分析 data.sav 文件结构")
print("=" * 60)

data = joblib.load(data_path)
print(f"\nData 总长度: {len(data)}")

# 分析每个元素
for i, item in enumerate(data):
    print(f"\n--- Item {i} ---")
    print(f"  类型: {type(item)}")
    
    if isinstance(item, str):
        print(f"  值: {item}")
    elif isinstance(item, int):
        print(f"  值: {item}")
    elif isinstance(item, list):
        print(f"  长度: {len(item)}")
        if len(item) > 0:
            print(f"  第一个元素类型: {type(item[0])}")
            if isinstance(item[0], str):
                print(f"  所有字符串值:")
                for j, s in enumerate(item):
                    print(f"    [{j}] {s}")
            elif isinstance(item[0], pd.DataFrame):
                print(f"  DataFrame形状: {item[0].shape}")
                print(f"  DataFrame列名: {list(item[0].columns)}")
            elif isinstance(item[0], list):
                print(f"  子列表长度: {len(item[0])}")
                print(f"  子列表内容: {item[0][:3]}")
    elif isinstance(item, np.ndarray):
        print(f"  形状: {item.shape}")
        print(f"  前5个值: {item[:5]}")

print("\n" + "=" * 60)
print("分析 predictions.sav 文件结构")
print("=" * 60)

predictions = joblib.load(predictions_path)
print(f"\nPredictions 总长度: {len(predictions)}")

# 分析每个元素
for i, item in enumerate(predictions):
    print(f"\n--- Item {i} ---")
    print(f"  类型: {type(item)}")
    
    if isinstance(item, str):
        print(f"  值: {item}")
    elif isinstance(item, int):
        print(f"  值: {item}")
    elif isinstance(item, list):
        print(f"  长度: {len(item)}")
        if len(item) > 0:
            print(f"  第一个元素类型: {type(item[0])}")
            if isinstance(item[0], str):
                print(f"  所有字符串值:")
                for j, s in enumerate(item[:10]):  # 只显示前10个
                    print(f"    [{j}] {s}")
                if len(item) > 10:
                    print(f"    ... 还有 {len(item)-10} 个")
            elif hasattr(item[0], 'shape'):
                print(f"  数组形状: {item[0].shape}")
                print(f"  第一个数组前3行:\n{item[0][:3]}")

print("\n" + "=" * 60)
print("尝试识别组信息")
print("=" * 60)

# 从predictions[2]获取文件名，提取组名
if len(predictions) > 2 and isinstance(predictions[2], list):
    filenames = predictions[2]
    print(f"\n找到 {len(filenames)} 个文件")
    
    # 提取组名
    groups = {}
    for i, filepath in enumerate(filenames):
        # 从路径中提取组名
        parts = filepath.replace('\\', '/').split('/')
        if len(parts) >= 2:
            group_name = parts[-2]  # 假设组名是倒数第二个目录
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(i)
    
    print(f"\n识别到 {len(groups)} 个组:")
    for group_name, indices in groups.items():
        print(f"  组名: {group_name}")
        print(f"  包含 {len(indices)} 条鱼")
        print(f"  索引范围: {indices[0]} - {indices[-1]}")
        
        # 获取第一条鱼的标点数
        if len(predictions) > 3 and indices[0] < len(predictions[3]):
            first_fish_coords = predictions[3][indices[0]]
            if hasattr(first_fish_coords, 'shape'):
                num_keypoints = first_fish_coords.shape[1] // 2
                print(f"  每条鱼标点数: {num_keypoints}")
        print()
