"""
数据结构探测工具
用于分析B-SOiD导出的.sav文件结构
"""

import os
import numpy as np
import joblib

def profile_data(working_dir, prefix):
    """
    分析数据结构
    :param working_dir: 工作目录
    :param prefix: 前缀
    """
    print("=== 数据结构探测开始 ===")
    
    # 加载数据
    data_path = os.path.join(working_dir, f"{prefix}_data.sav")
    predictions_path = os.path.join(working_dir, f"{prefix}_predictions.sav")
    
    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return
    
    if not os.path.exists(predictions_path):
        print(f"预测文件不存在: {predictions_path}")
        return
    
    print(f"加载数据文件: {data_path}")
    with open(data_path, 'rb') as f:
        data = joblib.load(f)
    
    print(f"加载预测文件: {predictions_path}")
    with open(predictions_path, 'rb') as f:
        predictions = joblib.load(f)
    
    print("\n=== 数据结构分析 ===")
    print(f"data类型: {type(data)}")
    if isinstance(data, (list, np.ndarray)):
        print(f"data长度: {len(data)}")
    
    print(f"\npredictions类型: {type(predictions)}")
    if isinstance(predictions, list):
        print(f"predictions长度: {len(predictions)}")
        
        # 分析每个元素
        for i, item in enumerate(predictions):
            print(f"\npredictions[{i}]:")
            print(f"  类型: {type(item)}")
            
            if isinstance(item, (list, np.ndarray)):
                if len(item) > 0:
                    print(f"  长度: {len(item)}")
                    print(f"  第一个元素类型: {type(item[0])}")
                    
                    # 检查是否为坐标数据
                    if hasattr(item[0], 'shape'):
                        print(f"  第一个元素形状: {item[0].shape}")
                    elif isinstance(item[0], (list, np.ndarray)):
                        print(f"  第一个元素长度: {len(item[0])}")
                        if len(item[0]) > 0:
                            print(f"  第一个元素的第一个值: {item[0][0]}")
                            print(f"  第一个元素的前10个值: {item[0][:10]}")
                            
                            # 分析数值范围
                            values = item[0]
                            if isinstance(values, np.ndarray):
                                values = values.tolist()
                            if values:
                                min_val = min(values)
                                max_val = max(values)
                                print(f"  数值范围: {min_val} - {max_val}")
                                print(f"  平均值: {sum(values)/len(values)}")
                                
                                # 检测是否包含似然值（0-1之间）
                                likelihood_count = sum(1 for v in values if 0 <= v <= 1)
                                print(f"  0-1之间的值数量: {likelihood_count}")
                                print(f"  0-1之间的值比例: {likelihood_count/len(values):.2f}")
    
    # 重点分析predictions[3]（坐标数据）
    if len(predictions) > 3:
        print("\n=== 重点分析 predictions[3]（坐标数据）===")
        coords_data = predictions[3]
        print(f"类型: {type(coords_data)}")
        
        if isinstance(coords_data, (list, np.ndarray)):
            print(f"长度: {len(coords_data)}")
            
            if len(coords_data) > 0:
                first_item = coords_data[0]
                print(f"第一个元素类型: {type(first_item)}")
                
                if hasattr(first_item, 'shape'):
                    print(f"第一个元素形状: {first_item.shape}")
                    
                    # 分析第一帧数据
                    if first_item.shape[0] > 0:
                        first_frame = first_item[0]
                        print(f"第一帧数据类型: {type(first_frame)}")
                        print(f"第一帧数据长度: {len(first_frame)}")
                        print(f"第一帧数据: {first_frame}")
                        
                        # 分析数值范围
                        min_val = np.min(first_frame)
                        max_val = np.max(first_frame)
                        print(f"数值范围: {min_val} - {max_val}")
                        print(f"平均值: {np.mean(first_frame)}")
                        
                        # 检查是否包含似然值（0-1之间）
                        likelihood_count = np.sum((first_frame >= 0) & (first_frame <= 1))
                        print(f"0-1之间的值数量: {likelihood_count}")
                        print(f"0-1之间的值比例: {likelihood_count/len(first_frame):.2f}")
                
                # 分析第二条鱼
                if len(coords_data) > 1:
                    second_fish = coords_data[1]
                    print(f"\n第二条鱼数据形状: {second_fish.shape}")
                    if second_fish.shape[0] > 0:
                        second_frame = second_fish[0]
                        print(f"第二条鱼第一帧数据: {second_frame}")
    
    print("\n=== 数据结构探测完成 ===")

if __name__ == "__main__":
    # 示例使用 - 直接指定路径
    working_dir = r"D:\study\gradu_pro\tongji\test\video"
    prefix = "Mar-28-2026"
    print(f"使用工作目录: {working_dir}")
    print(f"使用前缀: {prefix}")
    profile_data(working_dir, prefix)
