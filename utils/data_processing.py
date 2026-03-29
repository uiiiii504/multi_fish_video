"""
多鱼数据处理模块
"""

import os
import numpy as np
import joblib

def load_multi_fish_data(working_dir, prefix):
    """
    加载多条鱼的数据
    :param working_dir: 工作目录
    :param prefix: 前缀
    :return: 数据和预测结果
    """
    # 加载数据
    data_path = os.path.join(working_dir, f"{prefix}_data.sav")
    predictions_path = os.path.join(working_dir, f"{prefix}_predictions.sav")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"预测文件不存在: {predictions_path}")
    
    with open(data_path, 'rb') as f:
        data = joblib.load(f)
    
    with open(predictions_path, 'rb') as f:
        predictions = joblib.load(f)
    
    return data, predictions

def preprocess_multi_fish(data):
    """
    预处理多条鱼的数据
    :param data: 原始数据
    :return: 预处理后的数据
    """
    # 假设数据结构为 [鱼1数据, 鱼2数据, 鱼3数据, 鱼4数据]
    # 这里需要根据实际数据结构进行调整
    processed_data = []
    
    for fish_data in data:
        # 处理NaN值
        fish_data = np.nan_to_num(fish_data, nan=0.0, posinf=0.0, neginf=0.0)
        processed_data.append(fish_data)
    
    return processed_data

def split_fish_data(data):
    """
    将数据分割为四条鱼的数据
    :param data: 原始数据
    :return: 四条鱼的数据列表
    """
    # 这里需要根据实际数据结构进行调整
    # 假设数据是一个数组，每列代表不同的鱼
    fish_data = []
    
    # 示例：假设每4列对应一条鱼
    for i in range(4):
        fish_data.append(data[:, i::4])
    
    return fish_data