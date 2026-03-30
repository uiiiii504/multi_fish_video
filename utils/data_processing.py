"""
多鱼数据处理模块
支持自动识别组信息和标点数
"""

import os
import re
import numpy as np
import joblib


def load_multi_fish_data(working_dir, prefix):
    """
    加载多条鱼的数据
    :param working_dir: 工作目录
    :param prefix: 前缀
    :return: data (B-SOID data文件内容), predictions (B-SOID predictions文件内容)
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


def analyze_groups(predictions):
    """
    自动分析数据中的组信息
    :param predictions: B-SOID predictions数据
    :return: 组信息字典
    """
    groups_info = {
        'total_fish': 0,
        'num_groups': 0,
        'group_names': [],
        'groups': {},  # {组名: {start_idx, end_idx, count}}
        'fish_per_group': 0,
        'num_keypoints': 0
    }
    
    # 从predictions[1]或predictions[2]获取组信息
    # predictions[1]是组标签列表，predictions[2]是文件路径列表
    
    if len(predictions) > 2 and isinstance(predictions[2], list):
        # 使用文件路径提取组名
        filepaths = predictions[2]
        groups_info['total_fish'] = len(filepaths)
        
        # 分析每个文件的组名
        for i, filepath in enumerate(filepaths):
            # 从路径中提取组名
            parts = filepath.replace('\\', '/').split('/')
            
            # 尝试多种方式提取组名
            group_name = None
            
            # 方式1: 从倒数第二个目录名提取
            if len(parts) >= 2:
                dir_name = parts[-2]
                # 清理组名（移除特殊字符）
                group_name = re.sub(r'^[\\/]+', '', dir_name)
            
            # 方式2: 从文件名提取
            if not group_name and len(parts) >= 1:
                filename = parts[-1]
                # 尝试从文件名中提取组名（如 AA1_control.csv -> control）
                match = re.search(r'_([a-zA-Z0-9]+)\.csv$', filename)
                if match:
                    group_name = match.group(1)
            
            if not group_name:
                group_name = f"Group_{i}"
            
            # 记录组信息
            if group_name not in groups_info['groups']:
                groups_info['groups'][group_name] = {
                    'start_idx': i,
                    'end_idx': i,
                    'count': 0,
                    'indices': []
                }
                groups_info['group_names'].append(group_name)
            
            groups_info['groups'][group_name]['end_idx'] = i
            groups_info['groups'][group_name]['count'] += 1
            groups_info['groups'][group_name]['indices'].append(i)
    
    elif len(predictions) > 1 and isinstance(predictions[1], list):
        # 使用组标签列表
        group_labels = predictions[1]
        groups_info['total_fish'] = len(group_labels)
        
        for i, label in enumerate(group_labels):
            group_name = str(label).strip('/\\')
            
            if group_name not in groups_info['groups']:
                groups_info['groups'][group_name] = {
                    'start_idx': i,
                    'end_idx': i,
                    'count': 0,
                    'indices': []
                }
                groups_info['group_names'].append(group_name)
            
            groups_info['groups'][group_name]['end_idx'] = i
            groups_info['groups'][group_name]['count'] += 1
            groups_info['groups'][group_name]['indices'].append(i)
    
    groups_info['num_groups'] = len(groups_info['group_names'])
    
    # 计算每组的鱼数量
    if groups_info['groups']:
        first_group = list(groups_info['groups'].values())[0]
        groups_info['fish_per_group'] = first_group['count']
    
    # 自动检测标点数
    if len(predictions) > 3 and groups_info['total_fish'] > 0:
        first_fish_coords = predictions[3][0]
        if hasattr(first_fish_coords, 'shape'):
            num_columns = first_fish_coords.shape[1]
            groups_info['num_keypoints'] = num_columns // 2  # 每2列是一个点的x,y坐标
        elif isinstance(first_fish_coords, (list, np.ndarray)):
            num_columns = len(first_fish_coords[0]) if len(first_fish_coords) > 0 else 0
            groups_info['num_keypoints'] = num_columns // 2
    
    return groups_info


def extract_fish_data(data, predictions, selected_groups=None, fish_indices=None):
    """
    从B-SOID输出中提取鱼的数据
    
    :param data: B-SOID data.sav加载的数据
    :param predictions: B-SOID predictions.sav加载的数据
    :param selected_groups: 选择的组名列表，None则使用所有组
    :param fish_indices: 指定鱼的索引列表，None则根据组自动选择
    :return: fish_coords_list, fish_predictions_list, groups_info
    """
    # 分析组信息
    groups_info = analyze_groups(predictions)
    
    # 确定要提取的鱼的索引
    if fish_indices is not None:
        # 使用指定的索引
        selected_indices = fish_indices
    elif selected_groups is not None:
        # 根据选择的组提取索引
        selected_indices = []
        for group_name in selected_groups:
            if group_name in groups_info['groups']:
                selected_indices.extend(groups_info['groups'][group_name]['indices'])
    else:
        # 使用所有鱼
        selected_indices = list(range(groups_info['total_fish']))
    
    # 获取坐标数据 (predictions[3]是坐标列表)
    if len(predictions) > 3:
        all_coords = predictions[3]
    else:
        raise ValueError("predictions文件中没有坐标数据（索引3）")
    
    # 获取行为预测数据 (predictions[4]是预测列表)
    if len(predictions) > 4:
        all_predictions = predictions[4]
    else:
        raise ValueError("predictions文件中没有预测数据（索引4）")
    
    # 提取选中的鱼的数据
    fish_coords_list = []
    fish_predictions_list = []
    
    for i in selected_indices:
        if i >= len(all_coords):
            continue
            
        # 获取第i条鱼的坐标数据
        coords = all_coords[i]
        
        # 如果是DataFrame，转换为numpy数组
        if hasattr(coords, 'values'):
            coords = coords.values
        
        # 确保是numpy数组
        coords = np.array(coords)
        fish_coords_list.append(coords)
        
        # 获取第i条鱼的预测数据
        if i < len(all_predictions):
            pred = all_predictions[i]
            if hasattr(pred, 'values'):
                pred = pred.values
            fish_predictions_list.append(np.array(pred))
        else:
            # 如果没有预测数据，创建空数组
            fish_predictions_list.append(np.zeros(len(coords)))
    
    return fish_coords_list, fish_predictions_list, groups_info


def preprocess_multi_fish(fish_coords_list, fish_predictions_list):
    """
    预处理多条鱼的数据
    :param fish_coords_list: 鱼的坐标数据列表
    :param fish_predictions_list: 鱼的预测数据列表
    :return: 处理后的坐标数据和预测数据
    """
    processed_coords = []
    processed_predictions = []
    
    for coords in fish_coords_list:
        # 处理NaN值
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        processed_coords.append(coords)
    
    for pred in fish_predictions_list:
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        processed_predictions.append(pred)
    
    return processed_coords, processed_predictions


def get_fish_info(data, predictions):
    """
    获取数据中的鱼的信息（兼容旧版本）
    :param data: B-SOID data
    :param predictions: B-SOID predictions
    :return: 字典包含鱼的数量、帧数、标点数等信息
    """
    groups_info = analyze_groups(predictions)
    
    info = {
        'total_fish_in_file': groups_info['total_fish'],
        'num_groups': groups_info['num_groups'],
        'group_names': groups_info['group_names'],
        'groups': groups_info['groups'],
        'fish_per_group': groups_info['fish_per_group'],
        'num_keypoints': groups_info['num_keypoints'],
        'frames_per_fish': [],
        'behavior_labels': None
    }
    
    # 获取每条鱼的帧数
    if len(predictions) > 3:
        for coords in predictions[3]:
            if hasattr(coords, 'shape'):
                info['frames_per_fish'].append(coords.shape[0])
            elif hasattr(coords, '__len__'):
                info['frames_per_fish'].append(len(coords))
    
    # 从predictions[0]获取行为标签
    if len(predictions) > 0 and isinstance(predictions[0], list):
        info['behavior_labels'] = predictions[0]
    
    return info
