"""
多鱼可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt

def multi_fish_umap_scatter(umap_data, predictions, num_fish=4):
    """
    生成多条鱼的UMAP散点图
    :param umap_data: UMAP嵌入数据
    :param predictions: 预测结果
    :param num_fish: 鱼的数量
    :return: 图形
    """
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', 'purple', 'orange']  # 支持最多10条鱼
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(num_fish):
        if i < len(umap_data):
            fish_umap = umap_data[i]
            fish_preds = predictions[i]
            
            # 检查数据是否为空
            if len(fish_umap) > 0:
                umap_x = fish_umap[:, 0]
                umap_y = fish_umap[:, 1]
                
                # 检查数据是否有效
                if len(umap_x) > 0 and len(umap_y) > 0:
                    # 绘制散点图
                    scatter = ax.scatter(umap_x, umap_y, c=fish_preds, cmap='tab20',
                                        alpha=0.5, label=f'Fish {i+1}')
    
    # 设置图例
    ax.legend()
    
    # 设置坐标轴范围
    all_x = []
    all_y = []
    for i in range(num_fish):
        if i < len(umap_data):
            fish_umap = umap_data[i]
            if len(fish_umap) > 0:
                all_x.extend(fish_umap[:, 0])
                all_y.extend(fish_umap[:, 1])
    
    if all_x and all_y:
        x_min, x_max = min(all_x) - 0.2, max(all_x) + 0.2
        y_min, y_max = min(all_y) - 0.2, max(all_y) + 0.2
        ax.axis([x_min, x_max, y_min, y_max])
    
    ax.set_title('Multi-Fish UMAP Scatter Plot')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    return fig

def plot_multi_fish_behavior(data, predictions, num_fish=4):
    """
    绘制多条鱼的行为序列
    :param data: 数据
    :param predictions: 预测结果
    :param num_fish: 鱼的数量
    :return: 图形
    """
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', 'purple', 'orange']  # 支持最多10条鱼
    
    fig, axes = plt.subplots(num_fish, 1, figsize=(15, 2*num_fish))
    
    for i in range(num_fish):
        if i < len(predictions):
            fish_preds = predictions[i]
            
            if len(fish_preds) > 0:
                # 使用循环颜色索引，确保不会越界
                color_idx = i % len(colors)
                axes[i].plot(fish_preds, color=colors[color_idx])
                axes[i].set_title(f'Fish {i+1} Behavior Sequence')
                axes[i].set_ylabel('Behavior')
                axes[i].set_xlabel('Frame')
    
    plt.tight_layout()
    
    return fig