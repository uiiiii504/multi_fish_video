"""
多鱼分析配置文件
"""

# 视频配置
VIDEO_FPS = 30
VIDEO_CODEC = 'mp4v'

# 数据配置
NUM_FISH = 4
FISH_COLORS = ['r', 'g', 'b', 'y']

# UMAP配置
UMAP_N_NEIGHBORS = 60
UMAP_N_COMPONENTS = 2

# 行为分类配置
NUM_BEHAVIORS = 12
BEHAVIOR_LABELS = [f'Group {i}' for i in range(NUM_BEHAVIORS)]

# 目录配置
OUTPUT_DIR = 'output'
FIGURE_DIR = 'figures'