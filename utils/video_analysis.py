"""
多鱼视频分析模块
"""

import os
import cv2
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .visualization import multi_fish_umap_scatter

class MultiFishVideoAnalyzer:
    def __init__(self, vid_path, vid_name, data, predictions):
        """
        初始化多鱼视频分析器
        :param vid_path: 视频路径
        :param vid_name: 视频文件名
        :param data: 多鱼数据
        :param predictions: 预测结果
        """
        self.vid_path = vid_path
        self.vid_name = vid_name
        self.data = data
        self.predictions = predictions
        self.num_fish = len(data)  # 从数据中动态获取鱼的数量
        
        # 扩展颜色列表以支持更多鱼
        self.colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', 'purple', 'orange']  # 支持最多10条鱼
        
        # 检查鱼的数量是否过多
        if self.num_fish > 8:
            print("警告：主体过多可能导致渲染缓慢")
        
        # 检查视频文件
        self.video_path = os.path.join(vid_path, vid_name)
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        
        # 获取视频信息
        self.get_video_info()
    
    def get_video_info(self):
        """
        获取视频信息
        """
        probe = ffmpeg.probe(self.video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        self.width = int(video_stream['width'])
        self.height = int(video_stream['height'])
        self.fps = eval(video_stream['r_frame_rate'])
    
    def generate(self):
        """
        生成视频
        """
        # 打开视频
        cap = cv2.VideoCapture(self.video_path)
        
        # 创建输出视频
        output_path = os.path.join(self.vid_path, f"{os.path.splitext(self.vid_name)[0]}_multi_fish.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # 处理每一帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 绘制四条鱼的轨迹和行为
            frame = self.draw_fish_behavior(frame, frame_idx)
            
            # 写入输出视频
            out.write(frame)
        
        # 释放资源
        cap.release()
        out.release()
        
        print(f"视频生成完成: {output_path}")
    
    def draw_fish_behavior(self, frame, frame_idx):
        """
        在帧上绘制鱼的行为
        :param frame: 视频帧
        :param frame_idx: 帧索引
        :return: 绘制后的帧
        """
        # 为每条鱼绘制轨迹和行为
        for i in range(self.num_fish):
            if frame_idx < len(self.data[i]):
                # 获取鱼的坐标
                x, y = self.data[i][frame_idx, 0], self.data[i][frame_idx, 1]
                
                # 获取行为预测
                behavior = self.predictions[i][frame_idx] if frame_idx < len(self.predictions[i]) else 0
                
                # 绘制鱼的位置
                color_idx = i % len(self.colors)
                color = self.colors[color_idx]
                cv2.circle(frame, (int(x), int(y)), 5, self.get_color(color), 2)
                
                # 绘制行为标签
                cv2.putText(frame, f"Fish {i+1}: {behavior}", (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.get_color(color), 2)
        
        return frame
    
    def get_color(self, color_name):
        """
        获取颜色的BGR值
        :param color_name: 颜色名称
        :return: BGR颜色值
        """
        color_map = {
            'r': (0, 0, 255),  # 红色
            'g': (0, 255, 0),  # 绿色
            'b': (255, 0, 0),  # 蓝色
            'y': (0, 255, 255),  # 黄色
            'c': (255, 255, 0),  # 青色
            'm': (255, 0, 255),  # 洋红色
            'k': (0, 0, 0),  # 黑色
            'w': (255, 255, 255),  # 白色
            'purple': (128, 0, 128),  # 紫色
            'orange': (0, 165, 255)  # 橙色
        }
        return color_map.get(color_name, (255, 255, 255))