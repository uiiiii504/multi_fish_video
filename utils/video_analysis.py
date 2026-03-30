"""
多鱼视频分析模块
支持自定义鱼数量和标点数
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .visualization import multi_fish_umap_scatter

class MultiFishVideoAnalyzer:
    def __init__(self, vid_path, vid_name, data, predictions, num_keypoints=6, fish_group_names=None, fish_numbers=None):
        """
        初始化多鱼视频分析器
        :param vid_path: 视频路径
        :param vid_name: 视频文件名
        :param data: 多鱼坐标数据列表 [fish1_coords, fish2_coords, ...]
        :param predictions: 多鱼预测结果列表 [fish1_pred, fish2_pred, ...]
        :param num_keypoints: 每条鱼的标点数
        :param fish_group_names: 每条鱼的组名列表
        :param fish_numbers: 每条鱼在组内的编号列表（1-based）
        """
        self.vid_path = vid_path
        self.vid_name = vid_name
        self.data = data
        self.predictions = predictions
        self.num_keypoints = num_keypoints
        self.num_fish = len(data)  # 从数据中动态获取鱼的数量
        self.fish_group_names = fish_group_names if fish_group_names else ["Unknown"] * self.num_fish
        self.fish_numbers = fish_numbers if fish_numbers else [i+1 for i in range(self.num_fish)]
        
        # 扩展颜色列表以支持更多鱼
        self.colors = [
            (0, 0, 255),      # 红色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (0, 255, 255),    # 黄色
            (255, 255, 0),    # 青色
            (255, 0, 255),    # 洋红色
            (128, 0, 128),    # 紫色
            (0, 165, 255),    # 橙色
            (255, 255, 255),  # 白色
            (128, 128, 128),  # 灰色
        ]
        
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
        获取视频信息（使用OpenCV替代FFmpeg）
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频文件: {self.video_path}")
        
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
    
    def generate(self, output_path=None):
        """
        生成视频
        :param output_path: 自定义输出路径
        :return: 输出视频路径
        """
        # 打开视频
        cap = cv2.VideoCapture(self.video_path)
        
        # 创建输出视频
        if output_path is None:
            output_path = os.path.join(self.vid_path, f"{os.path.splitext(self.vid_name)[0]}_multi_fish.mp4")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # 处理每一帧
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in tqdm(range(total_frames), desc="生成视频"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 绘制多条鱼的轨迹和行为
            frame = self.draw_fish_behavior(frame, frame_idx)
            
            # 写入输出视频
            out.write(frame)
        
        # 释放资源
        cap.release()
        out.release()
        
        print(f"视频生成完成: {output_path}")
        return output_path
    
    def draw_fish_behavior(self, frame, frame_idx):
        """
        在帧上绘制鱼的行为
        :param frame: 视频帧
        :param frame_idx: 帧索引
        :return: 绘制后的帧
        """
        # 为每条鱼绘制轨迹和行为
        for i in range(self.num_fish):
            if frame_idx >= len(self.data[i]):
                continue
                
            # 获取颜色
            color = self.colors[i % len(self.colors)]
            
            # 获取当前帧的坐标数据
            coords = self.data[i][frame_idx]
            
            # 获取当前鱼的组名
            current_group_name = self.fish_group_names[i] if i < len(self.fish_group_names) else "Unknown"
            
            # 绘制所有关键点
            # 根据数据结构探测结果，我们知道这里使用步长2（无似然值）
            keypoints = self.extract_keypoints(coords, stride=2)
            
            # 获取当前鱼的编号
            current_fish_number = self.fish_numbers[i] if i < len(self.fish_numbers) else (i + 1)
            
            # 绘制边界框
            if keypoints:
                # 获取行为预测（bsoid group结果）
                behavior = self.predictions[i][frame_idx] if frame_idx < len(self.predictions[i]) else 0
                frame = self.draw_bounding_box(frame, keypoints, color, behavior, current_group_name)
            
            # 绘制关键点
            for kp_idx, (x, y) in enumerate(keypoints):
                if not (np.isnan(x) or np.isnan(y)):
                    # 绘制关键点
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                    # 标注关键点编号
                    cv2.putText(frame, str(kp_idx+1), (int(x)+5, int(y)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 连接关键点形成骨架（如果有多个点）
            if len(keypoints) > 1:
                for j in range(len(keypoints) - 1):
                    x1, y1 = keypoints[j]
                    x2, y2 = keypoints[j + 1]
                    if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            
            # 只显示前4组的标签，且使用每组的第一个鱼编号
            if i < 4:
                # 获取行为预测
                behavior = self.predictions[i][frame_idx] if frame_idx < len(self.predictions[i]) else 0
                
                # 显示组名和该组的第一个鱼编号
                # 查找该组的第一个鱼编号
                group_first_number = current_fish_number
                # 遍历所有鱼，找到同组的第一个鱼编号
                for j in range(len(self.fish_group_names)):
                    if self.fish_group_names[j] == current_group_name:
                        group_first_number = self.fish_numbers[j]
                        break
                
                # 在左上角绘制组别标签 - 增大字体，显示该组的第一个鱼编号
                label = f"{current_group_name}: {group_first_number}"
                # 字体大小从0.6增大到1.0，线宽从2增大到2（稍微缩小）
                cv2.putText(frame, label, (10, 40 + i*45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return frame
    
    def draw_bounding_box(self, frame, keypoints, color, fish_id, group_name):
        """
        绘制边界框并标注鱼信息
        :param frame: 视频帧
        :param keypoints: 关键点列表
        :param color: 颜色
        :param fish_id: 鱼的ID
        :param group_name: 组名
        :return: 绘制后的帧
        """
        if not keypoints:
            return frame
        
        # 计算边界框
        x_coords = [x for x, y in keypoints if not np.isnan(x) and not np.isnan(y)]
        y_coords = [y for x, y in keypoints if not np.isnan(x) and not np.isnan(y)]
        
        if not x_coords or not y_coords:
            return frame
        
        x_min = int(min(x_coords)) - 10
        y_min = int(min(y_coords)) - 10
        x_max = int(max(x_coords)) + 10
        y_max = int(max(y_coords)) + 10
        
        # 确保边界在画面内
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.width, x_max)
        y_max = min(self.height, y_max)
        
        # 绘制边界框
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # 标注鱼信息 - 显示组名和bsoid group结果
        label = f"{group_name}: {fish_id}"
        # 稍微缩小字体和线宽
        cv2.putText(frame, label, (x_min, y_min - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return frame
    
    def extract_keypoints(self, coords, stride=2, likelihood_threshold=0.6):
        """
        从坐标数据中提取关键点
        :param coords: 坐标数组
        :param stride: 步长 (2=无似然值, 3=有似然值)
        :param likelihood_threshold: 似然值阈值
        :return: 关键点列表 [(x1,y1), (x2,y2), ...]
        """
        keypoints = []
        
        # 计算关键点数量
        num_points = len(coords) // stride
        
        for i in range(min(num_points, self.num_keypoints)):
            if stride == 3:
                x = coords[i * 3]
                y = coords[i * 3 + 1]
                likelihood = coords[i * 3 + 2]
                
                # 似然值过滤
                if likelihood < likelihood_threshold:
                    continue
            else:
                x = coords[i * 2]
                y = coords[i * 2 + 1]
            
            # 过滤无效点
            if not (np.isnan(x) or np.isnan(y)) and x > 0 and y > 0:
                keypoints.append((x, y))
        
        return keypoints
