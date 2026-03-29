"""
多鱼分析应用
适配四条鱼的行为分析工具
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_processing import load_multi_fish_data, preprocess_multi_fish
from utils.video_analysis import MultiFishVideoAnalyzer
from utils.visualization import multi_fish_umap_scatter

def main():
    st.title("多鱼行为分析工具")
    st.write("适配四条鱼的行为分析和视频生成")
    
    # 工作目录选择
    working_dir = st.text_input("工作目录", value="D:\\study\\gradu_pro\\tongji\\motocsv\\output")
    
    # 前缀选择
    if os.path.exists(working_dir):
        files = os.listdir(working_dir)
        prefixes = list(set([f.split('_')[0] for f in files if f.endswith('.sav')]))
        prefix = st.selectbox("选择前缀", prefixes)
    else:
        st.error("工作目录不存在")
        return
    
    # 视频路径
    vid_path = st.text_input("视频路径", value="D:\\study\\gradu_pro\\tongji\\motocsv")
    vid_name = st.text_input("视频文件名", value="video.mp4")
    
    if st.button("开始分析"):
        try:
            # 加载数据
            st.info("加载数据...")
            data, predictions = load_multi_fish_data(working_dir, prefix)
            
            # 预处理数据
            st.info("预处理数据...")
            processed_data = preprocess_multi_fish(data)
            
            # 视频分析
            st.info("生成视频...")
            analyzer = MultiFishVideoAnalyzer(vid_path, vid_name, processed_data, predictions)
            analyzer.generate()
            
            st.success("分析完成！")
        except Exception as e:
            st.error(f"分析过程中出错: {str(e)}")

if __name__ == "__main__":
    main()