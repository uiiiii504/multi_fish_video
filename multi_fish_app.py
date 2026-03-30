"""
多鱼分析应用
支持自动识别组信息和自定义标点数
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
from utils.data_processing import (load_multi_fish_data, extract_fish_data, 
                                   preprocess_multi_fish, get_fish_info)
from utils.video_analysis import MultiFishVideoAnalyzer
from utils.visualization import multi_fish_umap_scatter

def main():
    st.title("多鱼行为分析工具")
    
    # 创建侧边栏
    # 使用按钮直接切换页面
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "主页"
    
    if st.sidebar.button("🏠 主页", use_container_width=True):
        st.session_state.current_page = "主页"
    
    st.sidebar.divider()
    
    st.sidebar.header("⚠️ 注意事项")
    st.sidebar.markdown("""
    - 骨架数据来自.sav文件（B-SOID处理后的数据）
    - 视频只是背景，骨架是独立绘制的
    - 系统会自动识别数据中的组信息和标点数
    """)
    
    st.sidebar.divider()
    
    if st.sidebar.button("🎬 MultiFishVideo", use_container_width=True):
        st.session_state.current_page = "MultiFishVideo"
    
    page = st.session_state.current_page
    
    if page == "主页":
        st.subheader("欢迎使用多鱼行为分析工具！")
        st.markdown("""
        **使用方法：**
        
        1. 点击左侧 "MultiFishVideo" 进入视频分析工具
        2. 输入工作目录（包含.sav文件的路径）
        3. 选择数据前缀
        4. 输入视频路径和视频文件名
        5. 在高级配置中选择要分析的鱼
        6. 点击"开始分析"生成视频
        """)
    
    elif page == "MultiFishVideo":
        st.subheader("多鱼视频分析")
        st.write("自动识别组信息，支持自定义分析")
        
        # 工作目录选择
        working_dir = st.text_input("工作目录 (请输入包含.sav文件的路径)", value="")
        
        # 前缀选择
        prefix = None
        if working_dir and os.path.exists(working_dir):
            files = os.listdir(working_dir)
            prefixes = list(set([f.split('_')[0] for f in files if f.endswith('.sav')]))
            if not prefixes:
                st.warning("工作目录中没有找到.sav文件")
            else:
                prefix = st.selectbox("选择前缀", prefixes)
        elif working_dir:
            st.error("工作目录不存在")
        
        # 视频路径选择
        st.write("**视频文件选择：**")
        vid_select_mode = st.radio(
            "选择视频方式",
            ["手动输入路径", "从文件夹选择"],
            index=0,
            horizontal=True
        )
        
        if vid_select_mode == "手动输入路径":
            vid_path = st.text_input("视频路径 (请输入包含视频文件的路径)", value="")
            vid_name = st.text_input(
                "视频文件名", 
                value="",
                help="支持的视频格式: mp4, avi, mov, wmv"
            )
        else:
            # 从文件夹选择
            vid_folder = st.text_input("视频文件夹路径", value="")
            if vid_folder and os.path.exists(vid_folder):
                video_files = [f for f in os.listdir(vid_folder) 
                              if f.lower().endswith(('.mp4', '.avi', '.mov', '.wmv'))]
                if video_files:
                    selected_video = st.selectbox("选择视频文件", video_files)
                    vid_path = vid_folder
                    vid_name = selected_video
                else:
                    st.warning("该文件夹中没有找到视频文件")
                    vid_path = ""
                    vid_name = ""
            else:
                vid_path = ""
                vid_name = ""
        
        # 自定义导出路径
        output_path = st.text_input(
            "自定义导出路径 (可选)", 
            value="",
            help="留空则使用默认路径"
        )
        
        st.info("支持的视频格式: mp4, avi, mov, wmv")
        
        # 加载数据并显示自动识别的信息
        info = None
        data = None
        predictions = None
        
        if working_dir and prefix and os.path.exists(working_dir):
            try:
                data, predictions = load_multi_fish_data(working_dir, prefix)
                info = get_fish_info(data, predictions)
                
                # 显示自动识别的信息
                st.subheader("📊 自动识别的数据信息")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总鱼数", info['total_fish_in_file'])
                with col2:
                    st.metric("组数", info['num_groups'])
                with col3:
                    st.metric("每组鱼数", info['fish_per_group'])
                
                st.metric("每条鱼标点数", info['num_keypoints'])
                
                # 显示组信息
                if info['group_names']:
                    st.write("**识别到的组：**")
                    for group_name in info['group_names']:
                        group_info = info['groups'][group_name]
                        st.write(f"- {group_name}: {group_info['count']}条鱼 (索引 {group_info['start_idx']}-{group_info['end_idx']})")
                
            except Exception as e:
                st.error(f"加载数据信息失败: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # 高级配置
        selected_groups = []
        selected_fish = []
        selected_keypoints = []
        fish_per_group = 1
        num_keypoints = 6
        video_name_display = ""
        
        if info and info['group_names']:
            with st.expander("⚙️ 高级配置"):
                
                st.write("**视频信息：**")
                fish_per_group = info['fish_per_group'] if info['fish_per_group'] > 0 else 1
                
                # 视频名称显示
                video_name_display = os.path.splitext(vid_name)[0] if vid_name else "未选择视频"
                st.write(f"当前视频: {video_name_display}")
                
                # 从视频名称中提取数字（用于默认选中）
                import re
                numbers = re.findall(r'\d+', vid_name) if vid_name else []
                default_fish_num = int(numbers[0]) if numbers else 1
                default_fish_num = max(1, min(default_fish_num, fish_per_group))
                
                st.write("**选择要分析的鱼：**")
                st.info(f"💡 系统检测到 {info['num_groups']} 个组，每组 {fish_per_group} 条鱼")
                
                group_names = info['group_names']
                selected_fish_by_group = {}
                
                # 为每个组选择鱼（支持多选）
                for group_name in group_names:
                    if group_name in info['groups']:
                        group_info = info['groups'][group_name]
                        max_fish_num = group_info['count']
                        
                        # 生成鱼编号选项列表
                        fish_options = list(range(1, max_fish_num + 1))
                        
                        # 选择该组的鱼（支持多选）
                        selected_nums = st.multiselect(
                            f"{group_name} 选择鱼编号",
                            options=fish_options,
                            default=[min(default_fish_num, max_fish_num)],
                            help=f"选择 {group_name} 组的鱼编号（可多选，例如：1,3,5,7,9）"
                        )
                        selected_fish_by_group[group_name] = selected_nums
                
                # 根据选择的鱼编号计算实际的鱼索引
                selected_fish = []
                selected_groups = list(selected_fish_by_group.keys())
                
                st.write("**已选择的鱼：**")
                for group_name, selected_nums in selected_fish_by_group.items():
                    if group_name in info['groups']:
                        group_info = info['groups'][group_name]
                        # 计算实际的鱼索引
                        group_fish = []
                        for fish_num in selected_nums:
                            fish_idx = group_info['start_idx'] + (fish_num - 1)
                            if fish_idx <= group_info['end_idx']:
                                selected_fish.append(f"鱼_{fish_idx}")
                                group_fish.append(f"{fish_num}(索引:{fish_idx})")
                        # 每个组显示一行，列出所有选择的鱼
                        if group_fish:
                            fish_list = ", ".join(group_fish)
                            st.write(f"- {group_name}: {fish_list}")
                
                st.write("**标点数设置：**")
                num_keypoints = st.number_input(
                    "标点数",
                    min_value=1,
                    max_value=info['num_keypoints'],
                    value=info['num_keypoints'],
                    help=f"每条鱼标注的关键点数量（最多{info['num_keypoints']}个）"
                )
            
            st.info(f"将分析视频 {video_name_display}，包含 {len(selected_fish)} 条鱼（来自 {len(selected_groups)} 个组）")
        
        if st.button("🚀 开始分析", type="primary"):
            try:
                # 验证输入
                if not working_dir:
                    st.error("请输入工作目录")
                    return
                if not os.path.exists(working_dir):
                    st.error("工作目录不存在")
                    return
                if not prefix:
                    st.error("请选择前缀")
                    return
                if not vid_path:
                    st.error("请输入视频路径")
                    return
                if not os.path.exists(vid_path):
                    st.error("视频路径不存在")
                    return
                if not vid_name:
                    st.error("请输入视频文件名")
                    return
                
                with st.spinner("正在处理..."):
                    # 构建鱼的索引列表
                    fish_indices = []
                    if selected_fish:
                        # 从选择的鱼编号中提取索引
                        for fish_str in selected_fish:
                            idx = int(fish_str.split('_')[1])
                            fish_indices.append(idx)
                    else:
                        # 按组选择鱼
                        for group_name in selected_groups:
                            if group_name in info['groups']:
                                group_indices = info['groups'][group_name]['indices']
                                # 只取前fish_per_group条
                                fish_indices.extend(group_indices[:fish_per_group])
                    
                    # 提取鱼的数据
                    st.info(f"提取 {len(fish_indices)} 条鱼的数据...")
                    fish_coords_list, fish_predictions_list, _ = extract_fish_data(
                        data, predictions, 
                        fish_indices=fish_indices
                    )
                    
                    st.success(f"成功提取 {len(fish_coords_list)} 条鱼的数据")
                    
                    # 预处理数据
                    st.info("预处理数据...")
                    processed_coords, processed_predictions = preprocess_multi_fish(
                        fish_coords_list, fish_predictions_list
                    )
                    
                    # 视频分析
                    st.info("生成视频...")
                    
                    # 构建每条鱼的组名列表和编号列表
                    fish_group_names = []
                    fish_numbers = []  # 每条鱼在组内的编号（1-based）
                    for idx in fish_indices:
                        # 找到这个鱼索引对应的组名和编号
                        fish_group = "Unknown"
                        fish_num = 1
                        for group_name in info['group_names']:
                            if group_name in info['groups']:
                                group_indices = info['groups'][group_name]['indices']
                                if idx in group_indices:
                                    fish_group = group_name
                                    # 计算组内编号
                                    fish_num = group_indices.index(idx) + 1
                                    break
                        fish_group_names.append(fish_group)
                        fish_numbers.append(fish_num)
                    
                    # 显示选择的鱼数量
                    st.info(f"共选择 {len(fish_indices)} 条鱼进行分析")
                    
                    # 构建自定义导出路径
                    custom_output_path = None
                    if output_path:
                        # 构建输出文件名: 视频名称+组别（简洁格式）
                        group_suffix = "_".join(selected_groups)
                        # 使用视频名称作为文件名
                        output_filename = f"{video_name_display}_{group_suffix}.mp4"
                        custom_output_path = os.path.join(output_path, output_filename)
                    
                    analyzer = MultiFishVideoAnalyzer(
                        vid_path, vid_name, 
                        processed_coords, processed_predictions,
                        num_keypoints=num_keypoints,
                        fish_group_names=fish_group_names,
                        fish_numbers=fish_numbers
                    )
                    output_path = analyzer.generate(
                        output_path=custom_output_path
                    )
                    
                    st.success(f"✅ 分析完成！视频保存至: {output_path}")
                    
            except Exception as e:
                st.error(f"❌ 分析过程中出错: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
