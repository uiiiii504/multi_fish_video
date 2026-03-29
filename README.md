# 多鱼行为分析工具

## 项目简介

这个工具是基于B-SOID的多鱼行为分析扩展，用于分析和可视化多条鱼的行为。

## 功能特点

- 支持动态数量的鱼（根据输入数据自动调整）
- 生成包含多条鱼行为标注的同步视频
- 当鱼的数量超过8时会给出性能警告
- 提供丰富的颜色区分不同的鱼
- 直接在原始视频帧上绘制鱼的位置和行为标签

## 目录结构

```
multi_fish_video/
├── multi_fish_app.py        # 主应用文件
├── utils/
│   ├── data_processing.py   # 数据处理模块
│   ├── video_analysis.py    # 视频分析模块
│   └── visualization.py     # 可视化模块
├── config/
│   └── config.py            # 配置文件
└── README.md                # 说明文件
```

## 环境要求

- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Matplotlib
- OpenCV
- ffmpeg
- joblib
- tqdm

## 安装依赖

```bash
pip install streamlit numpy pandas matplotlib opencv-python ffmpeg-python joblib tqdm
```

## 使用方法

1. **准备数据**
   - 确保在工作目录中有B-SOID生成的数据文件（如`prefix_data.sav`和`prefix_predictions.sav`）
   - 数据文件格式应为 `[鱼1数据, 鱼2数据, ...]`，支持任意数量的鱼
   - 确保视频文件存在
2. **运行应用**
   ```bash
   cd multi_fish_video
   streamlit run multi_fish_app.py
   ```
3. **配置参数**
   - 输入工作目录路径
   - 选择数据前缀
   - 输入视频路径和文件名
4. **开始分析**
   - 点击"开始分析"按钮
   - 等待分析完成
   - 查看生成的视频结果

## B-SOID数据生成SOP

### 文件说明

对于每一个4缸视频（比如AA1-AA4），需要以下两类.sav文件：

| 文件类型 | B-SOID生成的文件名示例               | 作用                         |
| ---- | ---------------------------- | -------------------------- |
| 轨迹数据 | Mar-28-2026\_data.sav        | 提供4条鱼的物理坐标(x, y)           |
| 预测标签 | Mar-28-2026\_predictions.sav | 提供每一帧对应的行为ID（如0代表静止，1代表游动） |

### 使用B-SOID生成数据文件的步骤

#### 第一步：数据加载 (Loading)

1. 打开B-SOID应用
2. 进入 **Preprocessing** 界面
3. 同时选中对应同一个视频的4个CSV文件（比如AA1.csv到AA4.csv）
4. 注意：如果想一次性处理所有76个文件，也可以全部选中
5. B-SOID会把这些文件作为一个整体进行处理

#### 第二步：特征提取与聚类 (Feature & Clustering)

1. 运行 **Extract Features**
2. 运行 **Run UMAP**
3. 运行 **Run HDBSCAN**
4. 这一步会决定鱼有多少种动作（Group 0, 1, 2...）

#### 第三步：模型训练 (Training)

1. 运行 **Train Random Forest**
2. 这一步是让AI学习如何通过特征识别动作

#### 第四步：导出关键文件 (Exporting) —— 最关键的一步

1. 在B-SOID界面找到 **Export** 或 **Save Analysis** 按钮
2. 点击导出分析结果
3. 生成的文件夹里会出现：
   - `_data.sav`：包含了刚才导入的所有鱼的坐标
   - `_predictions.sav`：包含了AI对这些鱼的分类预测结果

### 注意事项

- 不能直接从CSV文件"变"出这些文件，必须走完B-SOID的完整分析流程
- 确保CSV文件的格式与B-SOID要求一致
- 数据文件名的前缀（如Mar-28-2026）将用于后续的多鱼分析

## 数据格式要求

### 输入数据

- **数据文件** (`prefix_data.sav`): 包含多条鱼的坐标数据，格式为 `[鱼1数据, 鱼2数据, ...]`
- **预测文件** (`prefix_predictions.sav`): 包含多条鱼的行为预测结果，格式为 `[鱼1预测, 鱼2预测, ...]`

### 输出结果

- **视频文件**: 在视频目录生成包含多条鱼行为标注的视频

## 注意事项

1. 确保视频文件与数据文件的帧速率匹配
2. 如遇内存不足问题，考虑增加虚拟内存或使用更小的数据集
3. 鱼的数量超过8条时，可能会导致渲染缓慢

## 故障排除

- **视频文件不存在**: 检查视频路径和文件名是否正确
- **数据文件不存在**: 确保工作目录中有正确的数据文件
- **内存不足**: 增加虚拟内存或使用更小的数据集
- **FFmpeg错误**: 确保FFmpeg已正确安装并添加到系统路径

## 联系方式

如有问题，请联系开发者。
