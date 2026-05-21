# 任务二 REFUGE 视盘视杯分割报告

> 这份文档用于后续制作 PPT，因此优先组织“图片 + 表格 + 关键结论”。每个小节都给出了建议插图、建议表格和训练后待补充内容。

## 一、Task 2.1 认识数据集

### 1. 数据集加载与路径说明
本项目使用本地目录 `REFUGE/{train,val,test}`。其中：
- `train/Images` 和 `train/gts` 为训练集图像与标签
- `val/Images` 和 `val/gts` 为验证集图像与标签
- `test/Images` 只有图像，没有公开标签

对应代码：
- `src/refuge_seg/datasets/refuge_dataset.py`

建议 PPT 图片：
- `outputs/figures/task2_dataset_overview.png`
- `outputs/figures/task2_dataset_stats.png`
- `outputs/figures/task2_label_sanity.png`

生成命令：
```bash
bash scripts/plot_task2_dataset.sh
```

该脚本会同时生成：
- `outputs/figures/task2_dataset_overview.png`
- `outputs/figures/task2_dataset_stats.png`
- `outputs/figures/task2_structure_example.png`

额外建议生成答辩诊断图：
```bash
bash scripts/plot_task2_presentation.sh
```

其中 `outputs/figures/task2_label_sanity.png` 用于展示原始 mask 灰度值、自动推断后的类别映射，以及背景/视盘环/视杯像素占比。这张图适合放在数据集介绍之后，用来说明训练前必须检查标签编码，否则会出现“指标虚高但可视化错误”的问题。

本地已确认的数据划分如下：

| 数据划分 | 图像数 | 标签数 |
|---|---:|---:|
| train | 400 | 400 |
| val | 400 | 400 |
| test | 400 | 0 |

建议直接在 PPT 中用一页展示数据规模，并强调 `test` 集无标签，因此模型选择和参数调整主要依赖 `val` 集结果。

![REFUGE 数据划分与标签统计](../outputs/figures/task2_dataset_stats.png)

### 2. 图像与标签展示
建议在 PPT 中直接放 4 到 6 组典型样本，说明原图、标签图和视盘/视杯区域。

建议插图：
- 图 2-1 数据集样本总览：`outputs/figures/task2_dataset_overview.png`
- 图 2-2 结构示意图：`outputs/figures/task2_structure_example.png`

可直接放入文档/PPT：

![REFUGE 样本总览](../outputs/figures/task2_dataset_overview.png)

![视盘与视杯结构示意](../outputs/figures/task2_structure_example.png)

当前可直接写入的观察结论：
- 视盘区域整体比视杯更大，且边界通常较清晰。
- 视杯位于视盘内部，二者存在明确的包含关系。
- 目标区域在整张图中的占比很小，这会带来明显的前景/背景不平衡问题。

### 3. 视盘与视杯的直观特点
- 视盘是眼底图中较大的亮色区域，血管在该区域汇聚。
- 视杯位于视盘内部，是更小、更中心的区域。
- 从拓扑结构看，正常情况下视杯应当被视盘完全包含。

本项目会在训练前自动推断原始标签灰度值和训练类别之间的映射。以本地 REFUGE 数据为例，推断结果为：
- `255 -> 0`：背景
- `128 -> 1`：视盘环区
- `0 -> 2`：视杯

在前 50 张训练标签中，像素占比约为：

| 类别 | 原始灰度值 | 语义 | 像素占比 |
|---|---:|---|---:|
| class 0 | 255 | 背景 | 98.29% |
| class 1 | 128 | 视盘环区 | 0.99% |
| class 2 | 0 | 视杯 | 0.72% |

这说明：
- 数据集存在极强的像素级类别不平衡。
- 单独使用像素级 `CrossEntropy` 时，模型容易偏向背景类。
- `Dice`、`Dice+CE` 和拓扑约束设计都有明确必要性。

建议 PPT 要点：
- 用 1 页图说明“视杯在视盘内部”
- 为后续 optional task 的“拓扑约束”埋下逻辑基础
- 用 `task2_label_sanity.png` 说明标签编码和类别不平衡，这是答辩时解释多次训练异常的关键证据

## 二、Task 2.2 网络调研和选择

### 1. 主流分割网络简述
- `FCN`：较早的像素级预测框架，结构简单
- `U-Net`：编码器-解码器结构，带跳跃连接，医学分割常用基线
- `DeepLabV3+`：空洞卷积加强上下文建模
- `UNet++`：多层级跳跃连接，更强调多尺度融合
- `Attention U-Net`：对 skip feature 做注意力筛选

### 2. 本项目选型
本项目基线模型使用 `U-Net`，可选拓扑约束实验使用 `Attention U-Net`。

原因：
- REFUGE 样本规模不大，`U-Net` 更稳健
- 视盘和视杯边界需要细节恢复，跳跃连接有效
- `Attention U-Net` 适合做可选任务对比，突出“结构感知”能力

对应代码：
- `src/refuge_seg/models/unet.py`
- `src/refuge_seg/models/attention_unet.py`

建议 PPT 配图：
- 可手动画一个 U-Net 结构示意图
- 或插入训练输出图说明“基线模型 + 改进模型”的关系

## 三、Task 2.3 AI 辅助编写网络代码与损失函数对比

### 1. 分割任务与分类任务的区别
- 分类任务输出维度是 `[B, C]`
- 分割任务输出维度是 `[B, C, H, W]`
- 分类预测的是整张图的类别
- 分割预测的是每个像素的类别
- 分割更依赖区域重叠指标，如 `Dice` 和 `IoU`

### 2. 损失函数设计
本项目实现了以下损失函数：
- `CrossEntropyLoss`
- `DiceLoss`
- `Dice + CrossEntropy`
- `TopologyAwareLoss`（用于 optional task）

对应代码：
- `src/refuge_seg/losses/segmentation_losses.py`

### 3. 评价指标
本项目输出：
- `Dice_disc`
- `Dice_cup`
- `IoU_disc`
- `IoU_cup`
- `Mean Dice`
- `Mean IoU`

对应代码：
- `src/refuge_seg/utils/metrics.py`

### 4. 损失函数对比实验
建议至少做下面几组：

| 实验编号 | 模型 | 损失函数 | 是否后处理 | Mean Dice | Mean IoU | 备注 |
|---|---|---|---|---:|---:|---|
| E1 | U-Net | CE | 否 | `待补充` | `待补充` | |
| E2 | U-Net | Dice | 否 | `待补充` | `待补充` | |
| E3 | U-Net | Dice+CE | 否 | `待补充` | `待补充` | 推荐主结果 |
| E4 | U-Net | Dice+CE | 是 | `待补充` | `待补充` | 与 optional task 衔接 |

建议 PPT 图片：
- `outputs/figures/task2_training_curves.png`
- 若保存了典型预测图，也建议插入一页做 CE / Dice / Dice+CE 可视化比较
- `outputs/figures/task2_prediction_diagnostics.png`，用于检查预测面积分布是否塌缩成单类或固定模板

训练后运行汇总图：
```bash
bash scripts/plot_task2_results.sh
```

训练后补充分析：
- `[CE 为什么稳定，但对小目标 cup 不一定最好]`
- `[Dice 为什么更关注区域重叠]`
- `[Dice+CE 为什么通常更平衡]`

## 四、Task 2.4 模型参数调整与优化

### 1. 可调参数
- 学习率 `lr`
- 批次大小 `batch size`
- 优化器 `Adam / SGD`
- 学习率调度器 `scheduler`
- 训练轮数 `epochs`
- 输入尺寸 `image_size`
- 基础通道数 `base_channels`
- 拓扑惩罚系数 `lambda_topology`

配置文件：
- `configs/task2_refuge_baseline.yaml`
- `configs/task2_refuge_topology.yaml`

### 2. 建议调参表

| 参数 | 取值 | 单次训练时长 | Mean Dice | 现象记录 |
|---|---|---:|---:|---|
| lr | `3e-4` | `待补充` | `待补充` | baseline |
| lr | `1e-4` | `待补充` | `待补充` | |
| batch size | `4` | `待补充` | `待补充` | baseline |
| batch size | `8` | `待补充` | `待补充` | |
| scheduler | `cosine` | `待补充` | `待补充` | baseline |
| scheduler | `step` | `待补充` | `待补充` | |

建议 PPT 呈现：
- 1 页表格总结
- 1 页曲线图展示收敛差异

训练后补充分析：
- `[学习率过大/过小各自的影响]`
- `[batch size 对稳定性和显存的影响]`
- `[scheduler 对后期收敛的帮助]`

## 五、Task 2.5 Optional Task

### 5.1 分割结果的评估和后处理

#### 1. 孔洞检查
若视盘或视杯预测区域内部出现空洞，通常说明边界置信度不足，或出现局部断裂。

#### 2. 连通域检查
正常情况下，一张眼底图只应有 1 个主要视盘区域和 1 个主要视杯区域。若出现多个连通域，通常代表伪阳性噪声或局部误分割。

#### 3. 修复算法设计
本项目在 `src/refuge_seg/utils/postprocess.py` 中实现了：
1. 提取视盘和视杯二值区域
2. 保留最大连通域
3. 填补孔洞
4. 强制 `cup ⊆ disc`

这部分是 optional task 的重点，可直接作为 PPT 的方法页。

训练前就可以在 PPT 中先给出设计动机：
- 由于目标区域非常小，模型容易产生碎片化预测。
- 由于视杯必须位于视盘内部，仅靠普通像素损失不足以保证结构正确。
- 因此需要“后处理修复 + 拓扑约束训练”两条增强路线。

建议 PPT 图片：
- `outputs/figures/task2_optional_cases.png`
- `outputs/figures/task2_postprocess_flow.png`

`task2_postprocess_flow.png` 是一张算法示意图，用人工构造的错误预测展示：
1. 原始预测中可能存在碎片、孔洞或杯盘结构错误；
2. 后处理先保留最大连通域；
3. 再填补孔洞；
4. 最后强制 `cup` 位于 `disc` 内。

这张图适合放在 optional task 的方法页，比只展示最终预测更容易说明算法设计思路。

建议表格：

| 实验 | 是否后处理 | 视盘连通域数 | 视杯连通域数 | Mean Dice | 备注 |
|---|---|---:|---:|---:|---|
| 原始预测 | 否 | `待补充` | `待补充` | `待补充` | |
| 修复后预测 | 是 | `待补充` | `待补充` | `待补充` | |

训练后补充分析：
- `[后处理主要修复了哪些错误]`
- `[后处理对定量指标和视觉效果分别有什么影响]`

### 5.2 拓扑结构约束

#### 1. 杯盘拓扑关系
视杯必须位于视盘内部。如果预测结果中有杯区超出盘区边界，则违反解剖结构先验。

#### 2. 约束损失函数设计
本项目实现了 `TopologyAwareLoss`：
- 基础项：`Dice + CrossEntropy`
- 约束项：惩罚高概率视杯边界邻近背景的区域
- 直观含义：如果 cup 边界直接贴近 background，说明 cup 周围缺少合理的 disc rim 缓冲，更容易违反“杯在盘内”的结构先验

优点：
- 直接作用在 soft probability 上
- 可微，可反向传播
- 能把医学结构先验引入训练过程

这部分是 optional task 的核心亮点，建议在 PPT 里单独做 1 到 2 页。

#### 3. 难样本对比
建议比较：
- 基线模型预测
- 基线 + 后处理
- 拓扑约束模型预测

建议 PPT 图片：
- `outputs/figures/task2_optional_cases.png`

建议表格：

| 实验 | 模型 | 损失函数 | Mean Dice | 拓扑违规数 | 难样本表现 |
|---|---|---|---:|---:|---|
| Baseline | U-Net | Dice+CE | `待补充` | `待补充` | |
| Topology | Attention U-Net | Topology | `待补充` | `待补充` | |

训练后补充分析：
- `[拓扑约束是否减少了 cup 超出 disc 的现象]`
- `[是否牺牲了部分像素级指标换取更合理的结构]`
- `[在 hard sample 上是否更稳]`

## 六、建议生成的图片清单

建议在训练后统一生成并整理到 PPT：

| 文件 | 用途 | 建议放置章节 |
|---|---|---|
| `outputs/figures/task2_dataset_overview.png` | 数据集样本图 | Task 2.1 |
| `outputs/figures/task2_label_sanity.png` | 标签编码、类别占比、sanity check | Task 2.1 / 训练异常分析 |
| `outputs/figures/task2_training_curves.png` | 训练曲线图 | Task 2.3 / 2.4 |
| `outputs/figures/task2_metrics_bar.png` | 指标柱状图 | Task 2.3 / 2.4 |
| `outputs/figures/task2_optional_cases.png` | 后处理与拓扑约束对比图 | Task 2.5 |
| `outputs/figures/task2_postprocess_flow.png` | 后处理算法流程示意 | Task 2.5 |
| `outputs/figures/task2_prediction_diagnostics.png` | 预测面积分布诊断 | Task 2.3 / 结果可信度检查 |
| `outputs/checkpoints/*/best_prediction.png` | 单个最佳预测示例 | 任意结果页 |
| `outputs/checkpoints/*/eval_val/*.png` | 多个验证样本可视化 | Task 2.3 / 2.5 |

## 七、运行顺序

### 1. 生成数据集样本图
```bash
bash scripts/plot_task2_dataset.sh
```

### 2. 训练基线模型
```bash
bash scripts/train_task2_baseline.sh
```

### 3. 训练拓扑约束模型
```bash
bash scripts/train_task2_topology.sh
```

### 4. 评估并输出验证集结果图
```bash
bash scripts/eval_task2.sh
bash scripts/eval_task2_topology.sh
```

### 5. 汇总训练曲线和指标图
```bash
bash scripts/plot_task2_results.sh
```

### 6. 生成 optional task 对比图
```bash
bash scripts/plot_task2_optional.sh
```

### 7. 生成答辩诊断图
```bash
bash scripts/plot_task2_presentation.sh
```

该脚本生成：
- `outputs/figures/task2_label_sanity.png`
- `outputs/figures/task2_postprocess_flow.png`
- `outputs/figures/task2_prediction_diagnostics.png`

建议在每次服务器训练完成、下载 `outputs` 后都运行一次。若 `prediction_summary.json` 不存在，说明当前训练产物不是由最新评估/预测脚本生成，需要重新运行：
```bash
bash scripts/eval_task2.sh
bash scripts/eval_task2_topology.sh
python3 -m refuge_seg.predict \
  --config configs/task2_refuge_baseline.yaml \
  --checkpoint outputs/checkpoints/task2_refuge_baseline/best_model.pt \
  --input_dir REFUGE/test/Images \
  --output_dir outputs/predictions/test \
  --postprocess
```

## 八、最终结论页待补充

训练结束后，你需要在最终 PPT/Markdown 中补充：
- 哪个损失函数最好，为什么
- 后处理是否提升了结构合理性
- 拓扑约束是否减少了解剖结构违规
- 哪组配置最适合 REFUGE 视盘视杯分割
- optional task 相比主线任务的增益在哪里
