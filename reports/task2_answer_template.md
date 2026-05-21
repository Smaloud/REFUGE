# 任务二 REFUGE 视盘视杯分割报告工作稿

> 用途：先作为信息记录和答辩材料库，训练完成后再据此制作 PPT。本文只写入已经能由代码、数据检查或训练产物支撑的内容；必须依赖训练结果的部分统一保留 `待补充`。

## 0. 覆盖矩阵

| PDF 要求 | 当前状态 | 代码/文件证据 | 训练后待补充 |
|---|---|---|---|
| Task 2.1 加载 REFUGE 数据集并解决路径问题 | 已完成 | `src/refuge_seg/datasets/refuge_dataset.py`、`src/refuge_seg/check_data.py` | 无 |
| Task 2.1 展示 image 和 label | 已完成脚本 | `scripts/plot_task2_dataset.sh`、`scripts/plot_task2_presentation.sh` | 用最新数据重新生成图片 |
| Task 2.1 描述视盘/视杯特点 | 已完成文字 | 本报告 Task 2.1 | 可配合样本图调整措辞 |
| Task 2.2 调研主流分割网络 | 已完成文字 | 本报告 Task 2.2 | 可补一页网络示意图 |
| Task 2.2 选择基座模型并说明理由 | 已完成 | `src/refuge_seg/models/unet.py`、`configs/task2_refuge_baseline.yaml` | 结合最终指标补充选择合理性 |
| Task 2.3 分割与分类任务差异 | 已完成文字 | 本报告 Task 2.3 | 无 |
| Task 2.3 损失函数方向：CE/Dice/Dice+CE | 已完成代码，待训练对比 | `src/refuge_seg/losses/segmentation_losses.py`、`configs/task2_refuge_loss_*.yaml` | CE、Dice、Dice+CE 指标和曲线 |
| Task 2.3 评估指标 IoU/Dice | 已完成 | `src/refuge_seg/utils/metrics.py`、`src/refuge_seg/evaluate.py` | 最终数值表 |
| Task 2.4 可调参数列表 | 已完成文字和配置 | `configs/task2_refuge_*.yaml` | 无 |
| Task 2.4 参数影响实验 | 已完成配置，待训练 | `configs/task2_refuge_lr_1e4.yaml`、`configs/task2_refuge_batch2.yaml`、`configs/task2_refuge_scheduler_step.yaml` | 训练时长、曲线、收敛分析 |
| Task 2.5.1 孔洞/连通域检查 | 已完成代码 | `src/refuge_seg/utils/postprocess.py`、`src/refuge_seg/evaluate.py` | 诊断统计表和前后对比图 |
| Task 2.5.1 后处理修复算法 | 已完成代码和示意图脚本 | `postprocess_prediction`、`scripts/plot_task2_presentation.sh` | 用真实预测补充案例 |
| Task 2.5.2 杯在盘内拓扑约束检查 | 已完成诊断代码 | `diagnose_prediction` | 拓扑违规数 |
| Task 2.5.2 自定义约束损失 | 已完成代码 | `TopologyAwareLoss` | 结合训练结果解释效果 |
| Task 2.5.2 难样本普通损失 vs 约束损失 | 已完成脚本，待训练 | `scripts/plot_task2_optional.sh` | 难样本图和结论 |

## 1. Task 2.1 认识数据集

### 1.1 数据集路径与加载

本项目使用 REFUGE 视盘/视杯分割数据集，目录约定为：

| 子目录 | 内容 | 标签情况 |
|---|---|---|
| `REFUGE/train/Images` | 训练图像 | 有对应 `train/gts` |
| `REFUGE/train/gts` | 训练标签 | 灰度三类标签 |
| `REFUGE/val/Images` | 验证图像 | 有对应 `val/gts` |
| `REFUGE/val/gts` | 验证标签 | 灰度三类标签 |
| `REFUGE/test/Images` | 测试图像 | 无公开标签 |

本地数据检查结果：

| split | 图像数 | 标签数 |
|---|---:|---:|
| train | 400 | 400 |
| val | 400 | 400 |
| test | 400 | 0 |

本地标签样例已经确认是三类灰度：

| 文件 | 像素值统计 |
|---|---|
| `REFUGE/train/gts/g0001.bmp` | `{0: 36895, 128: 51823, 255: 4278226}` |
| `REFUGE/train/gts/g0002.bmp` | `{0: 51962, 128: 23263, 255: 4291719}` |
| `REFUGE/val/gts/V0001.bmp` | `{0: 12124, 128: 35053, 255: 2622779}` |

标签语义映射：

| 原始灰度 | 训练类别 | 语义 |
|---:|---:|---|
| 255 | 0 | 背景 |
| 128 | 1 | 视盘环区 |
| 0 | 2 | 视杯 |

说明：训练时的视盘指标按 `class 1 + class 2` 计算，因为完整视盘包含视杯区域；视杯指标只按 `class 2` 计算。

对应代码：
- `src/refuge_seg/datasets/refuge_dataset.py`
- `src/refuge_seg/check_data.py`

数据检查命令：

```bash
export PYTHONPATH=src
python3 -m refuge_seg.check_data --data-root REFUGE
```

### 1.2 图像和标签展示

需要放入 PPT 的图片：

| 图片 | 生成脚本 | 用途 |
|---|---|---|
| `outputs/figures/task2_dataset_overview.png` | `scripts/plot_task2_dataset.sh` | 展示多组 image/label |
| `outputs/figures/task2_dataset_stats.png` | `scripts/plot_task2_dataset.sh` | 展示数据划分和像素比例 |
| `outputs/figures/task2_structure_example.png` | `scripts/plot_task2_dataset.sh` | 展示视盘/视杯结构 |
| `outputs/figures/task2_label_sanity.png` | `scripts/plot_task2_presentation.sh` | 展示标签值、映射和类别不平衡 |

生成命令：

```bash
bash scripts/plot_task2_dataset.sh
bash scripts/plot_task2_presentation.sh
```

### 1.3 视盘和视杯特点

可直接用于 PPT 的描述：

- 视盘是眼底图中较大的亮色椭圆区域，通常位于血管汇聚位置。
- 视杯位于视盘内部，面积更小、更靠中心。
- 正常结构中视杯应完全包含在视盘内部，这是后续拓扑约束的医学先验。
- 视盘和视杯在整张图中占比很小，前景/背景极不平衡，因此只用像素级 CE 可能偏向背景类。

## 2. Task 2.2 网络调研和选择

### 2.1 主流图像分割网络

| 网络 | 核心思想 | 优点 | 局限 |
|---|---|---|---|
| FCN | 将分类网络改为全卷积结构做像素预测 | 简洁，是语义分割基础框架 | 细节恢复较弱 |
| U-Net | 编码器-解码器 + skip connection | 医学图像常用，小数据集表现稳健 | 对复杂边界和结构先验没有显式约束 |
| DeepLabV3+ | 空洞卷积和 ASPP 多尺度上下文 | 上下文建模强 | 模型较重，训练成本更高 |
| UNet++ | 密集跳跃连接和多尺度融合 | 边界和尺度融合更细 | 结构更复杂 |
| Attention U-Net | 在 skip feature 上加入注意力门控 | 能抑制无关背景特征 | 训练成本略高 |

### 2.2 本项目模型选择

基线模型选择 `U-Net`：

- REFUGE 训练集规模不大，U-Net 是医学分割中稳定、可解释的基线。
- 视盘/视杯边界需要从低层纹理中恢复，U-Net 的 skip connection 适合保留空间细节。
- U-Net 输出 `[B, 3, H, W]`，直接对应背景、视盘环、视杯三类像素。

Optional task 中使用 `Attention U-Net + TopologyAwareLoss`：

- Attention U-Net 用于增强对目标区域的关注。
- Topology-aware loss 用于引入“视杯必须在视盘内”的结构先验。

对应代码：
- `src/refuge_seg/models/unet.py`
- `src/refuge_seg/models/attention_unet.py`
- `configs/task2_refuge_baseline.yaml`
- `configs/task2_refuge_topology.yaml`

## 3. Task 2.3 网络代码、损失函数和指标

### 3.1 分割任务与分类任务差异

| 对比项 | 分类任务 | 分割任务 |
|---|---|---|
| 输入 | `[B, C, H, W]` | `[B, C, H, W]` |
| 输出 logits | `[B, num_classes]` | `[B, num_classes, H, W]` |
| 标签 | 图像级类别 | 每个像素一个类别 |
| 优化目标 | 判断整张图类别 | 判断每个像素类别并保持区域结构 |
| 常用指标 | Accuracy、F1 | Dice、IoU、边界质量、连通域结构 |

### 3.2 已实现损失函数

| 损失函数 | 配置名 | 优化倾向 | 代码 |
|---|---|---|---|
| CrossEntropy | `ce` | 像素级分类，训练稳定，但受背景占比影响大 | `CrossEntropyLoss` |
| Dice | `dice` | 直接优化区域重叠，对小目标更友好 | `DiceLoss` |
| Dice + CE | `dice_ce` | 同时兼顾像素分类稳定性和区域重叠 | `DiceCrossEntropyLoss` |
| Topology | `topology` | 在 Dice+CE 基础上加入结构先验惩罚 | `TopologyAwareLoss` |

对应代码：
- `src/refuge_seg/losses/segmentation_losses.py`

### 3.3 已实现评价指标

| 指标 | 计算对象 | 说明 |
|---|---|---|
| `dice_disc` | `pred in {1,2}` vs `target in {1,2}` | 完整视盘 Dice |
| `dice_cup` | `pred == 2` vs `target == 2` | 视杯 Dice |
| `iou_disc` | 完整视盘 | 视盘 IoU |
| `iou_cup` | 视杯 | 视杯 IoU |
| `mean_dice` | disc/cup 平均 | 主模型选择指标 |
| `mean_iou` | disc/cup 平均 | 辅助指标 |

对应代码：
- `src/refuge_seg/utils/metrics.py`
- `src/refuge_seg/evaluate.py`

### 3.4 损失函数对比实验设计

已补齐可复现实验配置：

| 实验 | 配置文件 | 模型 | 损失 | 输出目录 | Mean Dice | Mean IoU | 结论 |
|---|---|---|---|---|---:|---:|---|
| E1 | `configs/task2_refuge_loss_ce.yaml` | U-Net | CE | `outputs/checkpoints/task2_refuge_loss_ce` | 待补充 | 待补充 | 待补充 |
| E2 | `configs/task2_refuge_loss_dice.yaml` | U-Net | Dice | `outputs/checkpoints/task2_refuge_loss_dice` | 待补充 | 待补充 | 待补充 |
| E3 | `configs/task2_refuge_baseline.yaml` | U-Net | Dice+CE | `outputs/checkpoints/task2_refuge_baseline` | 待补充 | 待补充 | 推荐主线 |

运行命令：

```bash
bash scripts/train_task2_loss_ablation.sh
bash scripts/eval_task2_loss_ablation.sh
bash scripts/plot_task2_results.sh
```

训练后需要补充：
- CE 是否出现背景偏置或小目标 cup 分割弱的问题。
- Dice 是否提升区域重叠，但训练曲线是否更波动。
- Dice+CE 是否在稳定性和分割精度之间更平衡。

## 4. Task 2.4 模型参数调整与优化

### 4.1 可调参数

| 参数 | 当前默认值 | 作用 |
|---|---|---|
| `lr` | `0.0003` | 控制参数更新步长，影响收敛速度和稳定性 |
| `batch_size` | `4` | 影响显存占用、梯度稳定性和每 epoch 时间 |
| `optimizer` | `adam` | 控制优化方式 |
| `scheduler` | `cosine` | 控制训练过程中学习率变化 |
| `epochs` | `40` | 控制训练轮数 |
| `image_size` | `512` | 控制输入分辨率和显存开销 |
| `base_channels` | `32` | 控制模型容量 |
| `lambda_topology` | `0.5` in topology config | 控制拓扑约束强度 |

### 4.2 参数实验设计

已补齐可复现实验配置：

| 实验 | 配置文件 | 改动参数 | 输出目录 | 平均每 epoch 时间 | Mean Dice | 现象记录 |
|---|---|---|---|---:|---:|---|
| P1 | `configs/task2_refuge_baseline.yaml` | `lr=3e-4, batch=4, scheduler=cosine` | `task2_refuge_baseline` | 待补充 | 待补充 | baseline |
| P2 | `configs/task2_refuge_lr_1e4.yaml` | `lr=1e-4` | `task2_refuge_lr_1e4` | 待补充 | 待补充 | 待补充 |
| P3 | `configs/task2_refuge_batch2.yaml` | `batch_size=2` | `task2_refuge_batch2` | 待补充 | 待补充 | 待补充 |
| P4 | `configs/task2_refuge_scheduler_step.yaml` | `scheduler=step` | `task2_refuge_scheduler_step` | 待补充 | 待补充 | 待补充 |

说明：这里选择 `batch_size=2` 而不是 `8`，是为了降低服务器显存风险；它仍能回答 batch size 对训练速度和稳定性的影响。

运行命令：

```bash
bash scripts/train_task2_param_ablation.sh
bash scripts/eval_task2_param_ablation.sh
bash scripts/plot_task2_results.sh
```

代码已经保存训练耗时：

| 文件 | 内容 |
|---|---|
| `outputs/checkpoints/*/history.json` | 每轮 train loss、val loss、val mean dice、epoch time |
| `outputs/checkpoints/*/training_summary.json` | 总训练时长、平均 epoch 时长、最佳 mean dice |

训练后需要补充：
- 学习率降低后是否收敛更慢但更稳定。
- batch size 变小后单 epoch 是否更慢、曲线是否更波动。
- step scheduler 与 cosine scheduler 的后期收敛差异。

## 5. Task 2.5 Optional Task

### 5.1 分割效果评估和后处理

#### 孔洞检查

分割结果中，如果视盘或视杯内部出现背景孔洞，说明模型边界预测不连续或局部置信度不足。正常眼底结构中，视盘和视杯区域应是相对完整的连通区域。

#### 连通域检查

正常情况下，一张眼底图应只有一个主要视盘区域和一个主要视杯区域。如果预测出现多个连通域，通常代表：

- 背景区域被误分为视盘或视杯；
- 小目标 cup 受类别不平衡影响出现碎片；
- 图像边界、血管遮挡或低对比度造成误判。

#### 后处理算法设计

本项目实现的后处理流程：

1. 将预测结果转换为 disc mask 和 cup mask。
2. 对 disc 和 cup 分别保留最大连通域。
3. 对 disc 和 cup 分别填补孔洞。
4. 强制执行 `cup = cup * disc`，保证视杯不会超出视盘。
5. 重新组合成三分类 mask。

对应代码：
- `src/refuge_seg/utils/postprocess.py`

建议插图：
- `outputs/figures/task2_postprocess_flow.png`：人工构造错误样例，展示修复流程。
- `outputs/figures/task2_optional_cases.png`：真实验证集 hard cases 的修复前后对比。

训练后结果表：

| 实验 | 是否后处理 | disc 多连通域样本数 | cup 多连通域样本数 | 孔洞样本数 | Mean Dice | 结论 |
|---|---|---:|---:|---:|---:|---|
| Baseline raw | 否 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| Baseline post | 是 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

说明：`evaluate.py` 已保存 `diagnosis_summary`，训练后从 `outputs/checkpoints/*/eval_val/metrics.json` 读取即可。

为了同时保留修复前和修复后的指标，使用：

```bash
bash scripts/eval_task2_postprocess_ablation.sh
```

输出目录：
- `outputs/checkpoints/task2_refuge_baseline/eval_val_raw`
- `outputs/checkpoints/task2_refuge_baseline/eval_val_post`

### 5.2 拓扑结构约束

#### 杯盘拓扑关系

视杯必须位于视盘内部。如果预测结果中出现 `cup outside disc`，则违反解剖结构先验。这个约束不一定能通过普通 CE 或 Dice 自动满足，因为普通像素损失更关注逐像素分类和区域重叠。

#### 约束损失函数设计

本项目实现 `TopologyAwareLoss`：

- 基础项：`Dice + CrossEntropy`
- 约束项：对高概率 cup 边界附近出现高背景概率的区域进行惩罚
- 直观含义：如果 cup 边界直接接近 background，说明 cup 周围缺少合理的 disc rim，容易出现 cup 超出 disc 的结构错误

该损失直接作用于 softmax probability，由 PyTorch tensor 运算组成，因此可微，支持反向传播。

对应代码：
- `topology_penalty`
- `TopologyAwareLoss`
- `configs/task2_refuge_topology.yaml`

#### 难样本对比

训练后需要比较：

| 实验 | 模型 | 损失函数 | Mean Dice | cup outside disc 样本数 | 难样本表现 |
|---|---|---|---:|---:|---|
| Baseline | U-Net | Dice+CE | 待补充 | 待补充 | 待补充 |
| Baseline + Postprocess | U-Net | Dice+CE + 后处理 | 待补充 | 待补充 | 待补充 |
| Topology | Attention U-Net | TopologyAwareLoss | 待补充 | 待补充 | 待补充 |

运行命令：

```bash
bash scripts/train_task2_topology.sh
bash scripts/eval_task2_topology.sh
bash scripts/plot_task2_optional.sh
```

训练后需要补充：
- 拓扑约束是否减少 `cup_outside_disc_cases`。
- 拓扑约束是否改善 hard sample 的结构合理性。
- 是否存在像素级指标和结构合理性之间的取舍。

## 6. 需要生成的图表清单

| 文件 | 用途 | PPT 章节 |
|---|---|---|
| `outputs/figures/task2_dataset_overview.png` | image/label 示例 | Task 2.1 |
| `outputs/figures/task2_dataset_stats.png` | 数据划分和像素比例 | Task 2.1 |
| `outputs/figures/task2_structure_example.png` | 视盘/视杯结构说明 | Task 2.1 |
| `outputs/figures/task2_label_sanity.png` | 标签编码和类别不平衡检查 | Task 2.1 |
| `outputs/figures/task2_training_curves.png` | loss 曲线和 mean dice 曲线 | Task 2.3 / 2.4 |
| `outputs/figures/task2_metrics_bar.png` | 不同实验指标柱状图 | Task 2.3 / 2.4 |
| `outputs/figures/task2_postprocess_flow.png` | 后处理算法示意 | Task 2.5.1 |
| `outputs/figures/task2_optional_cases.png` | hard cases 对比 | Task 2.5 |
| `outputs/figures/task2_prediction_diagnostics.png` | 预测面积分布诊断 | 结果可信度检查 |
| `outputs/checkpoints/*/best_prediction.png` | 最佳 epoch 的预测示例 | 结果展示 |
| `outputs/checkpoints/*/eval_val/*.png` | 验证集预测可视化 | 结果展示 |
| `reports/task2_experiment_summary.md` | 自动汇总实验结果表 | Task 2.3 / 2.4 / 2.5 |

## 7. 服务器训练后任务表

| 顺序 | 任务 | 命令 | 产物 | 完成后填入报告的位置 |
|---:|---|---|---|---|
| 1 | 拉取最新代码并检查三类标签 | `git pull --no-rebase origin main && python3 -m refuge_seg.check_data --data-root REFUGE` | 标签统计 | Task 2.1 |
| 2 | 生成数据集和标签检查图 | `bash scripts/plot_task2_dataset.sh && bash scripts/plot_task2_presentation.sh` | dataset/label sanity 图 | Task 2.1 |
| 3 | 跑损失函数对比 | `bash scripts/train_task2_loss_ablation.sh` | CE/Dice/Dice+CE checkpoints | Task 2.3 |
| 4 | 评估损失函数对比 | `bash scripts/eval_task2_loss_ablation.sh` | `metrics.json`、验证图 | Task 2.3 |
| 5 | 跑参数对比 | `bash scripts/train_task2_param_ablation.sh` | lr/batch/scheduler checkpoints | Task 2.4 |
| 6 | 评估参数对比 | `bash scripts/eval_task2_param_ablation.sh` | `metrics.json`、耗时统计 | Task 2.4 |
| 7 | 跑拓扑约束模型 | `bash scripts/train_task2_topology.sh` | topology checkpoint | Task 2.5.2 |
| 8 | 评估拓扑模型 | `bash scripts/eval_task2_topology.sh` | topology metrics | Task 2.5.2 |
| 9 | 保留后处理前后评估 | `bash scripts/eval_task2_postprocess_ablation.sh` | `eval_val_raw`、`eval_val_post` | Task 2.5.1 |
| 10 | 汇总曲线和指标图 | `bash scripts/plot_task2_results.sh` | curves/bar charts | Task 2.3 / 2.4 |
| 11 | 生成后处理和 hard sample 图 | `bash scripts/plot_task2_optional.sh && bash scripts/plot_task2_presentation.sh` | optional cases、diagnostics | Task 2.5 |
| 12 | 自动生成实验结果汇总表 | `bash scripts/summarize_task2_experiments.sh` | `reports/task2_experiment_summary.md` | 复制到 Task 2.3/2.4/2.5 |
| 13 | 导出 test 预测 | `python3 -m refuge_seg.predict --config configs/task2_refuge_baseline.yaml --checkpoint outputs/checkpoints/task2_refuge_baseline/best_model.pt --input_dir REFUGE/test/Images --output_dir outputs/predictions/test --postprocess` | test masks | 最终提交/展示 |
| 14 | 打包训练产物下载到本地 | `tar -czf task2_results_$(date +%Y%m%d_%H%M%S).tar.gz outputs reports configs scripts src tests` | 结果压缩包 | 本地制作 PPT |

## 8. PPT 建议结构

| 页码 | 内容 |
|---:|---|
| 1 | 题目：REFUGE 视盘视杯分割 |
| 2 | 数据集结构和任务目标 |
| 3 | image/label 示例，说明视盘和视杯 |
| 4 | 标签编码、类别不平衡和数据检查 |
| 5 | 主流分割网络调研 |
| 6 | 选择 U-Net 的理由 |
| 7 | 分割任务 vs 分类任务 |
| 8 | 损失函数设计：CE、Dice、Dice+CE |
| 9 | 损失函数对比结果，训练后补图和表 |
| 10 | 可调参数和实验配置 |
| 11 | 参数实验结果，训练后补图和表 |
| 12 | 后处理：孔洞、连通域、cup in disc |
| 13 | 后处理前后对比，训练后补 hard sample |
| 14 | 拓扑约束损失设计，可微性说明 |
| 15 | 普通损失 vs 拓扑约束难样本对比 |
| 16 | 总结：最佳配置、主要发现、局限 |

## 9. 最终结论页待补充

训练全部完成后补充以下结论：

- CE、Dice、Dice+CE 中哪个综合效果最好，原因是什么。
- 哪组参数配置在精度、收敛速度和训练成本之间最合适。
- 后处理是否减少孔洞、多连通域和拓扑违规。
- 拓扑约束是否改善 hard sample 的结构合理性。
- 最终推荐用于 test 预测的 checkpoint 是哪一个。
