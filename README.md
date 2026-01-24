# 新芽计划论文复现（PMN）

- 这是**南开大学人工智能实践课**的第四次作业 jittor复现部分内容，选择的主题是**图像RAW域去噪**

- 本次作业选择的论文是**2022 年 ACMMM 奖项候选论文：低光照原始数据去噪的可学习性增强：配对真实数据与噪声建模的结合**并于2023年8月3日被IEEE模式分析与机器智能汇刊（TPAMI）2023年接收。

- 本次作业完成了**PMN的jittor框架的迁移**，成功在jittor框架下复现实验。

## </font> <font color="#ce0c0c">环境配置</font> 

Python 版本 >= 3.7，Jittor 版本 >= 1.3
所需工具：opencv-python、rawpy、exifread、h5py、scipy、tqdm、PyYAML、matplotlib、scikit-image
平台：Ubuntu 24.04、cuda-12.0
可以在CPU上运行，但建议在GPU上运行。

## </font> <font color="#ce0c0c">数据准备</font>

请将SID数据集和ELD数据集放置在指定目录下。

ELD ([官方](https://github.com/Vandermode/ELD)): [下载 (11.46 GB)](https://drive.google.com/file/d/13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX/view?usp=sharing)  
SID ([官方](https://github.com/cchen156/Learning-to-See-in-the-Dark)):  [下载 (25 GB)](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)

可以通过修改`get_dataset_infos.py`和`trainer_SID.py`中的`root_dir`参数来指定数据集的根目录。

## </font> <font color="#ce0c0c">模型训练与测试脚本的说明</font>

**主要脚本说明**

| 脚本 | 说明 |
|------|------|
| `trainer_SID.py` | 核心训练和评估脚本，支持训练、评估、测试等多种模式 |
| `get_dataset_infos.py` | 数据集信息生成脚本，预处理数据集并生成 info 文件 |
| `base_trainer.py` | 基础训练器，包含通用的训练逻辑 |
| `utils.py` | 工具函数库，包含 AverageMeter、PSNR 计算等 |
| `losses.py` | 损失函数定义 |
| `scripts/plot_psnr.py` | PSNR 曲线绘制脚本，用于可视化训练过程 |

**训练脚本模式说明**

- **`train`**: 训练模式，对模型进行训练
- **`eval`**: 评估模式，在 ELD 数据集上评估
- **`test`**: 测试模式，在 SID 数据集上测试
- **`evaltest`**: 综合评估模式，同时在 ELD 和 SID 数据集上评估

**配置文件说明**

| 配置文件 | 说明 |
|---------|------|
| `Ours.yml` | 主要训练配置，包含模型、数据、超参等设置 |
| `Ours_finetune.yml` | 微调配置 |
| `Paired.yml` | 成对数据配置 |
| `ELD.yml` | ELD 数据集配置 |

**关键参数说明**

```yaml
# 模型与数据
model_name: 'SonyA7S2_Mix_Unet'  # 模型名称
arch:
  name: 'UNetSeeInDark'           # 网络架构
  nf: 32                          # 网络特征数
dst_train:
  dataset: 'Mix_Dataset'          # 训练数据集
dst_eval:
  dataset: 'ELD_Dataset'          # 评估数据集

# 训练超参
hyper:
  lr_scheduler: 'WarmupCosine'   # 学习率调度器
  learning_rate: 2.e-4           # 学习率
  batch_size: 2                  # 批大小
  stop_epoch: 1800               # 总训练轮数
  plot_freq: 100                 # 每隔多少轮进行一次评估
  save_freq: 10                  # 每隔多少轮保存一次模型
```

## </font> <font color="#ce0c0c">训练日志</font>

训练日志会记录在`logs/log_SonyA7S2_Mix_Unet.log`和`logs/mytrain.log`中。

*   `logs/log_SonyA7S2_Mix_Unet.log`: 记录每训练100轮的得分以及每次测试结果。
*   `logs/mytrain.log`: 记录每轮每次训练后的损失函数和得分。

您可以通过以下命令查看训练日志：
```bash
tail -f logs/log_SonyA7S2_Mix_Unet.log
tail -f logs/mytrain.log
```




## </font> <font color="#ce0c0c">训练/评估命令</font>

### 1. 生成数据集信息

首先需要生成数据集的信息文件（仅需执行一次）：

```bash
# 生成 ELD 数据集信息（用于评估）
python3 get_dataset_infos.py --dstname ELD --root_dir /home/xie/datasets/ELD --mode SonyA7S2

# 生成 SID 数据集信息（用于测试）
python3 get_dataset_infos.py --dstname SID --root_dir /home/xie/datasets/SID/Sony --mode evaltest

# 生成 SID 数据集信息（用于训练）
python3 get_dataset_infos.py --dstname SID --root_dir /home/xie/datasets/SID/Sony --mode train
```

### 2. 训练模型

**单 GPU 训练**（推荐在 GPU 不足时使用）：
```bash
python3 trainer_SID.py -f runfiles/Ours.yml --mode train
```

**后台训练**（推荐方式，可在断开连接后继续训练）：
```bash
nohup python3 trainer_SID.py -f runfiles/Ours.yml --mode train > logs/mytrain.log 2>&1 &
```

**多 GPU 训练**（需要 MPI 环境，使用 2 个 GPU）：
```bash
mpirun -np 2 python3 trainer_SID.py -f runfiles/Ours.yml --mode train
```

### 3. 模型评估与测试

**综合评估**（同时评估 ELD 和 SID 数据集）：
```bash
python3 trainer_SID.py -f runfiles/Ours.yml --mode evaltest
```

**仅评估 ELD 数据集**：
```bash
python3 trainer_SID.py -f runfiles/Ours.yml --mode eval
```

**仅测试 SID 数据集**：
```bash
python3 trainer_SID.py -f runfiles/Ours.yml --mode test
```

**不保存图像结果**（加快评估速度）：
```bash
python3 trainer_SID.py -f runfiles/Ours.yml --mode evaltest --save_plot False
```


## </font> <font color="#ce0c0c">和与原版 PyTorch 实现对齐的实验结果</font> 

### </font> <font color="#ce0c0c">Psnr曲线（替代loss曲线）</font> 

本项目使用 Jittor 框架复现，PSNR 曲线与原版 PyTorch 实现基本对齐。训练共 1800 轮，每 100 轮进行一次评估。

| Pytorch PSNR 曲线 | Jittor PSNR 曲线 |
| :---: | :---: |
| ![](images/github/原psnr曲线.jpg) | ![](images/github/psnr曲线.png) |

### </font> <font color="#ce0c0c">评估指标</font> 

*   **PSNR (峰值信噪比)**: 衡量图像重建质量的指标，值越高表示图像质量越好。常用于图像降噪、超分辨率等任务评估。公式：$PSNR = 10 \log_{10}(MAX^2 / MSE)$
*   **SSIM (结构相似性)**: 衡量两幅图像结构相似性的指标，值越高表示图像相似度越高。相比 PSNR，SSIM 更符合人眼感知。范围：[-1, 1]，通常取 [0, 1]。


| Dataset | Ratio | Index | P-G   | ELD   | SFRN  | Paired      | pytorch  |     jittor600轮     |    jittor1800轮    |
|---------|-------|-------|-------|-------|-------|-------------|-------|---------------|---------------|
| ELD     | ×100  | PSNR  | 42.05 | 45.45 | 46.02 | 44.47       | 46.50 |     46.25     |    46.42      |
|         |       | SSIM  | 0.872 | 0.975 | 0.977 | 0.968       | 0.985 |     0.982     |    0.981      |
|         | ×200  | PSNR  | 38.18 | 43.43 | 44.10 | 41.97       | 44.51 |     44.19     |    44.68      |
|         |       | SSIM  | 0.782 | 0.954 | 0.964 | 0.928       | 0.973 |     0.965     |    0.971      |
| SID     | ×100  | PSNR  | 39.44 | 41.95 | 42.29 | 42.06       | 43.16 |     42.84     |    43.01      |
|         |       | SSIM  | 0.890 | 0.963 | 0.951 | 0.955       | 0.960 |     0.956     |    0.959      |
|         | ×250  | PSNR  | 34.32 | 39.44 | 40.22 | 39.60       | 40.92 |     40.75     |    40.68      |
|         |       | SSIM  | 0.768 | 0.931 | 0.938 | 0.938       | 0.947 |     0.944     |    0.943      |
|         | ×300  | PSNR  | 30.66 | 36.36 | 36.87 | 36.85       | 37.77 |     37.41     |    37.51      |
|         |       | SSIM  | 0.657 | 0.911 | 0.917 | 0.923       | 0.934 |     0.928     |    0.929      |

当训练到达 600 批次时，是整个流程的局部最佳点，后续重新选择前600轮中的最优模型开始训练，所以也对600轮模型进行了评估。

### </font> <font color="#ce0c0c">可视化结果</font> 

![对比图](images/github/li.jpg)