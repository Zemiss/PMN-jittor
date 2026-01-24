# PMN 新芽计划论文复现

## 环境配置

建议使用Python 3.7及以上版本。

请确保安装以下依赖：
```bash
pip install rawpy numpy opencv-python tqdm PyYAML matplotlib jittor
```

## 数据准备

请将SID数据集和ELD数据集放置在指定目录下。您可以通过修改`get_dataset_infos.py`和`trainer_SID.py`中的`root_dir`参数来指定数据集的根目录。

## 模型训练与测试脚本的说明

## 训练日志

训练日志会记录在`logs/log_SonyA7S2_Mix_Unet.log`和`logs/mytrain.log`中。

*   `logs/log_SonyA7S2_Mix_Unet.log`: 记录每训练100轮的得分以及每次测试结果。
*   `logs/mytrain.log`: 记录每轮每次训练后的损失函数和得分。

您可以通过以下命令查看训练日志：
```bash
tail -f logs/log_SonyA7S2_Mix_Unet.log
tail -f logs/mytrain.log
```

## 性能指标

*   **PSNR (峰值信噪比)**: 衡量图像重建质量的指标，值越高表示图像质量越好。
*   **SSIM (结构相似性)**: 衡量两幅图像结构相似性的指标，值越高表示图像相似度越高。


## 命令
$ 1. 生成数据集信息 $
```bash 
# Evaluate
python3 get_dataset_infos.py --dstname ELD --root_dir /home/xie/datasets/ELD --mode SonyA7S2
python3 get_dataset_infos.py --dstname SID --root_dir /home/xie/datasets/SID/Sony --mode evaltest
# Train
python3 get_dataset_infos.py --dstname SID --root_dir /home/xie/datasets/SID/Sony --mode train
```
$ 2. 评估$
```bash 
# 加入'--save_plot False'可以不保存图片
# ELD & SID
python3 trainer_SID.py -f runfiles/Ours.yml --mode evaltest
# ELD only
python3 trainer_SID.py -f runfiles/Ours.yml --mode eval
# SID only
python3 trainer_SID.py -f runfiles/Ours.yml --mode test
```
$ 3. 训练$ 
```bash 
mpirun -np 2 python3 trainer_SID.py -f runfiles/Ours.yml --mode train
```
```bash 
nohup python3 trainer_SID.py -f runfiles/Ours.yml --mode train > mytrain.log 2>&1 &
```



## 和与原版 PyTorch 实现对齐的实验结果

### Loss曲线

### 评估指标

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

在 SID 数据集上的定量结果与 ELD（TPAMI）中提供的结果存在差异，差异原因是：ELD（TPAMI）在 SID 数据集上仅对中心区域进行了结果对比

当训练到达 600 批次时，它找到了一个当时的局部最佳点。但随着训练继续，模型权重得到进一步调整，在 $1800$ 批次时成功突破了之前的性能，达到了整个实验记录中的峰值。

### 可视化结果
