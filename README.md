<div align="center">

# DeepLearningWithCNN

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-13.2-76B900?style=flat-square&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-FashionMNIST-FF6F00?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-10%20Monkeys-FF6F00?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-CIFAR--10-FF6F00?style=flat-square&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p>基于 PyTorch 的卷积神经网络图像分类实战系列</p>
<p>以 <b>FashionMNIST → 10 Monkeys → CIFAR-10</b> 为主线，循序渐进演示从基础 CNN 到 ResNet、InceptionNet 等经典架构的完整训练流程</p>
<p>每个 Notebook 均配有详细中文注释，适合深度学习入门与进阶学习</p>

</div>

---

## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [Notebook 介绍](#-notebook-介绍)
- [数据集](#-数据集)
- [学习路径](#-学习路径)
- [环境依赖](#️-环境依赖)
- [快速开始](#-快速开始)

---

## 🗺 项目概览


| #   | Notebook                | 核心技术                                            | 亮点                |
| --- | ----------------------- | ----------------------------------------------- | ----------------- |
| 1   | FashionMNIST CNN + SELU | CNN · SELU · LeCun 初始化 · AlphaDropout           | 自归一化卷积网络 · 无需 BN  |
| 2   | FashionMNIST 深度可分离卷积    | Depthwise Separable Conv · ReLU · He 初始化 · BN2D | 参数量更少 · 推理更快      |
| 3   | 10 Monkeys 从头训练         | 自定义 CNN · 数据增强 · 自定义 Dataset                    | 真实彩色图像分类入门        |
| 4   | 10 Monkeys 迁移学习         | ResNet50 · 迁移学习 · 微调 · 分层学习率                    | 快速收敛 · 工业级训练范式    |
| 5   | CIFAR-10 自定义 CNN        | CNN · 分层采样 · 数据增强                               | 类别均衡采样策略          |
| 6   | CIFAR-10 ResNet-34      | ResNet-34 · 残差块 · 跳跃连接                          | 从零复现经典残差网络        |
| 7   | CIFAR-10 InceptionNet   | Inception 模块 · 多尺度并行卷积                          | 从零复现 Inception 架构 |


---

## 📁 目录结构

```
DeepLearningWithCNN/
├── 📓 1.FashionMNISTImage_classification_cnn_selu+lecun+AlphaDropout.ipynb
├── 📓 2.FashionMNISTImage_classification_SeparableCNN_relu+he+bn2d+AlphaDropout.ipynb
├── 📓 3.10MonkeysImage_classification_cnn_relu+he+bn2d+AlphaDropout.ipynb
├── 📓 4.10MonkeysImage_classification_resnet50_finetune_DataAugmentation.ipynb
├── 📓 5.Cifar10Image_classification_cnn_relu+he+bn2d+AlphaDropout_StratifiedSampling_DataAugmentation.ipynb
├── 📓 6.Cifar10Image_classification_ResNet34.ipynb
├── 📓 7.Cifar10Image_classification_Inception.ipynb
│
├── 📂 data/
│   ├── FashionMNIST/              # FashionMNIST 数据集（自动下载）
│   ├── cifar-10/                  # CIFAR-10 数据集
│   └── archive/                   # 10 Monkey Species 数据集
│       ├── training/
│       └── validation/
│
├── 📂 model_checkpoints/          # 各实验最优模型权重（.ckpt）
│   ├── 1_model/
│   ├── 2_model/
│   ├── 3_model/
│   ├── 4_model/
│   ├── 5_model/
│   ├── 6_model/
│   └── 7_model/
│
├── 📄 resnet34_graph.png           # ResNet-34 网络结构可视化
├── 📄 submission.csv               # CIFAR-10 测试集预测结果
├── 📄 requirements.txt
├── 📄 LICENSE
└── 📄 README.md
```

---

## 📚 Notebook 介绍

### 1 · FashionMNIST CNN + SELU + AlphaDropout

> `1.FashionMNISTImage_classification_cnn_selu+lecun+AlphaDropout.ipynb`

使用 PyTorch 搭建 CNN，以 **SELU（自缩放指数线性单元）** 代替传统 ReLU 实现自归一化特性，无需 BatchNorm，在 FashionMNIST 上完成十类服装图像分类。


| 章节    | 内容                                             |
| ----- | ---------------------------------------------- |
| 环境导入  | 依赖库导入 · 设备检测（CPU / GPU）                        |
| 数据集准备 | FashionMNIST 加载 · 均值/方差计算 · 标准化变换 · DataLoader |
| 构建模型  | CNN + SELU + LeCun 初始化 + AlphaDropout          |
| 模型训练  | 损失函数 · 优化器 · 训练循环 · 最优模型保存                     |
| 模型评估  | 验证集准确率 · TensorBoard 可视化                       |


> **SELU 原理** · 具有自归一化特性，各层激活值自动保持均值 ≈ 0、方差 ≈ 1，从根本上解决梯度消失/爆炸，**无需 BatchNorm**。配套使用 LeCun 初始化与 AlphaDropout，使 Dropout 后输出仍保持自归一化特性。
>
> 参考论文：*Self-Normalizing Neural Networks*（Klambauer et al., 2017）

---

### 2 · FashionMNIST 深度可分离卷积（Separable CNN）

> `2.FashionMNISTImage_classification_SeparableCNN_relu+he+bn2d+AlphaDropout.ipynb`

使用 **深度可分离卷积（Depthwise Separable Convolution）** 替代标准卷积，在保持精度的同时大幅削减参数量与计算量，PyTorch 未内置该层，本 Notebook 从头手动实现。


| 对比维度 | 标准卷积（参考）            | 深度可分离卷积（本篇）                |
| ---- | ------------------- | -------------------------- |
| 参数量  | `C_in × C_out × k²` | `C_in × k² + C_in × C_out` |
| 计算量  | 高                   | 约为标准卷积的 `1/k²`             |
| 实现方式 | `nn.Conv2d`         | 手动实现 `DepthWiseConv2d`     |
| 精度   | ✅                   | 接近（略有取舍）                   |


> **原理** · 将标准卷积拆分为逐通道卷积（Depthwise）和逐点卷积（Pointwise）两步，参数量约为标准卷积的 `1/k²`（k 为卷积核大小），适合资源受限场景。

---

### 3 · 10 Monkeys 从头训练 CNN

> `3.10MonkeysImage_classification_cnn_relu+he+bn2d+AlphaDropout.ipynb`

使用 PyTorch 对 **10 Monkey Species** 真实彩色图像数据集，从零开始训练自定义 CNN，实现对十种猴子的细粒度图像分类，重点演示自定义数据集加载与数据增强流程。


| 章节    | 内容                                              |
| ----- | ----------------------------------------------- |
| 环境导入  | 依赖库导入 · 设备检测                                    |
| 数据集准备 | 自定义 `MonkeyDataset` · 类别映射查看 · 图片路径/标签验证 · 数据增强 |
| 构建模型  | 自定义 CNN（ReLU + He 初始化 + BN2D + AlphaDropout）    |
| 模型训练  | 训练循环 · 最优模型保存                                   |
| 模型评估  | 验证集准确率 · TensorBoard 可视化                        |


---

### 4 · 10 Monkeys ResNet50 迁移学习 + 微调

> `4.10MonkeysImage_classification_resnet50_finetune_DataAugmentation.ipynb`

使用 ImageNet 预训练 **ResNet50**，通过**迁移学习 + 微调**策略，在 10 Monkey Species 数据集上快速达到高精度，展示工业级迁移学习范式。


| 章节    | 内容                                          |
| ----- | ------------------------------------------- |
| 环境导入  | 依赖库导入 · 设备检测                                |
| 数据集准备 | 自定义 `MonkeyDataset` · 全局均值/方差计算 · 训练/验证变换管道 |
| 构建模型  | 加载预训练 ResNet50 · 替换分类头 · 冻结主干网络             |
| 模型训练  | 阶段一：分类头预训练 · 阶段二：分层学习率微调                    |
| 模型评估  | 迁移前后性能对比 · TensorBoard 可视化                  |


> **迁移策略** · 先冻结主干，仅训练替换后的分类头；待收敛后解冻顶层，采用**分层差异学习率**进行全局微调，底层使用极小学习率以保留通用特征，顶层使用较大学习率适配新任务。

---

### 5 · CIFAR-10 自定义 CNN + 分层采样

> `5.Cifar10Image_classification_cnn_relu+he+bn2d+AlphaDropout_StratifiedSampling_DataAugmentation.ipynb`

在 CIFAR-10 上训练自定义 CNN，重点引入 **分层采样（Stratified Sampling）** 策略，确保训练/验证集的类别分布与原始数据完全一致，每类验证样本精确为 500 张。


| 章节    | 内容                                                               |
| ----- | ---------------------------------------------------------------- |
| 环境导入  | 依赖库导入                                                            |
| 数据集准备 | CSV 解析 · 分层采样划分（训练 45000 / 验证 5000）· 自定义 `Cifar10Dataset` · 数据增强 |
| 构建模型  | 自定义 CNN（ReLU + He 初始化 + BN2D + AlphaDropout）                     |
| 模型训练  | 训练循环 · 最优模型保存                                                    |
| 模型评估  | 验证集准确率 · TensorBoard 可视化                                         |


---

### 6 · CIFAR-10 ResNet-34 手动实现

> `6.Cifar10Image_classification_ResNet34.ipynb`

从零手动复现 **ResNet-34** 完整架构，包含基本残差块（Basic Block）与跳跃连接（Skip Connection），在 CIFAR-10 上完成训练并生成预测提交文件。


| 章节    | 内容                                        |
| ----- | ----------------------------------------- |
| 环境导入  | 依赖库导入 · 设备检测                              |
| 数据集准备 | 标签 CSV 解析 · 训练/验证集划分 · 自定义 Dataset · 图像变换 |
| 构建模型  | 基本残差块 · 残差组 · ResNet-34 完整网络 · 结构可视化      |
| 模型训练  | 损失函数 · 优化器 · 训练循环 · 最优模型保存                |
| 模型评估  | 验证集准确率 · TensorBoard 可视化                  |
| 预测与提交 | 测试集推理 · 生成 `submission.csv`               |


> **残差连接原理** · 跳跃连接将输入 `x` 直接加到卷积输出 `F(x)` 上，使网络学习残差映射 `H(x) = F(x) + x`，解决深层网络的梯度消失与退化问题。
>
> 参考论文：*Deep Residual Learning for Image Recognition*（He et al., 2016）

---

### 7 · CIFAR-10 InceptionNet 手动实现

> `7.Cifar10Image_classification_Inception.ipynb`

从零手动复现简化版 **InceptionNet**，将不同尺度的并行卷积分支（1×1、3×3、5×5）与最大池化分支拼接为 Inception 模块，在 CIFAR-10 上进行十类图像分类。


| 章节    | 内容                                        |
| ----- | ----------------------------------------- |
| 环境导入  | 依赖库导入 · 设备检测                              |
| 数据集准备 | 标签 CSV 解析 · 分层采样划分 · 自定义 Dataset · 数据增强变换 |
| 构建模型  | Inception 模块（多尺度并行分支）· 完整 InceptionNet    |
| 模型训练  | 损失函数 · 优化器 · 训练循环 · 最优模型保存                |
| 模型评估  | 验证集准确率 · TensorBoard 可视化                  |
| 预测与提交 | 测试集推理 · 生成 `submission.csv`               |


> **Inception 模块原理** · 在同一层中并行使用多种尺度的卷积核（1×1、3×3、5×5）捕获不同感受野的特征，并通过 1×1 卷积进行降维控制参数量，最后拼接各分支输出。
>
> 参考论文：*Going Deeper with Convolutions*（Szegedy et al., 2014）

---

## 🗃 数据集

**FashionMNIST** · 由 Zalando 发布的服装图像数据集


| 属性   | 详情                                                                                            |
| ---- | --------------------------------------------------------------------------------------------- |
| 类别数  | 10（T-shirt · Trouser · Pullover · Dress · Coat · Sandal · Shirt · Sneaker · Bag · Ankle boot） |
| 训练集  | 60,000 张                                                                                      |
| 测试集  | 10,000 张                                                                                      |
| 图像尺寸 | 28 × 28 灰度图                                                                                   |


数据集通过 `torchvision.datasets.FashionMNIST` 首次运行时**自动下载**至 `data/FashionMNIST/`，无需手动准备。

---

**10 Monkey Species** · 来自 [Kaggle](https://www.kaggle.com/datasets/slothkong/10-monkey-species) 的十种猴子细粒度识别数据集


| 属性   | 详情                                |
| ---- | --------------------------------- |
| 类别数  | 10                                |
| 训练集  | 1,098 张                           |
| 验证集  | 272 张                             |
| 图像尺寸 | 400×300 彩色图（训练时 Resize 至 224×224） |


手动下载后解压至 `data/archive/`。

---

**CIFAR-10** · 来自 [Kaggle CIFAR-10 竞赛](https://www.kaggle.com/competitions/cifar-10/data)


| 属性   | 详情                                            |
| ---- | --------------------------------------------- |
| 类别数  | 10（飞机 · 汽车 · 鸟 · 猫 · 鹿 · 狗 · 青蛙 · 马 · 船 · 卡车） |
| 训练集  | 50,000 张                                      |
| 测试集  | 300,000 张（含干扰项）                               |
| 图像尺寸 | 32 × 32 彩色图                                   |


手动下载后解压至 `data/cifar-10/`。

---

## 🛤 学习路径

```
Notebook 1          Notebook 2
FashionMNIST    →   FashionMNIST
CNN + SELU          深度可分离 CNN
      │
      ▼
Notebook 3          Notebook 4
10 Monkeys      →   10 Monkeys
从头训练 CNN        ResNet50 迁移学习
      │
      ▼
Notebook 5          Notebook 6          Notebook 7
CIFAR-10        →   CIFAR-10        →   CIFAR-10
自定义 CNN          ResNet-34           InceptionNet
+ 分层采样          手动实现            手动实现
```

每个 Notebook 均为前一篇的进阶扩展，建议按编号顺序学习。

---

## ⚙️ 环境依赖


| 包名             | 版本           | 用途                  |
| -------------- | ------------ | ------------------- |
| `torch`        | 2.12.0+cu132 | 深度学习框架核心（CUDA 13.2） |
| `torchvision`  | 0.27.0+cu132 | 数据集加载与图像变换          |
| `torchviz`     | 0.0.3        | 网络结构可视化（06）         |
| `tensorboard`  | 2.20.0       | 训练过程实时可视化（全部）       |
| `matplotlib`   | 3.10.9       | 数据与结果绘图             |
| `numpy`        | 2.4.6        | 数值计算                |
| `pandas`       | 3.0.3        | 标签 CSV 读取与管理        |
| `scikit-learn` | 1.9.0        | 分层采样 · 评估指标计算       |
| `Pillow`       | 12.2.0       | 图像读取（PIL）           |
| `tqdm`         | 4.68.2       | 训练进度条显示             |


---

## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone https://github.com/your-username/DeepLearningWithCNN.git
cd DeepLearningWithCNN
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

> ⚠️ `torch` 与 `torchvision` 带有 `+cu132`（CUDA 13.2）后缀，如需 CPU 版本或其他 CUDA 版本，请参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 替换。

**3. 准备数据集**

- **FashionMNIST**：首次运行 Notebook 时自动下载，无需操作。
- **10 Monkey Species**：从 [Kaggle](https://www.kaggle.com/datasets/slothkong/10-monkey-species) 下载后解压至 `data/archive/`。
- **CIFAR-10**：从 [Kaggle](https://www.kaggle.com/competitions/cifar-10/data) 下载后解压至 `data/cifar-10/`。

**4. 启动 Jupyter**

```bash
jupyter notebook
```

按编号顺序打开 Notebook 开始学习即可。

---

## 📄 License

[MIT](LICENSE) © 2026