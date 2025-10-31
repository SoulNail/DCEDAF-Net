# DCEDAF-Net

DCEDAF-Net Code Repository

## 简介

DCEDAF-Net 是论文实现的代码仓库。包含了模型训练、测试和推理的完整流程。

## 环境要求

- Python 3.12
- Conda (推荐用于环境管理)
- CUDA (如果使用GPU训练)

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/SoulNail/DCEDAF-Net.git
cd DCEDAF-Net
```

### 2. 创建并激活 Conda 环境

```bash
conda create -n dcedaf python=3.12
conda activate dcedaf
```

### 3. 安装依赖

**提示**: 建议先安装 PyTorch，然后再安装其他依赖包。

#### 安装 PyTorch

根据您的系统和CUDA版本，从 [PyTorch官网](https://pytorch.org/get-started/locally/) 选择合适的安装命令。

项目使用的是pytorch2.8+cuda12.9的版本

#### 安装其他依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
DCEDAF-Net/
├── auc_analysis/          # AUC分析相关代码
├── configs/               # 配置文件
├── exp/                   # 实验结果和日志
│   └── DCEDAF-Net/
├── models/                # 模型定义
├── .gitignore            # Git忽略文件配置
├── inference.py          # 推理脚本
├── main.py               # 主程序入口
├── main_test.py          # 测试主程序
├── requirements.txt      # 项目依赖
└── train.py              # main调用的训练脚本
```

## 使用方法

### 训练模型

```bash
python main.py
```

### 模型推理
把权重文件放在exp/DCEDAF-Net/2025-04-14 015351/MyModel_modify-Epoch_2186-f1_score_0.832594694653013后，运行以下指令

```bash
python main_test.py
```

### 配置文件

可以在 `configs/` 目录下修改配置文件来调整模型参数、训练设置等。

## paper中训练好的模型pt文件(DRIVE)可以从以下链接获取

[OneDrive链接](https://1drv.ms/u/c/bdd64c2b3becb118/ERXcszvMZ0dEs0Qzn6PdYdABIvU1z8owo1SbuGp41XyQ8Q)

## 联系方式

如有问题或建议，请通过 GitHub Issues 提交或通过邮件harukazew@outlook.com联系。
