# CIFAKE 频域特征提取流水线

本仓库是《数字图像处理》课程项目中成员 1 的交付内容，主要负责：

- CIFAKE 数据预处理
- 图像灰度化
- 2D FFT 频域变换
- 功率谱与对数功率谱计算
- 径向平均功率谱特征提取
- 输出可供后续 SVM 使用的一维频域特征数组

## 已完成工作

- 已将 CIFAKE 图像统一转换为 `32x32` 灰度图。
- 全项目标签约定已固定：
  - `REAL = 0`
  - `FAKE = 1`
- 已生成元数据文件：
  - `outputs/metadata/train_metadata.csv`
  - `outputs/metadata/test_metadata.csv`
- 已提取 FFT 径向平均对数功率谱特征：
  - `outputs/features/X_train_fft.npy`
  - `outputs/features/y_train.npy`
  - `outputs/features/X_test_fft.npy`
  - `outputs/features/y_test.npy`

## 最终特征规模

```text
X_train_fft shape: (100000, 23)
y_train shape: (100000,)
X_test_fft shape: (20000, 23)
y_test shape: (20000,)
```

标签数量：

```text
训练集：REAL 50000, FAKE 50000
测试集：REAL 10000, FAKE 10000
```

其中每张图片最终被转换为一个 `23` 维的一维频域特征向量。

## 环境安装

```powershell
python -m pip install -r requirements.txt
```

依赖库包括：

- `numpy`
- `pandas`
- `pillow`
- `tqdm`
- `matplotlib`

## 数据目录要求

如果需要重新运行预处理，请先将 CIFAKE 原始数据放到以下目录：

```text
data/raw/train/REAL/
data/raw/train/FAKE/
data/raw/test/REAL/
data/raw/test/FAKE/
```

注意：`data/raw/` 原始图片目录没有提交到 GitHub，因为数据量较大。

## 重新运行数据预处理

默认小样本调试运行：

```powershell
python src/preprocess.py
```

完整数据集预处理：

```powershell
python src/preprocess.py --full
```

检查灰度图效果：

```powershell
python src/check_gray.py --save outputs/metadata/gray_check.png --no-show
```

## 重新提取 FFT 特征

测试单张图片：

```powershell
python src/test_fft_single.py --save outputs/features/fft_single_check.png --no-show
```

小样本调试提取：

```powershell
python src/extract_fft_features.py --max-per-class 100
```

完整数据集 FFT 特征提取：

```powershell
python src/extract_fft_features.py
```

检查生成的特征文件：

```powershell
python src/check_fft_features.py
```

正常检查结果应满足：

```text
X_train has NaN: False
X_train has Inf: False
X_test has NaN: False
X_test has Inf: False
```

## 成员 2 使用方式

成员 2 可以直接读取成员 1 输出的 FFT 特征：

```python
import numpy as np

X_train_fft = np.load("outputs/features/X_train_fft.npy")
y_train = np.load("outputs/features/y_train.npy")
X_test_fft = np.load("outputs/features/X_test_fft.npy")
y_test = np.load("outputs/features/y_test.npy")
```

其中：

- `X_train_fft`：训练集 FFT 频域特征
- `y_train`：训练集标签
- `X_test_fft`：测试集 FFT 频域特征
- `y_test`：测试集标签

成员 2 后续可以将 `X_train_fft` 与自己提取的 LBP 特征进行拼接，然后输入 SVM 分类器训练。

## 成员 3 和成员 4 使用建议

成员 3 做前端展示时，可以复用：

- `src/fft_features.py`：单张图片 FFT 特征提取函数
- `src/test_fft_single.py`：频谱图和径向平均曲线可视化逻辑

成员 4 做实验报告时，可以使用以下结果描述成员 1 的工作：

> 本项目首先将图像统一转换为 `32x32` 灰度图，然后使用二维傅里叶变换将图像从空间域转换到频域。经过 `fftshift` 操作后，低频分量被移动到频谱中心。随后计算功率谱并使用 `log(1+x)` 压缩动态范围。最后以频谱中心为圆心，对不同半径上的对数功率谱进行平均，得到一维径向平均功率谱特征。

## Git 说明

本仓库应该提交：

- 源代码：`src/`
- 依赖文件：`requirements.txt`
- 项目说明：`README.md`
- 元数据 CSV：`outputs/metadata/*.csv`
- FFT 特征文件：`outputs/features/*.npy`

本仓库不提交：

- 原始 CIFAKE 图片：`data/raw/`
- 灰度图输出目录：`outputs/processed_gray/`
- 下载得到的压缩包：`archive*.zip`
- 可重新生成的检查图片：`outputs/**/*.png`

如果其他组员需要原始图片或灰度图，请通过网盘、U 盘或共享文件夹传递；GitHub 中保留代码和最终 FFT 特征即可。
