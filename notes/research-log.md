 # 研究日志

  ---

  ## 2026-08-05 — Day 1：环境搭建与核心概念

  ### 问题
  模型、参数、损失和训练之间到底是什么关系？一个完整的机器学习工作流是怎样串起来的？

  ### 假设
  我认为：模型是一个数学公式，参数是这个公式里可以调的系数，损失衡量公式算出来的结果和真
  实答案差多远，训练就是不断调参数让损失变小的循环。但我还不清楚具体怎么"调"——梯度下降到
  底是什么原理。

  ### 方法
  - 阅读李宏毅 2022 机器学习第 1 讲（导论）
  - 阅读李沐《动手学深度学习》前言 + 2.1 节
  - 用自己的话写 `day01-core-concepts.md`
  - 运行 `day01_check_env.py` 验证环境

  ### 结果
  （看完课程后填写：你学到了什么？有什么让你"啊哈"的瞬间？）

  ### 解释
  （你最初的假设对吗？哪里需要修正？）





## 2026-08-06 — Day 4：从零实现线性回归

  ### 问题
  不依赖 torch.nn，纯手写训练循环能否让模型从数据中学到参数？

  ### 假设
  如果梯度下降正确实现，初始化为 0 的 w 和 b 应该逐步逼近 3 和 2。

  ### 方法
  生成 y=3x+2+noise 合成数据，手写 Forward/Loss/Backward/Update 训练循环。
  lr=0.1, epochs=100。

  ### 结果
  w=3.006, b=2.018, loss=0.195。
  训练曲线显示 loss 从 13.73 单调下降至收敛。

  ### 解释
  梯度下降有效。剩余 loss 来自数据中注入的高斯噪声。
  w.grad 初始为负值，说明 w=0 时增大 w 能降低 loss——与参数更新方向一致。

  ---

  ## 2026-08-07 — Day 5：torch.nn 重写线性回归

  ### 问题
  把 Day 4 的手写版换成 PyTorch 高层 API（nn.Linear / nn.MSELoss / optim.SGD），结果会变吗？

  ### 假设
  nn.Linear(1,1) 内部就是 y = wx + b，只是把参数、损失、优化封装成对象。封装不该改变原理——
  用完全相同的种子和数据，收敛后应得到与 Day 4 相同的 w≈3.006、b≈2.018。

  ### 方法
  相同数据（seed=42，y=3x+2+noise），三行定义 model/criterion/optimizer，
  训练循环 4 步：forward → loss → backward → step。对比手写版。

  ### 结果
  学到的参数：w=3.006, b=2.018，最终 loss=0.1951——与 Day 4 手写版完全一致。
  代码从约 25 行降到约 15 行。

  ### 解释
  封装只是把手写步骤藏进 API：zero_grad 清梯度、backward 算梯度、step 更新参数，底层计算完全一样。
  梯度清零由 optimizer.zero_grad() 统一完成，替代了手写的 w.grad.zero_()。
  结论：框架封装 ≠ 改变原理。理解了手写原理后再用 API，才知道每一步在做什么。

## Day 6 — MLP 拟合非线性函数

### 问题
线性回归只能拟合直线。加入隐藏层和 ReLU 后，MLP 能否拟合波浪形的非线性关系？

### 方法
使用 200 个样本拟合 y = sin(2x)，模型结构为 Linear(1, hidden) → ReLU → Linear(hidden, 1)，损失函数为 MSELoss，优化器为 SGD。

### 结果
学习率为 0.05 时训练较快但中途有波动；学习率为 0.01 时曲线更平稳但 1000 轮内收敛更慢。
隐藏层宽度为 2、20、50 时，第 1000 轮 loss 分别为 0.3091、0.0952、0.0489。

### 解释
ReLU 让多层网络不再等价于单条直线，因此能拟合非线性函数。隐藏层太窄会欠拟合；增大宽度提高了模型容量。较大学习率收敛更快，但更容易发生波动。

### 下一步
学习分类任务：模型输出类别分数、Softmax 和交叉熵。

## 第一周总结：张量、梯度、回归与分类

## 1. 张量形状

- `(m, n)` 表示一个有 `m` 行、`n` 列的二维张量；通常可理解为 `m` 个样本，每个样本有 `n` 个特征。
- 矩阵乘法 `(m, n) @ (n, p)` 的结果形状是 `(m, p)`：左矩阵的列数必须等于右矩阵的行数。
- 实际例子：Day 6 的全部数据 `X` 形状是 `(400, 2)`，即 400 个样本、每个样本有 2 个特征；模型输出 logits 的形状是 `(400, 2)`，即每个样本对两个类别各有一个原始分数。

## 2. 梯度与训练

- 梯度表示：当模型参数发生极小变化时，损失会如何变化；它指出让损失上升最快的方向，优化时向反方向更新参数。
- `loss.backward()` 做：从损失开始反向传播，计算模型所有可训练参数的梯度，并存到对应参数的 `.grad` 中。
- `optimizer.step()` 做：读取各参数的梯度，按优化器规则和学习率更新参数，使下一次预测的损失倾向于更小。

## 3. 回归

- 回归预测的是连续数值，例如房价、温度，或 Day 4 中由 `y = 3x + 2 + noise` 产生的连续标签。
- Day 4 的线性回归中，模型的权重应接近 3，偏置应接近 2；损失曲线总体下降说明模型从数据中学到了该关系。

## 4. 分类

- logits 是模型为每个类别输出的原始分数，不是概率；数值越大，模型越倾向该类别。
- `argmax(logits, dim=1)` 对每个样本的类别维度取最大 logit 的位置，并把该位置作为预测类别。
- loss 与 accuracy 的区别：accuracy 只统计预测类别是否正确；交叉熵 loss 还衡量模型对真实类别的置信程度。即使准确率相同，过度自信地预测错误会产生更高 loss。

## 5. 三个仍有疑问的问题

1. 为什么学习率过大可能导致训练不收敛，而过小又会导致训练很慢？
2. softmax 如何把 logits 转成概率，为什么计算时需要关注数值稳定性？
3. 当训练集准确率很高、测试集准确率很低时，如何判断和缓解过拟合？


## Day 8：MLP 与非线性分类

MLP 使用 `Linear → ReLU → Linear`。ReLU 为模型加入非线性，使其可以学习圆形等弯曲分类边界。

本实验中，测试集准确率为 0.985，训练 loss 从约 0.75 下降至 0.0253，说明训练稳定收敛。预测图与真实图基本一致，少数错误通常在圆形边界附近。

当隐藏层从 8 个单元减少为 1 个单元时，模型表达能力明显下降。一个隐藏单元只能形成非常有限的分段线性边界，无法表示封闭的圆形边界；模型可能偏向预测为占多数的“圆外”类别，因此准确率不能单独证明模型真正学会了圆形结构。

## Day 10：优化器与学习率

在固定数据、模型、随机种子和训练轮数的公平比较下，Adam 的收敛速度明显快于 SGD。

SGD 的学习率 0.001 过小，200 个 epoch 后仍未充分学习；增大到 0.01 和 0.1 后效果改善，但最终仍低于 Adam。

本实验中 Adam lr=0.1 获得最低 loss（0.0302）和最高测试准确率（0.985）。它在早期有小幅波动，但后续稳定收敛。这个结果只说明该参数组合适合当前简单任务，不表示 Adam lr=0.1 对所有任务都最优。

选择优化器和学习率时，应同时比较收敛速度、曲线稳定性和验证/测试指标，而不是只看训练集分数。

## Day 11：CNN 基础

使用两层卷积、ReLU、最大池化和全连接层训练 Fashion-MNIST 分类器。

输入 shape 为 `[128, 1, 28, 28]`。两次卷积和池化后变为 `[128, 32, 7, 7]`，展平为 `[128, 1568]`，最终输出 `[128, 10]` 个类别 logits。

训练 loss 从 0.6112 降至 0.2788，测试准确率从 84.70% 提升至 89.03%。卷积层通过局部连接和参数共享提取图像特征；池化层压缩特征图并降低对精确位置的依赖。

## Day 10–12 必须掌握的新代码
## Day 10：优化器与学习率实验
1. SGD 与 Adam
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01)
必须懂：
model.parameters()：把模型中需要训练的权重和偏置交给优化器；
lr：学习率，每次参数更新的步长；
SGD：主要按当前梯度更新；
Adam：利用梯度历史自动调整更新，通常收敛更快。
2. 保持对照实验公平
torch.manual_seed(123)
model = build_model()
必须懂：每组实验都重设随机种子，确保模型从相同初始权重开始。否则不同结果可能只是初始化不同，而不能说明优化器或学习率造成差异。
3. 根据字符串选择不同优化器
if optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=lr)
必须懂：同一训练函数可以根据传入参数执行不同实验，避免复制六遍训练代码。
Day 11：CNN 与图像数据
1. 自动选择 CPU / GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallCNN().to(device)
images = images.to(device)
必须懂：
cuda 是 NVIDIA GPU；
没有可用 GPU 时自动使用 CPU；
模型和数据必须在同一个设备上。
2. Fashion-MNIST 数据集
train_dataset = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
)
必须懂：
train=True：训练集；
train=False：测试集；
download=True：第一次自动下载；
transform=transforms.ToTensor()：把图片转成 PyTorch Tensor。
3. DataLoader：分批读取数据
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
)
必须懂：
batch_size=128：一次训练 128 张图；
shuffle=True：每轮训练都打乱训练集；
测试集通常 shuffle=False；
num_workers=0：Windows 下安全地在主进程读取数据。
4. 二维卷积层
nn.Conv2d(1, 16, kernel_size=3, padding=1)
必须懂：
1：输入通道数，灰度图是 1
16：输出 16 张特征图
3：卷积核大小是 3×3
padding=1：补边，使高宽不缩小
5. 最大池化
nn.MaxPool2d(2)
必须懂：每个 2×2 区域只保留最大值，使特征图高宽减半。
28×28 → 14×14
14×14 → 7×7
6. 自定义 CNN 类
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
必须懂：
nn.Module：所有 PyTorch 模型的基础类；
__init__：定义网络层；
forward()：定义数据经过网络的路径；
super().__init__()：初始化父类，让 PyTorch 能登记和训练这些层。
7. 展平
nn.Flatten()
nn.Linear(32 * 7 * 7, 64)
必须懂：卷积后的 shape 是 [batch, 32, 7, 7]，全连接层只能接收一维特征，因此将每张图展平为 32×7×7=1568 个数字。
Day 12：测试、错分样本与混淆矩阵
1. 测试模式与关闭梯度
model.eval()

with torch.no_grad():
    logits = model(images)
必须懂：
model.eval()：切到测试模式；
torch.no_grad()：测试不需要梯度，节省内存和计算；
训练时使用 model.train()，测试时使用 model.eval()。
2. 把多个 batch 拼成完整测试结果
all_images.append(images)
all_true_labels.append(labels)
all_pred_labels.append(predictions)

all_images = torch.cat(all_images)
all_true_labels = torch.cat(all_true_labels)
all_pred_labels = torch.cat(all_pred_labels)
必须懂：测试集被分批处理；先把每批结果存入列表，再用 torch.cat() 拼起来，才能对全部 10000 张测试图片做统一分析。
3. 计算总体准确率
accuracy = (all_true_labels == all_pred_labels).float().mean().item()
必须懂：
真实标签 == 预测标签
→ 得到 True/False
→ float() 转为 1/0
→ mean() 得到正确比例
→ item() 转成普通数字
4. 创建混淆矩阵
confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)

for true_label, pred_label in zip(all_true_labels, all_pred_labels):
    confusion_matrix[true_label, pred_label] += 1
必须懂：
行 = 真实类别
列 = 预测类别
例如：
confusion_matrix[6, 0] += 1
表示一个真实 Shirt（6）被误判为 T-shirt/top（0）。
5. 找到所有错分图片
wrong_indices = torch.where(all_true_labels != all_pred_labels)[0]
必须懂：
真实标签 != 预测标签
→ 得到所有预测错误的位置
wrong_indices 中每个数字都对应一张被模型预测错的测试图片。
6. 从 Tensor 取单个值
true_label = all_true_labels[image_index].item()
必须懂：.item() 将只含一个数的 Tensor 变成普通 Python 整数，方便用作类别名称列表的索引。

## Day 12：CNN 错误分析

小型 CNN 在 Fashion-MNIST 测试集上达到 89.46% 准确率，共错分 1054 张图片。

混淆矩阵显示，错误主要集中在上装类别：真实 Shirt 被预测为 T-shirt/top 的次数最多（128 次），也常被预测为 Coat（100 次）和 Pullover（86 次）。真实 Pullover 也常被预测为 Coat（97 次）。

错分图片表明，28×28 灰度图缺少颜色与细粒度纹理信息，导致模型难以区分领口、袖型和衣襟等细节。鞋类中 Ankle boot 与 Sneaker 也存在一定混淆。

假设：将卷积通道数从 16/32 增加为 32/64，可让模型学习更多视觉特征，从而降低上装类别之间的混淆。下一步只改变卷积通道数，其他训练条件保持不变，验证该假设。