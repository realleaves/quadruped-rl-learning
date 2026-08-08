# PyTorch 构建神经网络学习笔记（nn.Module）

> 📖 来源：[PyTorch 官方教程 — Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
> 🗓️ 学习日期：2026-08-08
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → PPO → Isaac Lab → HIMLoco → 四足机器人部署

---

## 1. 先看整体：一个神经网络长什么样

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

教程用 FashionMNIST 数据集：输入是 28×28 的灰度图片，要分类成 10 类衣服。

> 💡 先记住一句核心思想：**神经网络本身就是由其他模块（层）组成的模块**。`torch.nn` 里的一切（层、激活函数、整个网络）都继承自 `nn.Module`。所以"搭网络"其实就是**把层像积木一样嵌套组合**。

---

## 2. 选择训练设备

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
```

- 有 GPU（CUDA / MPS / MTIA / XPU）就用加速器，否则回落到 `cpu`。
- 这是新版统一 API，和之前 Tensor 笔记里 `torch.accelerator` 一脉相承。旧代码常用 `torch.device("cuda" if torch.cuda.is_available() else "cpu")`。
- 写好这行后，`model.to(device)`、数据 `.to(device)` 统一用它，避免硬编码。

---

## 3. 定义神经网络：`nn.Module` 子类

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),   # 784 输入 → 512
            nn.ReLU(),
            nn.Linear(512, 512),     # 512 → 512
            nn.ReLU(),
            nn.Linear(512, 10),      # 512 → 10（10 个类别）
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
```

**两个必须理解的关键点：**

1. **`__init__` 里定义层**：把各层作为属性挂到 `self` 上。`nn.Module` 会自动追踪这些子模块和参数（第 6 节会看到）。
2. **`forward` 里定义数据流向**：输入 `x` 依次经过 flatten → 堆叠层 → 返回 logits。**前向传播的"路线图"完全由 `forward` 决定。**

创建实例并搬到设备上：

```python
model = NeuralNetwork().to(device)
print(model)
```

打印出来的结构会像这样（可以看到嵌套层级）：

```text
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

---

## 4. 使用模型做预测

```python
X = torch.rand(1, 28, 28, device=device)   # 一张 28×28 的图，batch=1
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)    # logits → 概率
y_pred = pred_probab.argmax(1)             # 取概率最大的类别
print(f"Predicted class: {y_pred}")
```

三个步骤的语义链：
1. `model(X)` → 得到 **logits**：每类一个原始分数，范围是 (-∞, +∞)。
2. `Softmax(dim=1)` → 把 logits 缩放成 **概率**，所有类之和为 1。
3. `.argmax(1)` → 取概率最大的那一类的**索引**，就是预测结果。

> ⚠️ **教程明确警告：不要直接调用 `model.forward(x)`！** 应该调用 `model(x)`。直接调 `forward()` 会跳过 PyTorch 在背后做的钩子/缓存等操作，容易出错。记住：**网络对象本身是可调用的（callable），`forward` 由它内部去调用。**

---

## 5. 模型层逐一看（Model Layers）

教程准备了一张小批次图做演示：`input_image = torch.rand(3, 28, 28)`，形状 `[3, 28, 28]`（3 张图，每张 28×28）。

### 5.1 `nn.Flatten` —— 展平

把每张 2D 图像拉平成 1D 数组，**保留 batch 维**：

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())   # torch.Size([3, 784])  ← 3 张图 × 784 个像素
```

> 💡 `nn.Flatten(start_dim=1, end_dim=-1)` 默认从第 1 维展到最后一维，所以 batch 维（第 0 维）被保留。这就是为什么 3 张图得到 `[3, 784]` 而不是 `[2352]`。

### 5.2 `nn.Linear` —— 线性变换

`y = x·Wᵀ + b`，权重和偏置是自动初始化的参数：

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())   # torch.Size([3, 20])
```

输入 784 维 → 输出 20 维。`in_features` 必须和上一层输出匹配，这是新手最容易算错的地方。

### 5.3 `nn.ReLU` —— 非线性激活

```python
hidden1 = nn.ReLU()(hidden1)
```

ReLU 就是 `max(0, x)`：负数变 0，正数不变。教程输出里能看到 `-0.5757 → 0.0000`，而 `0.2219` 保持原样，且带上了 `grad_fn=<ReluBackward0>`（说明进入了计算图，可求导）。

> 💡 **为什么必须有非线性激活？** 如果全是线性层，多层叠在一起仍然等价于一层线性变换（线性函数复合还是线性）。ReLU 这类非线性激活让网络能拟合复杂函数——这是"深度学习"能 work 的根本原因之一。

### 5.4 `nn.Sequential` —— 顺序容器

**按定义顺序依次把数据传给每个模块**，是"堆层"的快捷方式：

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
logits = seq_modules(input_image)
```

> ⚠️ 注意这里 `input_image` 是 `[3, 28, 28]`，直接进 Sequential，内部第一个 `flatten` 会先展平成 `[3, 784]`，再依次过后面的层。顺序即执行顺序。

### 5.5 `nn.Softmax` —— 概率化

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

`dim=1` 表示**沿着类别这一维求和为 1**（把每张图的 10 个分数变成 10 个概率）。

> 💡 **对 RL 至关重要的一点**：Softmax 不仅是分类网络最后一步，也是**策略网络（Policy Network）输出动作概率分布**的标准做法。后面学 PPO 时，策略网络输出的正是 `Softmax` 后的动作概率，再用它做采样。现在这个印象先种下。

---

## 6. 模型参数

**核心机制**：`nn.Module` 会自动追踪所有 `self` 上定义的参数，用 `parameters()` 或 `named_parameters()` 就能拿到：

```python
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

教程模型共有 **6 个参数张量**（3 个线性层的 weight + bias）：

| 参数名 | 形状 |
|--------|------|
| `linear_relu_stack.0.weight` | [512, 784] |
| `linear_relu_stack.0.bias` | [512] |
| `linear_relu_stack.2.weight` | [512, 512] |
| `linear_relu_stack.2.bias` | [512] |
| `linear_relu_stack.4.weight` | [10, 512] |
| `linear_relu_stack.4.bias` | [10] |

> 💡 参数命名 = **`<属性名>.<层索引>.<weight/bias>`**。比如 `linear_relu_stack.2.weight` 指 Sequential 里第 2 个模块（中间那层 Linear）的权重。参数值带 `grad_fn=<SliceBackward0>`，说明**这些参数在计算图里、训练时会更新**。
>
> 后面训练神经网络时，`optimizer` 更新的就是这些参数；做 RL 时你也常要 `model.parameters()` 传给优化器。

---

## 7. 本节小结

**核心要点（背下来）：**

1. 一切皆是 `nn.Module`：层是模块，网络也是模块，靠**嵌套组合**搭结构。
2. 自定义网络三步：`class MyNet(nn.Module)` → `__init__` 里挂层 → `forward` 里写数据流向。
3. **调用 `model(x)`，不要直接调 `model.forward(x)`。**
4. 用 `model.to(device)` 和 `X.to(device)` 把模型和数据搬到同一设备。
5. 推理链：`logits → Softmax(dim=1) → argmax(1)` 得到预测类别。
6. 五层件：`Flatten`（展平保 batch）、`Linear`（y = xWᵀ+b）、`ReLU`（max(0,x)，提供非线性）、`Sequential`（顺序容器）、`Softmax`（分数→概率）。
7. 参数自动追踪：`named_parameters()` 查看所有 weight/bias，命名规则是 `<属性>.<索引>.<weight|bias>`。
8. `Linear` 的 `in_features` 必须匹配上一层输出——维度错了立刻报错，是高频踩坑点。

**和后面课程的联系：**
- 接下来学 **Loss Function / Optimizer**：loss 根据这里 `forward` 的 logits 计算，optimizer 更新这里的 `parameters()`。
- 在 **RL** 里：Actor（策略）网络和 Critic（价值）网络都是 `nn.Module` 子类；`Softmax` 输出动作概率（PPO 的核心）；`named_parameters()` 在调试、剪枝、加载部分权重时非常有用。
- 到了 **Isaac Lab / HIMLoco**，你会看到结构类似但更大的策略网络（输入是高维观测、输出是 12 个关节动作），原理和这节课一模一样。

---

## 8. 自测小练习

不需要下载数据集也能跑（用随机张量代替真实图片），建议动手验证：

```python
import torch
from torch import nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# 1. 定义一个和教程一样的两隐藏层网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = NeuralNetwork().to(device)
print(model)                      # 观察嵌套结构

# 2. 一张随机"图"走完整推理链
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred = nn.Softmax(dim=1)(logits).argmax(1)
print("logits shape:", logits.shape)   # [1, 10]
print("pred class:", pred)

# 3. 数一数总参数个数
total = sum(p.numel() for p in model.parameters())
print("total params:", total)
# 期望：784*512 + 512 + 512*512 + 512 + 10*512 + 10 = 669,706

# 4. 看参数名与形状
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}")
```
