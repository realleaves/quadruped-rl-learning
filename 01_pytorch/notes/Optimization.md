# PyTorch 模型参数优化（训练）学习笔记

> 📖 来源：[PyTorch 官方教程 — Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) ＋ 我的 ChatGPT 问答记录（第 7 节单独列出）＋ 我的真实训练输出
> 🗓️ 学习日期：2026-08-10
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → PPO → Isaac Lab → HIMLoco → 四足机器人部署

---

## 0. 前言：这一章把前面的知识串起来了

前面学的 Tensor → Neural Network（nn.Module）→ Autograd 在本章终于**汇合成一个能跑的完整训练流程**：

```text
数据 X → model(X) → 预测 pred → loss_fn(pred, y) → loss
        → loss.backward() → 梯度 → optimizer.step() → 改参数
        → 重复很多 batch → 完成一个 epoch → test_loop 检查效果 → 下一个 epoch
```

训练过程本质上是一个**迭代优化循环**：每个 epoch 两件事——**train_loop**（在训练集上收敛参数）+ **test_loop**（在测试集上检查效果，不更新参数）。

---

## 1. 超参数（Hyperparameters）

> 超参数 = 可调节的、用来控制模型优化过程的参数。它们不是训练学出来的，而是**你手动设定的**。

```python
learning_rate = 1e-3   # 0.001
batch_size = 64
epochs = 5
```

| 超参数 | 控制什么 | 直觉理解 |
|--------|----------|----------|
| `epochs` | 遍历整个数据集的**次数** | 学几遍 |
| `batch_size` | 每次更新参数前**过多少样本** | 一次看多少张图再改一次参数 |
| `learning_rate` | 每次更新参数的**步长** | 每次参数挪多大 |

> 💡 教程超参数区写 `epochs=5`，但主循环里用了 `epochs=10`，教程原话："Feel free to increase the number of epochs"——加大轮数能看到模型持续变好。我实际跑了 10 轮，验证了这一点。

---

## 2. 损失函数（Loss Function）

> 损失函数衡量**预测结果和目标值的差异**，训练就是要去最小化它。

```python
loss_fn = nn.CrossEntropyLoss()
```

- FashionMNIST 是 **10 分类**任务 → 用 `CrossEntropyLoss`（交叉熵）。
- 它内部 = `nn.LogSoftmax` + `nn.NLLLoss`（把 logits 归一化后再算误差）。
- 我们**直接传 logits**（模型的原始输出）进去就行，CrossEntropyLoss 会自己处理。
- 其他常见选择：回归用 `nn.MSELoss`，分类也可以用 `nn.NLLLoss`。

---

## 3. 优化器（Optimizer）

> 优化 = 调整模型参数，让每一步的错误变小。

```python
optimizer = torch.optim.SGD(
    model.parameters(),   # 注册所有可训练参数
    lr=learning_rate,     # 学习率
)
```

`SGD` 会拿到 `model.parameters()`（上一章 `named_parameters()` 那 6 个参数张量）和 lr。

**训练循环里的三件套**（顺序很重要）：

```python
optimizer.zero_grad()   # ① 梯度清零：防止上一轮梯度累加（Autograd 章讲的"梯度默认累加"）
loss.backward()         # ② 反向传播：把梯度算进每个参数的 .grad
optimizer.step()        # ③ 更新参数：用 .grad 和 lr 修改参数
```

> 💡 顺序为什么是 zero → backward → step？因为 backward 是**累加**梯度的，如果不清零，下一轮的梯度会叠在上一轮上面（回顾 Autograd 笔记第 7.4 节），参数更新就会错乱。

---

## 4. `train_loop` 逐行解读

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)                    # 训练集总数，60000
    model.train()                                     # 切换训练模式（BN/Dropout 需要）
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)                               # 前向预测
        loss = loss_fn(pred, y)                       # 算误差

        loss.backward()                               # 反向传播
        optimizer.step()                              # 更新参数
        optimizer.zero_grad()                         # 清零梯度

        if batch % 100 == 0:                          # 每 100 个 batch 打印一次
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

**关键点：**

1. `model.train()`：切到训练模式。本模型没有 BN/Dropout，但这行是**最佳实践**，先养成习惯。
2. **三件套顺序**：`backward → step → zero_grad`（清零放后面也可以，效果一样；教程这里放最后）。
3. `if batch % 100 == 0`：**不是每个 batch 都打印**，每 100 个 batch 打一次。`batch_size=64`，所以每 100 个 batch ≈ `100×64 = 6400` 张图片 → 显示的数字是 `64, 6464, 12864, 19264 ...`。
4. `[64/60000]` 的含义：**当前已处理的图片数 / 训练集总数**。

---

## 5. `test_loop` 逐行解读

```python
def test_loop(dataloader, model, loss_fn):
    model.eval()                                      # 切换评估模式
    size = len(dataloader.dataset)                    # 测试集总数，10000
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():                             # 评估不追踪梯度（省内存+省时间）
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()      # 累计 loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches                          # 平均 loss
    correct /= size                                   # 正确率 = 猜对的 / 总数
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

**关键点：**

1. `model.eval()`：切到评估模式（BN/Dropout 行为不同），最佳实践。
2. **`torch.no_grad()`**：测试阶段不反传、不需要梯度，包上它既省内存又加速（呼应 Autograd 笔记第 6 节）。
3. **Accuracy 怎么算的**：`pred.argmax(1)` 取预测类别 → 和真实 `y` 比是否相等 → `type(torch.float).sum()` 数出正确的个数 → 除以总数。
4. `correct += (pred.argmax(1) == y).type(torch.float).sum().item()` 这句是一个小技巧：布尔比较 → 转 float（True=1.0, False=0.0）→ 求和，一行算出本 batch 猜对几个。

---

## 6. 训练主循环

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)   # 先训练
    test_loop(test_dataloader, model, loss_fn)                # 再测试
print("Done!")
```

**结构**：`t` 依次取 0,1,…,9 → 打印 `Epoch 1`~`Epoch 10`。每个 epoch：

```text
Epoch 1
├─ 训练 60000 张图片（938 个 batch）
└─ 测试模型（10000 张）

Epoch 2
├─ 再训练 60000 张
└─ 再测试
...
```

---

## 7. 我的疑问 × GPT 解答（本次学习重点记录）🔑

### ❓Q1：这段代码整体在干什么？

**GPT 答（压缩成完整链条）：**

```text
数据 X
 ↓
model(X)
 ↓
预测 pred
 ↓
loss_fn(pred, y)
 ↓
loss
 ↓
loss.backward()
 ↓
得到梯度
 ↓
optimizer.step()
 ↓
修改参数
 ↓
重复很多 batch
 ↓
完成一个 epoch
 ↓
test_loop 检查学习效果
 ↓
继续下一个 epoch
```

**GPT 强调**：到这里，前面学的 Tensor → Neural Network → Autograd → Optimizer **基本串起来了**。`CrossEntropyLoss` 衡量预测和真实差距；`optimizer` 用 SGD 根据梯度改参数；每个 epoch = 训练一遍 + 测试一遍。

### ❓Q2：讲一下我的训练输出（10 个 epoch 的真实数据）

**我的真实输出（关键数字）：**

| Epoch | Accuracy | Avg loss |
|-------|----------|----------|
| 1 | 47.4% | 2.167 |
| 2 | 62.3% | 1.911 |
| 3 | 61.6% | 1.541 |
| 4 | 63.4% | 1.266 |
| 5 | 65.1% | 1.096 |
| 6 | 66.1% | 0.987 |
| 7 | 67.2% | 0.914 |
| 8 | 68.2% | 0.861 |
| 9 | 69.2% | 0.821 |
| 10 | 70.5% | 0.790 |

**GPT 解读：**

1. **趋势正确**：Loss 2.17 → 0.79 一直降，Accuracy 47.4% → 70.5% 一直升 → **模型确实在学会识别衣服**。
2. **`loss: 2.300684 [64/60000]` 的含义**：`2.300684` 是**当前这一个 batch（64 张图）**的 loss；`[64/60000]` 是"已经处理 64 张 / 总共 60000 张"。因为 `if batch % 100 == 0`，每 100 个 batch 打印一次，所以数字是 64、6464、12864、19264…
3. **为什么 batch loss 会忽高忽低**（比如 `1.158950 → 1.269915` 反而变大了）？因为打印的是**单个 batch** 的 loss，有的 batch 简单、有的 batch 难，所以 `↓ ↑ ↓ ↓ ↑ ↓ ↑` 都正常。**要看整体趋势**，不要盯单点。
4. **为什么 Accuracy 有小波动**（E2 62.3% → E3 61.6% 掉了）？训练曲线本来就不是整齐的 `50→55→60→65`，而是 `50→62→61→63→65…` 这种带波动的上升。
5. **总结**：从"参数随机"（47.4%）到"训练 10 遍"（70.5%），这是你第一次真正看到 **梯度 → 改参数 → Loss 下降 → 预测越来越准** 整个流程实际工作起来。

### ❓Q3：怎么提高准确率？

**GPT 建议（先不要改网络结构，只实验超参数）：**

| 实验 | 修改 | 其他 | 目的 |
|------|------|------|------|
| 基准 | lr=1e-3, epochs=10, SGD | —— | 70.5% |
| 实验 1 | **epochs=20** | 其他不动 | 看模型是否还没训练充分 |
| 实验 2 | **lr=1e-2** | 其他不动 | SGD 的 1e-3 更新太慢 |
| 实验 3 | **Adam** 替代 SGD | lr=1e-3 | Adam 通常更快到较好结果 |

**方法论要点：**
- ⚠️ **一次只改一个变量**——否则你无法判断到底哪个改动导致的准确率变化。
- 把 `shuffle=True` 加进 `DataLoader`，每个 epoch 开始打乱训练顺序（训练集一般建议这样）。
- **epochs 不是越大越好**——以后会学到**过拟合**（在训练集上太好、测试集变差）。
- 等做到 80%~90% 时，再往上就需要理解 **CNN（卷积网络）为什么比全连接更适合图片**，而不是单纯调参了。

---

## 8. 本节小结

**核心要点（背下来）：**

1. **训练 = 迭代优化循环**：每 epoch 先 `train_loop` 收敛参数，再 `test_loop` 检查效果。
2. **三个超参数**：`epochs`（学几遍）、`batch_size`（一次看几张再更新）、`learning_rate`（步子多大）。
3. **三件套顺序**：`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`。零梯度必须做，因为梯度默认累加。
4. `model.train()` / `model.eval()`：训练/评估模式切换，是最佳实践。
5. `test_loop` 用 `torch.no_grad()`：评估不需要梯度，省内存省时间。
6. **Accuracy 公式**：`(pred.argmax(1) == y).type(torch.float).sum() / 总数`。
7. **打印规律**：`if batch % 100 == 0` → 每 100 个 batch 打印一次，`[已处理/总数]`。
8. **看趋势不要看单点**：batch loss 忽高忽低、accuracy 小波动都正常，看 epoch 间整体趋势。
9. **提高准确率的正确姿势**：一次只改一个超参数（epochs / lr / 优化器），记录对比。
10. **还没学到的坑**：过拟合（epochs 过大）、网络结构升级（CNN）。

**和后面课程的联系：**
- 这个训练循环就是**所有深度学习/RL 训练器的骨架**。
- 在 **PPO** 里：也是 `model(x)` → 算 loss（策略损失 + 价值损失）→ `backward()` → `step()`，只是数据来源从"图片数据集"变成"机器人采样的轨迹（states, actions, rewards）"。
- **Isaac Lab / HIMLoco** 里你会看到同样结构但规模大得多的训练循环，超参数（lr、batch_size、epochs）调法也一样。

---

## 9. 自测小练习

跑一个 **mini 训练实验**，亲手验证三件套和 Accuracy 计算：

```python
import torch
from torch import nn

torch.manual_seed(0)

# 造一个假的"分类"任务：200 个样本，特征 8 维，标签 3 类
X = torch.randn(200, 8)
y = (X[:, 0] * 3 + X[:, 1] * 2 > 0).long()          # 简单的线性可分规则

dataloader = [(X[i:i+64], y[i:i+64]) for i in range(0, 200, 64)]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3),
        )
    def forward(self, x):
        return self.layers(x)

model = Net()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 三件套：zero_grad → backward → step
for epoch in range(20):
    model.train()
    for Xb, yb in dataloader:
        loss = loss_fn(model(Xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试：Accuracy 计算
model.eval()
with torch.no_grad():
    pred = model(X)
    correct = (pred.argmax(1) == y).type(torch.float).sum().item()
print(f"Accuracy: {100 * correct / len(X):.1f}%")   # 训练 20 轮后应该接近 90%+

# 练习：把 epochs 从 20 改成 5，看 accuracy 差多少 → 理解 epochs 的作用
```
