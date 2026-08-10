# PyTorch 模型保存与加载学习笔记

> 📖 来源：[PyTorch 官方教程 — Save and Load the Model](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html) ＋ 我的 ChatGPT 问答记录（第 5 节单独列出）
> 🗓️ 学习日期：2026-08-10
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → PPO → Isaac Lab → HIMLoco → 四足机器人部署

---

## 0. 前言：为什么要会保存/加载模型

训练好的模型（几百万个参数）不能每次用都重新训练。**保存 = 把学习成果写到磁盘；加载 = 下次直接恢复**。RL 里尤其重要：训练中断要能续训、部署到四足机器人上要能加载训练好的策略网络。

先准备好一个模型（教程用 VGG16 预训练权重，我本地已下载缓存）：

```python
import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')   # 加载预训练模型
```

---

## 1. 方式一：保存权重 `state_dict`（✅ 推荐）

### 保存

```python
torch.save(model.state_dict(), 'model_weights.pth')
```

`state_dict` 是什么？就是模型里**所有参数张量的名字→张量**的映射。VGG16 有 **32 个参数张量**，键名长这样：

```text
'features.0.weight'  → Tensor   (卷积层权重)
'features.0.bias'    → Tensor   (卷积层偏置)
'classifier.0.weight' → Tensor  (全连接层权重)
...
```

> 💡 呼应 Build-Model 笔记第 6 节：当时你的小网络 `named_parameters()` 只有 6 个参数张量，VGG16 是 32 个——同一个概念，模型大了只是数量变多。

### 加载（三步）

```python
model = models.vgg16()   # ① 先创建一个同结构的模型（不指定 weights，即未训练）
model.load_state_dict(
    torch.load('model_weights.pth', weights_only=True)   # ② 把权重填进去
)
model.eval()             # ③ 切到评估模式
```

**为什么必须先建同结构模型？** 因为**类定义了网络结构**，`state_dict` 只存了"参数"，没存"结构"。你得像搭骨架一样先把 `vgg16()` 建好，再把参数填进对应的层。**结构不匹配会直接报错**（键对不上）。

### 为什么 `model.eval()` 必须调？

> 教程原话：推理前一定要调用 `model.eval()`，把 dropout 和 batch normalization 层切到评估模式。不调用会得到不一致的推理结果。

---

## 2. 方式二：保存整个模型（⚠️ 不推荐 / legacy）

### 保存

```python
torch.save(model, 'model.pth')   # 传 model 本身，而不是 model.state_dict()
```

### 加载

```python
model = torch.load('model.pth', weights_only=False)
```

加载完 `model` 直接就是完整模型，**不用先建结构**。

### 为什么官方不推荐它？

`torch.save(model, ...)` 用的是 Python 的 **pickle** 序列化。它**不是**把模型类的所有代码塞进文件，而是存了"这个类是啥"的引用信息。所以：

> 加载时，**原来那个类定义必须能被 Python 找到**。

比如你自己写了 `class MyModel(nn.Module)`，`torch.save(model, 'model.pth')` 存了。以后如果你**删掉/改名了/移动了**这个类，再 `torch.load` 就可能失败。

而 `state_dict` 存的就是 `'features.0.weight' → Tensor` 这种纯数据，**不依赖类定义**，更稳定、更适合长期保存。

---

## 3. `weights_only=True` 还是 `False`？

| 加载场景 | 写法 | 原因 |
|----------|------|------|
| 加载 `state_dict`（推荐） | `torch.load('xxx.pth', weights_only=True)` | 只允许加载权重 Tensor，更安全 |
| 加载整个模型（legacy） | `torch.load('xxx.pth', weights_only=False)` | 保存的是完整 Python 对象，必须允许反序列化对象 |

> `weights_only=True` 是**现在的推荐做法**：它限制 pickle 只能还原"权重类的东西"，防止恶意 pickle 代码注入（安全考虑）。

---

## 4. 官方最佳实践总结

1. **保存 `state_dict`** 是官方推荐（比保存整个模型更稳）。
2. **加载时用 `weights_only=True`**。
3. **加载后必须 `model.eval()`** 再推理。

---

## 5. 我的疑问 × GPT 解答（本次学习重点记录）🔑

### ❓Q1：为什么我在终端跑，没看到教程里那一大串 VGG 结构输出？

**我的现象**：教程代码里有 `model.eval()`，教程截图能看到整个 VGG 结构被打印出来，但我在终端跑 `.py` 文件，啥都没打印。

**GPT 答**：
- **`model.eval()` 不是"打印模型"**，它的作用是**把模型切到评估模式，并返回 model 自己**。
- 教程大概率是在 **Jupyter Notebook** 里跑的：Jupyter 单元格**自动显示最后一行的返回值**，所以 `model.eval()` 返回的 model 被自动打印出来了。
- 你跑的是 `python SaveLoadModel.py`：**普通脚本不会自动显示表达式返回值**，所以没有输出，**你的代码其实完全正常**。
- 想看到输出就显式写：

```python
print(model)          # 或
print(model.eval())
```

**我的理解**：Jupyter 和 `.py` 脚本的"显式打印"行为不同；`model.eval()` 的返回值恰好是 model 本身，才让 Jupyter 顺带显示了它。

### ❓Q2：解释一下打印出来的 VGG16 结构

**GPT 答（结构说明书）**：

```text
VGG
├── features      ← 提取图片特征（卷积+ReLU+池化的堆叠）
├── avgpool       ← 把特征图调整成固定大小
└── classifier    ← 根据特征分类（全连接层）
```

**`features` 部分**：一组 `Conv2d → ReLU → ... → MaxPool2d` 循环堆叠。

- `Conv2d(3, 64, kernel_size=(3,3), ...)`：输入图片 3 个通道（RGB）→ 卷积 → 输出 **64 个特征图**。
- **通道数为什么越来越多**：`3 → 64 → 64 → 128 → 128 → 256 → ... → 512`。因为网络越往后提取的特征越复杂：
  - 前面卷积层 → 边缘、颜色、纹理
  - 中间卷积层 → 形状、局部结构
  - 后面卷积层 → 更复杂的物体特征
- **`MaxPool2d(kernel_size=2, stride=2)`**：把特征图尺寸减半。`224×224 → 112×112 → ... → 7×7`，VGG16 共 5 次池化。一张 `3×224×224` 的图经过 features 后变成约 **`512×7×7`**。

**`avgpool`**：`AdaptiveAvgPool2d(output_size=(7,7))`，确保最后是 `512×7×7`。

**`classifier` 部分**：全连接层堆叠。

- 关键数字：`512 × 7 × 7 = 25088` → 展开成 25088 维向量 → `Linear(25088, 4096)` → `Linear(4096, 4096)` → `Linear(4096, 1000)`。
- 为什么最后是 **1000**？因为你加载的是 `weights='IMAGENET1K_V1'`——ImageNet 预训练模型要区分 **1000 个类别**。所以模型最终输出 1000 个数（每个类别一个分数，比如猫 2.3、狗 8.7、汽车 -1.2……），取最大就是预测类别。

**完整数据流（背下来）**：

```text
图片 3×224×224
   ↓  Conv + ReLU（不断提取特征）
   ↓  MaxPool（不断缩小）
  512×7×7
   ↓  展开
  25088
   ↓  Linear → 4096 → 4096 → 1000
  1000 个分类结果
```

> 另外 `(0) (1) (2)` 这些数字只是 Sequential 里各层的**编号**，不是参数值。

### ❓Q3：两种保存方法对比，为什么官方推荐 `state_dict`？

**GPT 答（对比）**：

| | 方式一：`state_dict` ✅ | 方式二：整个模型 |
|---|---|---|
| 保存 | `torch.save(model.state_dict(), ...)` | `torch.save(model, ...)` |
| 存了什么 | **只存参数**（`'features.0.weight' → Tensor`） | **结构 + 参数**一起 |
| 加载 | 先建结构 → `load_state_dict(...)` | `torch.load(..., weights_only=False)` 直接用 |
| 依赖 | 不依赖类定义，稳定 | 用 pickle，**依赖原类定义存在** |

**为什么推荐 `state_dict`**：pickle 保存的不是"模型的所有 Python 代码"，而是与类相关的引用信息。你把类删除、改名、或代码移动位置，就可能加载失败。`state_dict` 是纯数据，更稳定、适合长期保存。

**你现在可以简单记成**：

```text
state_dict        = 只保存参数  = 推荐做法 ✅
torch.save(model) = 保存整个模型对象 = 使用方便，但依赖原类定义
```

> GPT 建议：对后面学深度学习和强化学习来说，**优先掌握和使用 `state_dict()`** 就够。

---

## 6. 本节小结

**核心要点（背下来）：**

1. **保存**：`torch.save(model.state_dict(), 'model_weights.pth')`（推荐）。
2. **加载**：先建同结构模型 → `load_state_dict(torch.load(..., weights_only=True))` → `model.eval()`。
3. **为什么先建结构**：类定义网络结构，state_dict 只存参数，不存结构。
4. **`model.eval()`**：切评估模式（影响 dropout/BN），且**返回 model 自己**。
5. **整个模型**：`torch.save(model, ...)` / `torch.load(..., weights_only=False)`，用 pickle，依赖类定义，legacy 不推荐。
6. **`weights_only=True`**：只允许加载权重 Tensor，更安全，是现在的推荐。
7. **VGG16 结构**：features（卷积提取特征）→ avgpool（固定尺寸）→ classifier（Linear 分类到 1000 类）；`512×7×7 = 25088` 是进全连接前的特征维数。
8. **state_dict 键名**：`<模块名>.<层号>.<weight|bias>`，VGG16 共 32 个参数张量。

**和后面课程的联系：**
- **RL / PPO 里**：策略网络训练好（或训到一半）都要 `torch.save(policy.state_dict(), 'checkpoint.pth')`，下次从 checkpoint 续训或部署。Isaac Lab / HIMLoco 里训练机器人策略就是这套。
- **部署到四足机器人**：上板子前把训练好的策略权重加载进来，只做推理（`model.eval()` + `no_grad()`）。
- **最佳实践复用**：`weights_only=True` 和 `eval()` 的习惯，RL 里同样适用。

---

## 7. 自测小练习

用你自己的小模型走一遍保存→加载→验证（不用 VGG16，秒完成）：

```python
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2),
        )
    def forward(self, x):
        return self.layers(x)

model = MyModel()
print("保存前 layers.0.weight:", model.layers[0].weight[0, :3])

# ① 保存 state_dict
torch.save(model.state_dict(), 'mymodel.pth')

# ② 建一个新模型，加载
loaded = MyModel()
loaded.load_state_dict(torch.load('mymodel.pth', weights_only=True))
loaded.eval()

# ③ 验证参数完全一致
same = all(torch.equal(a, b) for a, b in
           zip(model.state_dict().values(), loaded.state_dict().values()))
print("参数一致:", same)          # True

# ④ 改一下原模型的权重再存，看加载的是不是新的
model.layers[0].weight.data += 1.0
torch.save(model.state_dict(), 'mymodel.pth')
loaded2 = MyModel()
loaded2.load_state_dict(torch.load('mymodel.pth', weights_only=True))
print("重存后加载的参数是否变化:",
      not torch.equal(model.layers[0].weight, loaded.layers[0].weight))  # True

import os
os.remove('mymodel.pth')   # 清理
```
