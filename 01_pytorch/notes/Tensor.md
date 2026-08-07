# PyTorch Tensor 学习笔记

> 📖 来源：[PyTorch 官方教程 — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
> 🗓️ 学习日期：2026-08-07
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → PPO → Isaac Lab → HIMLoco → 四足机器人部署

---

## 1. 什么是 Tensor（为什么第一个学它）

Tensor（张量）是 PyTorch 中所有数据的基本载体，它和 NumPy 的 `ndarray` 非常相似，区别在于：

- **支持 GPU 加速**（把数据搬到显卡上并行计算）
- **支持自动求导（Autograd）**——深度学习反向传播的基础
- 深度学习中一切数据（图像、状态观测、奖励、网络权重）都用 Tensor 表示

在四足机器人 RL 里，后面会看到：机器人的状态观测（Observation）是一个 Tensor，动作输出（Action）是一个 Tensor，网络权重也是一个 Tensor。**Tensor 是贯穿整个学习路线的地基。**

---

## 2. 导入

```python
import torch
import numpy as np   # 教程用 NumPy 做对比，顺便熟悉
```

---

## 3. 创建 Tensor（5 种方式）

### ① 直接从数据创建

dtype（数据类型）会自动推断：

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

### ② 从 NumPy 数组创建

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

> ⚠️ 注意：这种方式与 NumPy 数组**共享内存**（见第 6 节），改一个会改另一个。

### ③ 从另一个 Tensor 创建

`*_like` 系列会保留原 Tensor 的 shape / dtype / device，除非显式覆盖：

```python
x_ones = torch.ones_like(x_data)                          # 保留 x_data 的属性和 dtype
x_rand = torch.rand_like(x_data, dtype=torch.float)       # 覆盖 dtype 为 float
```

### ④ 用随机值 / 常量创建

`shape` 是一个**元组**（tuple），表示各维度的大小：

```python
shape = (2, 3)                 # 2 行 3 列
rand_tensor = torch.rand(shape)     # 均匀分布 [0, 1) 的随机数
ones_tensor  = torch.ones(shape)    # 全 1
zeros_tensor = torch.zeros(shape)   # 全 0
```

> 💡 上面是教程涉及的全部创建方式。常用但教程未展开的还有：`torch.arange()`（等差数列）、`torch.eye()`（单位矩阵）、`torch.full()`（填充指定值）、`torch.linspace()`（等间距）。后面做 RL 时 `arange` / `eye` 会经常遇到。

---

## 4. Tensor 属性

创建后可用三个属性查看它是什么：

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")        # torch.Size([3, 4])
print(f"Datatype of tensor: {tensor.dtype}")     # torch.float32
print(f"Device tensor is stored on: {tensor.device}")  # cpu
```

| 属性 | 含义 | 例子 |
|------|------|------|
| `shape` | 每个维度的大小（元组） | `torch.Size([3, 4])` |
| `dtype` | 数据类型 | `torch.float32`、`torch.int64` |
| `device` | 数据存放在哪个设备 | `cpu`、`cuda:0` |

> 💡 做 RL 时这三者经常要"对齐"：状态向量、动作向量、奖励的 dtype 和 device 必须一致，否则会报错。这是新手最常踩的坑之一。

---

## 5. Tensor 运算

PyTorch 提供了 **1200+** 种运算，教程重点介绍了以下几类：

### 5.1 索引与切片（和 NumPy 一致）

```python
tensor = torch.ones(4, 4)

tensor[:, 1] = 0                       # 把第 1 列（索引从 0 开始）全部置 0
print(f"First row:  {tensor[0]}")      # 第 0 行
print(f"First column: {tensor[:, 0]}") # 第 0 列
print(f"Last column: {tensor[..., -1]}")  # 最后一列，... 表示"前面所有维度"
```

> 💡 `...`（省略号）在更高维时很有用：比如一个 4D 张量，`t[..., -1]` 表示"不管前面几维，取最后一维的最后一个元素"。

### 5.2 拼接

```python
# 沿 dim=1（第 2 个维度，即列）拼接
t1 = torch.cat([tensor, tensor, tensor], dim=1)
```

> 💡 `torch.stack()` 和 `cat` 类似但**含义不同**：`cat` 沿已有维度拼接（不增加维度），`stack` 会**新增一个维度**。例如把 N 个 `(3, 4)` 的张量 `stack` 成 `(N, 3, 4)`。在 RL 里收集一批 transition 时经常用到 stack。

### 5.3 算术运算

**矩阵乘法**（三种写法等价）：

```python
y1 = tensor @ tensor.T                # @ 运算符
y2 = tensor.matmul(tensor.T)          # 方法调用
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)  # out 参数：把结果写入预先分配的 y3，省内存
```

**逐元素相乘**（Hadamard 积）：

```python
z1 = tensor * tensor                  # * 运算符
z2 = tensor.mul(tensor)               # 方法调用
```

> ⚠️ 关键区分：`@` / `.matmul()` 是**矩阵乘法**（维度要匹配，类似线性代数 AB）；`*` / `.mul()` 是**逐元素相乘**（形状要广播兼容）。RL 里二者都常用，搞混会得到完全错误的结果且很难发现。

### 5.4 单元素 Tensor → Python 数值

`sum()` 返回一个 0 维 Tensor，用 `.item()` 转成 Python 标量：

```python
agg = tensor.sum()
agg_item = agg.item()    # 12.0，类型是 <class 'float'>
```

> 💡 `.item()` 只能用于**单元素** Tensor。多元素会报错。在 RL 训练里计算单步奖励标量时常这么用。

### 5.5 原地操作（In-place）

以**下划线 `_` 结尾**的运算会直接修改原 Tensor：

```python
tensor.add_(5)   # 等价于 tensor = tensor + 5，但直接改原值
```

其他例子：`x.copy_(y)`、`x.t_()`（转置）。

> ⚠️ 教程明确提醒：原地操作虽然**省内存**，但因为**立即丢失计算历史**，在自动求导时会出问题，**不推荐使用**。经验：默认写非原地版本，确需省内存时才考虑。

---

## 6. 与 NumPy 的桥接（重点易错）

教程原话：**CPU 上的 Tensor 和 NumPy 数组可以共享底层内存，改一个，另一个也会变。**

### Tensor → NumPy

```python
t = torch.ones(5)
n = t.numpy()          # n 与 t 共享内存
t.add_(1)              # t 变了，n 也变成 [2., 2., 2., 2., 2.]
```

### NumPy → Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)    # n 变了，t 也变了（注意 dtype 变成 float64）
```

> ⚠️ 关键前提：**只有 CPU 上的 Tensor** 才和 NumPy 共享内存。GPU 上的 Tensor 要先 `.cpu()` 再转。这一点在 Isaac Lab / HIMLoco 里做数据记录、可视化时经常要处理。

---

## 7. 设备与 GPU 运算

- 默认情况下 Tensor 创建在 **CPU** 上，要显式搬到加速器上。
- 支持的加速器：CUDA（NVIDIA 显卡）、MPS（Apple 芯片）、MTIA、XPU。
- 教程用的是较新的统一接口 `torch.accelerator`：

```python
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
```

> 💡 教程用的是新版统一 API。很多旧代码用 `torch.cuda.is_available()` + `tensor.cuda()` / `.to('cuda')`，功能相同，只是写法旧。`to(device)` 是通用写法，推荐优先掌握。
>
> ⚠️ 教程提醒：**跨设备拷贝大的 Tensor 很耗时且占内存**。训练中应把数据一次性放到目标设备，避免循环里反复搬运——这正好是 RL 训练里要养成的习惯（网络、观测、动作都放同一个 device）。

---

## 8. 本节小结

**核心要点（背下来）：**

1. Tensor = 能上 GPU + 能自动求导的 NumPy 数组，是深度学习/RL 的数据基元。
2. 创建 5 式：直接 `tensor()`、`from_numpy()`、`*_like`、`rand/ones/zeros(shape)`、其它工具（`arange/eye/full`）。
3. 三属性要齐：`shape`、`dtype`、`device`，做 RL 时三者必须对齐。
4. 运算分两类别搞混：`@`/`matmul` = 矩阵乘法，`*`/`mul` = 逐元素乘法。
5. `.item()` 只能用于单元素 Tensor。
6. `_` 结尾 = 原地操作，省内存但会丢求导历史，默认不要用。
7. CPU Tensor 与 NumPy **共享内存**，改一个另一个也变。
8. 默认在 CPU，需要时 `tensor.to(device)` 搬 GPU，别在循环里反复搬。

**和后面课程的联系：**
- 下一个要学的是 **Dataset / DataLoader** —— 数据组织方式，Tensor 是它们的核心数据单元。
- 再往后 **Autograd** 自动求导 —— 建立在 Tensor 的 `requires_grad` 机制上。
- 在 RL 里，机器人的 state、action、reward 全是 Tensor；`stack` 收集经验、`item()` 取奖励标量、`to(device)` 对齐设备，都会用到这里的基础。

---

## 9. 自测小练习

建议动手跑一遍验证理解：

```python
import torch
import numpy as np

# 1. 创建 3x4 的随机 tensor，打印 shape/dtype/device
t = torch.rand(3, 4)
print(t.shape, t.dtype, t.device)

# 2. 沿 dim=0 拼接两个 2x3 的 ones，看结果的 shape（应该是 4x3）
a = torch.ones(2, 3)
b = torch.ones(2, 3)
print(torch.cat([a, b], dim=0).shape)   # torch.Size([4, 3])

# 3. 区别矩阵乘法与逐元素乘法
x = torch.ones(2, 3)
print((x @ x.T).shape)      # (2, 2) 矩阵乘法
print((x * x).shape)        # (2, 3) 逐元素

# 4. 验证与 NumPy 共享内存
t = torch.ones(5)
n = t.numpy()
t.add_(1)
print(n)                    # [2., 2., 2., 2., 2.]

# 5. 单元素转标量
print(t.sum().item())
```
