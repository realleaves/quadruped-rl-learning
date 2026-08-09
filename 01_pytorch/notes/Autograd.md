# PyTorch Autograd 自动求导学习笔记

> 📖 来源：[PyTorch 官方教程 — Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) ＋ 我的 ChatGPT 问答记录（本节末尾单独列出）
> 🗓️ 学习日期：2026-08-08
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → PPO → Isaac Lab → HIMLoco → 四足机器人部署

---

## 1. autograd 是什么

PyTorch 内置了一个**自动微分引擎**，叫 `torch.autograd`。它能让**任意计算图**自动求梯度，正是它支撑起了**反向传播（back-propagation）**：

> 训练时，参数（模型权重）根据 loss 对它的梯度被调整。autograd 干的就是：你只管**前向算**，它**自动帮你算梯度**。

```python
import torch
```

---

## 2. 计算图（Computational Graph / DAG）

autograd 会记录**数据（Tensor）和所有执行过的运算**，构成一张**有向无环图（DAG）**：

- **叶子（leaves）** = 输入张量
- **根（roots）** = 输出张量
- **边（edges）** = 运算，边本身是 `Function` 类对象，既知道前向怎么算，也知道**反向时怎么求导**

每个运算产生的新 Tensor 都会挂一个 `grad_fn`，指向**生成它的那个反向函数**：

```python
x = torch.ones(5)              # 输入
y = torch.zeros(3)             # 期望输出
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```

查看梯度函数：

```python
print(f"z 的梯度函数 = {z.grad_fn}")
print(f"loss 的梯度函数 = {loss.grad_fn}")
```

实测输出：

```text
z 的梯度函数 = <AddBackward0 object at ...>
loss 的梯度函数 = <BinaryCrossEntropyWithLogitsBackward0 object at ...>
```

- `z.grad_fn` 是 **AddBackward**，因为生成 `z` 的**最后一步是加法**（`matmul(...) + b`）。
- `loss.grad_fn` 是 **BinaryCrossEntropyWithLogitsBackward**，PyTorch 知道反向经过这里该用什么求导公式。

> 💡 **两个关键特性：**
> 1. **DAG 是动态的**：每次 `.backward()` 之后计算图会被**重建**。所以 PyTorch 里控制流（if / for）随便写，每次迭代形状和运算可以不同。
> 2. **叶子才有 `.grad`**：只有 `requires_grad=True` 的**叶子节点**能拿到 `.grad`，中间节点没有。

---

## 3. requires_grad 与 grad_fn

- 创建时指定：`torch.randn(5, 3, requires_grad=True)`；也可以在之后用 `x.requires_grad_(True)` 开启。
- **只有 `requires_grad=True` 的叶子节点**，`backward()` 后才有 `.grad` 属性。
- 不写 `requires_grad=True` 时默认为 `False`，PyTorch 不会把它当作要求梯度的参数。

> ⚠️ **我最初的一个误区（GPT 纠正）：** `requires_grad=True` 是**关键字参数（keyword argument）**，不是 `torch.randn(5, 3, requires_grad=True)` 里的"第三个位置参数"。你没法用 `torch.randn(5, 3, True)` 这种方式传它。

**一句话记忆：**
> `requires_grad=True` = 这个变量以后要参与训练，我需要知道 loss 对它的梯度。

---

## 4. 官方示例逐行解读（BCE 二分类例子）

```python
import torch

x = torch.ones(5)                              # 输入数据，x.shape = (5,)
y = torch.zeros(3)                             # 期望输出（标签），y.shape = (3,)
w = torch.randn(5, 3, requires_grad=True)      # 要学习的参数 w：5×3 随机矩阵
b = torch.randn(3, requires_grad=True)         # 要学习的参数 b：3 个随机数
z = torch.matmul(x, w) + b                     # 前向：z = x@w + b（logits）
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)  # 二分类损失

print(f"z 的梯度函数 = {z.grad_fn}")
print(f"loss 的梯度函数 = {loss.grad_fn}")

loss.backward()                                # 反向传播
print(w.grad)                                  # ∂loss/∂w，形状 (5, 3)
print(b.grad)                                  # ∂loss/∂b，形状 (3,)
```

**整体流程（前向）：**

```text
x ──┐
    ├─> x @ w + b ─> z ─> BCEWithLogits ─> loss（标量）
w ──┘              ↑
                   b
```

**`loss.backward()` 之后（反向）：**

```text
loss ──反向传播──> z ──> w.grad
                         b.grad
```

**逐段拆解：**

| 步骤 | 代码 | 形状变化 | 含义 |
|------|------|----------|------|
| 1. 输入 | `x = torch.ones(5)` | `(5,)` | 模型输入数据 |
| 2. 标签 | `y = torch.zeros(3)` | `(3,)` | 正确答案 / 目标值 |
| 3. 参数 | `w = randn(5,3, requires_grad=True)` | `(5,3)` | 要学习的权重 |
| 4. 参数 | `b = randn(3, requires_grad=True)` | `(3,)` | 要学习的偏置 |
| 5. 前向 | `z = torch.matmul(x, w) + b` | `(5,)×(5,3)→(3,)+ (3,)→(3,)` | 原始预测 logits |
| 6. 损失 | `BCEWithLogits(z, y)` | 标量 | 预测有多差 |
| 7. 建图 | —— | —— | 因为 w、b 要梯度，前面自动记录了计算图 |
| 8. 反向 | `loss.backward()` | —— | 倒着走，用链式法则求 ∂loss/∂w、∂loss/∂b |
| 9. 取梯度 | `w.grad` / `b.grad` | `(5,3)` / `(3,)` | **每个元素 = 对应 w/b 元素改变一点会让 loss 怎么变** |

**`loss` 内部的处理链**（BCEWithLogits 内部做了什么）：

```text
z → Sigmoid → 预测概率 → 与 y 比较（Binary Cross Entropy）→ loss（标量）
```

**关于 `grad_fn` 的直观理解：**
- `z.grad_fn = <AddBackward0>`：生成 z 的最后一步是 `+b`，所以反向经过这里用**加法的求导规则**。
- `loss.grad_fn = <BinaryCrossEntropyWithLogitsBackward0>`：loss 是 BCE 算出来的，PyTorch 知道该用什么求导公式。

**链式法则（反向传播的本质）：**

```
∂loss/∂w = ∂loss/∂z · ∂z/∂w
```

PyTorch 自动完成这一串，这就是 **Automatic Differentiation**。

> ⚠️ **这段代码还没"训练"**：它只算出了 w.grad 和 b.grad（"应该怎么改"的信息），**并没有真正修改 w、b**。修改参数是后面 **optimizer（优化器）** 要做的事。这是最容易误解的地方：**算梯度 ≠ 更新参数**。

---

## 5. 计算梯度

```python
loss.backward()      # 对计算图根节点调用 backward
print(w.grad)
print(b.grad)
```

实测输出（值随机，重点看形状）：

```text
w.grad shape = (5, 3)   # 和 w 同形状
b.grad shape = (3,)     # 和 b 同形状
```

> ⚠️ **一个图只能 backward 一次**：出于性能考虑，同一张计算图上 `backward()` 只能调用一次。如果需要对同一张图多次反向传播，要传 `retain_graph=True`（见第 7 节）。

---

## 6. 禁用梯度追踪

某些情况下不需要梯度：

```python
z = torch.matmul(x, w) + b
print(z.requires_grad)      # True

with torch.no_grad():       # 进入 no_grad 上下文
    z = torch.matmul(x, w) + b
print(z.requires_grad)      # False
```

等价写法：`detach()`

```python
z_det = z.detach()          # 分离出一个不追踪梯度的副本
print(z_det.requires_grad)  # False
```

**为什么需要禁用：**
1. **冻结参数**：把网络里某些参数标记为不参与训练（如迁移学习中锁住骨干网络）。
2. **加速**：只做前向推理（不反传）时，不追踪梯度的张量计算更高效。

> 官方教程没展开 `torch.enable_grad()`，但它是 `no_grad()` 的反操作，作用就是"在这个上下文里重新开启梯度追踪"。

---

## 7. 进阶：非标量输出与 Jacobian 乘积 ⭐

前面 `loss.backward()` 不传参就能用，是因为 **loss 是标量**（一个数）。当输出是**向量/矩阵**时，需要告诉 PyTorch 每个输出以多大的权重参与反向传播。

### 7.1 示例代码

```python
inp = torch.eye(4, 5, requires_grad=True)          # 4×5，对角线为 1
out = (inp + 1).pow(2).t()                         # 先 +1，再平方，再转置 → out 是 5×4
out.backward(torch.ones_like(out), retain_graph=True)
print(inp.grad)
```

实测输出：

```text
First call
[[4, 2, 2, 2, 2],
 [2, 4, 2, 2, 2],
 [2, 2, 4, 2, 2],
 [2, 2, 2, 4, 2]]
```

### 7.2 为什么这里不能直接 `out.backward()`

因为 `out` 不是标量（5×4 = 20 个数），PyTorch 需要知道**这 20 个输出该怎么组合起来反传**。`backward(v)` 真正算的是 **vᵀ·J**（v 转置 乘 Jacobian 矩阵），而不是 v 直接乘 out。

**逐步简化（GPT 给的实用理解）：**

```python
out.backward(torch.ones_like(out))   # ≈ out.sum().backward()
```

即：`torch.ones_like(out)` 本质是把 out 里 20 个数**全部加起来成一个标量 L**，再对 inp 求梯度。

**完整数学背景（可选理解）：**
- inp 是 4×5 = 20 个输入，out 是 5×4 = 20 个输出，展平后都是 20 维向量。
- 完整 Jacobian 是 **20×20**：20 个输出 × 每个输出对 20 个输入求偏导。
- `torch.ones_like(out)` 看成 20 维的 v：**vᵀ（1×20）× J（20×20）= 1×20**。
- 得到 20 个梯度，再按 inp 原形状排回 **4×5**，就是 `inp.grad`。

### 7.3 为什么梯度是 4×5 而不是别的形状

`inp.grad` 的形状**必须和 inp 一致**，因为要一一对应地告诉每个 `inp[i,j]` 对 loss 的影响：

```text
inp[0,0] ──→ grad[0,0]
inp[0,1] ──→ grad[0,1]
...
```

这个例子中核心运算是 `(inp+1)²`，导数是 `2(inp+1)`：inp=1 处梯度 = 4，inp=0 处梯度 = 2。

### 7.4 ⭐ 梯度会累加（PyTorch 最重要的特性之一）

```python
out.backward(torch.ones_like(out), retain_graph=True)   # First call
out.backward(torch.ones_like(out), retain_graph=True)   # Second call
print(inp.grad)
```

实测输出对比：

```text
First call:   对角 4，其余 2
Second call:  对角 8，其余 4   ← 变两倍！
```

原因：**`backward()` 的梯度默认是"累加"到 `.grad` 上，不是覆盖。**

```python
inp.grad.zero_()   # 手动清零
out.backward(torch.ones_like(out), retain_graph=True)   # 又回到对角 4、其余 2
```

`retain_graph=True` 的作用：默认反向传播完会把计算图**释放掉**，这段代码连续反传 3 次，所以要保留图。

> 💡 **这直接解释了训练里为什么每轮都要 `optimizer.zero_grad()`**：每轮训练前要把上一轮残留的梯度清零，否则梯度会不断累加、参数更新会错乱。

---

## 8. 我的疑问 × GPT 解答（本次学习重点记录）🔑

> 以下是本次学习中我和 ChatGPT 的问答整理。问题是我提的，答案是 GPT 给的（原意整理），最后附我的理解。

### ❓Q1：`w = torch.randn(5, 3, requires_grad=True)` 是什么？

**GPT 答：**
- `5` = 5 行，`3` = 3 列，生成一个 5×3 的随机 Tensor。
- `requires_grad=True` = 告诉 PyTorch：**后面需要计算这个张量的梯度**。
- 有它之后，`loss.backward()` 才会计算并保存 `w.grad`（∂loss/∂w）。
- 不写它默认是 `requires_grad=False`，PyTorch 不会把 w 当要求梯度的参数。
- **纠正**：`requires_grad=True` 是**关键字参数**，不是"第三个位置参数"，不能写成 `torch.randn(5, 3, True)`。

**我的理解：** `requires_grad=True` 就像一个"报名参加训练"的开关——开了它，PyTorch 才会在反传时为这个变量算并存梯度。

### ❓Q2：把整个 BCE 示例代码完整解释一遍

**GPT 答（压缩版流程）：**

```text
① 创建数据 x、y
② 创建要学习的参数 w、b（requires_grad=True）
③ 前向传播 z = x@w + b
④ 计算误差 loss = BCE(z, y)
⑤ PyTorch 在前面过程中自动建立计算图
⑥ 反向传播 loss.backward()
⑦ 得到梯度 w.grad、b.grad
```

**GPT 强调的两点：**
1. `z.grad_fn = AddBackward0`（生成 z 的最后一步是加法），`loss.grad_fn = BinaryCrossEntropyWithLogitsBackward0`（PyTorch 知道反传该用什么求导公式）。
2. **这段代码只算出了"应该怎么修改 w、b"的信息（w.grad、b.grad），还没真正修改它们** —— 这正是后面 `optimizer` 要接上的部分。

### ❓Q3：为什么这里不能直接 `out.backward()`？vᵀ 乘出来变成 5×5 矩阵，跟算梯度有什么关系？

**我的困惑：** 我以为 `backward(v)` 是拿 vᵀ 去乘 out，得到 5×5，不知道这和梯度有什么关系。

**GPT 纠正（关键）：**
- `backward(v)` 不是拿 vᵀ 乘 out，而是拿 **vᵀ 乘 Jacobian 矩阵 J**。
- 一个输入 x → 一个输出 y：直接 `y.backward()`。
- 多个输入多个输出：需要完整 Jacobian J，`backward(v)` 算的是 **vᵀ·J**。
- **实用记法**：`out.backward(torch.ones_like(out)) ≈ out.sum().backward()` —— 先把 out 全加起来成一个标量，再对这个标量求梯度。
- 为什么最后 `inp.grad` 还是 4×5：因为 `inp.shape` 是 4×5，你需要知道 **inp 每个元素**对结果的影响，所以梯度也必须是 4×5，一一对应。
- 完整展开：inp 20 个输入、out 20 个输出 → J 是 20×20 → vᵀ(1×20)·J(20×20) = 1×20 → 按 inp 形状排回 4×5。

**我的理解：** `backward(v)` 的 v 不是"输出"，而是"给每个输出配的权重"。给全 1 就等价于"把输出全加起来再求导"。

### ❓Q4：L 是我 out 矩阵的和，L 是标量，为什么它对 inp 每个元素的导数有那么多（20）个？

**我的困惑：** L 明明只是一个数，为什么 ∂L/∂inp 有 20 个数？

**GPT 答（关键概念）：**
- **inp 不是一个变量，而是 20 个独立变量**（4×5 = 20 个元素）。
- 缩小到 2×2 就清楚了：设 `inp = [a b; c d]`，则 `L = (a+1)² + (b+1)² + (c+1)² + (d+1)²`。
- L 确实是一个数，但它**同时依赖 a、b、c、d 四个变量**，所以有 4 个导数：
  - ∂L/∂a = 2(a+1)，∂L/∂b = 2(b+1)，∂L/∂c = 2(c+1)，∂L/∂d = 2(d+1)
- 因为 `(b+1)² + (c+1)² + (d+1)²` 和 a 无关，对 a 求导是 0，所以 ∂L/∂a 只来自 `(a+1)²` 这一项。
- **一个标量 L，可以对很多变量求导。** 这些导数组合起来，按 inp 的形状排列，就是梯度。
- 你之前学 `w.grad` 也是同一道理：loss 是标量，但 w 是 5×3 矩阵，所以 loss 对 w 的梯度也是 5×3 矩阵。

**我的理解：** 梯度个数 = 变量个数，不是输出个数。标量 loss 依赖多少个参数，就有多少个梯度，形状跟参数保持一致。

---

## 9. 本节小结

**核心要点（背下来）：**

1. **autograd** = PyTorch 内置自动微分引擎，靠计算图（DAG）支撑反向传播。
2. **计算图**：叶子=输入、根=输出、边=运算（Function）；每个非叶子 Tensor 有 `grad_fn`。
3. **`requires_grad=True` 是关键字参数**，不是位置参数；只有叶子节点有 `.grad`。
4. **前向建图，反向求梯度**：`loss.backward()` → 链式法则 → `w.grad`、`b.grad`。
5. **`grad_fn` 看最后一步运算**：`z = matmul + b` 最后一步是加法，所以 `AddBackward0`。
6. **算梯度 ≠ 更新参数**：backward 只算"怎么改"，改参数是 optimizer 的事。
7. **禁用追踪**：`no_grad()` / `detach()`，用于冻结参数、加速前向。
8. **非标量输出**：`backward(v)` 算 vᵀ·J；`out.backward(ones_like(out)) ≈ out.sum().backward()`。
9. **梯度形状 = 输入形状**：标量 L 对每个变量都有一个偏导，拼起来按输入形状排。
10. ⭐ **梯度默认累加**：所以每轮训练前要 `optimizer.zero_grad()`。

**和后面课程的联系：**
- **Optimizer**：SGD/Adam 更新参数时用的就是 `.grad`；每轮 `zero_grad()` 清空累加梯度。
- **PPO / 策略梯度**：actor-critic 网络的反向传播全部依赖 autograd；策略梯度的核心也是算 loss 对参数的梯度。
- **调试**：看 `w.grad` 是否 NaN / 全 0 / 爆炸，是排查训练不收敛的第一抓手。

---

## 10. 自测小练习

```python
import torch
torch.manual_seed(0)

# 1. 复现官方 BCE 示例，确认 grad_fn 和梯度形状
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print("z.grad_fn:", z.grad_fn)          # AddBackward0
print("loss.grad_fn:", loss.grad_fn)    # BinaryCrossEntropyWithLogitsBackward0
loss.backward()
print("w.grad shape:", w.grad.shape)    # (5, 3)
print("b.grad shape:", b.grad.shape)    # (3,)

# 2. 验证梯度累加，以及 zero_grad 的作用
inp = torch.eye(2, requires_grad=True)   # 2×2 单位阵，方便看
out = (inp + 1).pow(2).sum()
out.backward(retain_graph=True)
print("第一次:", inp.grad)               # [[4,2],[2,4]]
out.backward(retain_graph=True)
print("第二次(累加):", inp.grad)         # [[8,4],[4,8]]
inp.grad.zero_()
out.backward()
print("清零后:", inp.grad)               # [[4,2],[2,4]]

# 3. 用 no_grad 关闭追踪，对比 requires_grad
x = torch.ones(2, requires_grad=True)
print("x.requires_grad:", x.requires_grad)   # True
with torch.no_grad():
    y = x * 2
print("no_grad 后 y.requires_grad:", y.requires_grad)  # False
print("detach:", (x * 2).detach().requires_grad)       # False

# 4. 手动实现一次完整的"算梯度 + 更新参数"（预热 optimizer）
w = torch.tensor([[0.5], [0.5]], requires_grad=True)   # (2,1)
x = torch.tensor([[1.0, 2.0]])                        # (1,2)
target = torch.tensor([[5.0]])
loss = (x @ w - target).pow(2)                        # 均方误差，x@w=1.5，2×(1.5-5)=-7
loss.backward()
print("梯度:", w.grad)                                 # [[-7],[-14]]（沿 x 权重 1 和 2 分摊）
w.data -= 0.01 * w.grad                                # 手动 SGD 一步：w ← w - lr * grad
print("更新后 w:", w.detach())
```
