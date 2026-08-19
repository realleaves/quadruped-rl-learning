# PPO（Proximal Policy Optimization）学习笔记

> 📖 来源：[OpenAI Spinning Up — Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
> 🗓️ 学习日期：2026-08-19
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → **PPO** → Isaac Lab → HIMLoco → 四足机器人部署

---

## 1. 什么是 PPO（为什么要学它）

PPO（Proximal Policy Optimization，近端策略优化）是一种 **Policy Gradient（策略梯度）强化学习算法**。

PPO 的核心思想可以概括成一句话：

> **好的动作以后多做，坏的动作以后少做，但每次更新策略不能变化太大。**

PPO 属于 **On-Policy（同策略）算法**：当前策略 $\pi_{\theta_k}$ 与环境交互得到数据，然后用这批数据更新当前策略；更新得到新策略之后，再重新采集新数据。Spinning Up 介绍了 PPO-Penalty 和 PPO-Clip 两种形式，其中实现的重点是 PPO-Clip。

在四足机器人中可以理解为：

```text
机器人当前状态 Observation
        ↓
      Actor
        ↓
产生关节动作 Action
        ↓
机器人执行动作
        ↓
环境计算 Reward
        ↓
判断这个动作比预期好还是差
        ↓
      PPO 更新
```

PPO 是后面学习 Isaac Lab、RSL-RL、HIMLoco 四足机器人强化学习时需要重点掌握的算法基础。

---

## 2. PPO 中的 Actor 和 Critic

PPO 通常使用 Actor-Critic 结构：

```text
           Observation s
            /        \
           ↓          ↓
        Actor       Critic
          ↓           ↓
       Action a     Value V(s)
```

### 2.1 Actor（策略网络）

Actor 表示：

$$\pi_\theta(a|s)$$

其中：

* $s$：当前状态 / Observation
* $a$：动作 / Action
* $\theta$：Actor 神经网络参数
* $\pi_\theta(a|s)$：状态 $s$ 下采取动作 $a$ 的概率或概率密度

Actor 负责：

$$\boxed{\text{现在应该做什么动作？}}$$

四足机器人里，Actor 最终可能输出：

```text
Observation
↓
Actor
↓
12 个关节动作
```

---

### 2.2 Critic（价值网络）

Critic 表示：

$$V_\phi(s)$$

其中：

* $s$：机器人当前状态
* $\phi$：Critic 网络参数
* $V_\phi(s)$：对当前状态未来价值的预测

Critic 负责：

$$\boxed{\text{当前这个状态有多好？}}$$

例如：

$$V(s_t)=20$$

可以理解为 Critic 判断：

> 从当前状态继续运行，未来预计能够获得大约 20 的累计回报。

---

### 2.3 Actor 与 Critic 的区别

| 网络        | 数学表示                 | 作用                |
| --------- | --------------------- | ----------------- |
| Actor     | $\pi_\theta(a \mid s)$ | 决定做什么          |
| Critic    | $V_\phi(s)$           | 判断当前状态有多好    |
| Actor 参数  | $\theta$              | PPO 更新的策略参数   |
| Critic 参数 | $\phi$                | Value Function 参数 |

记住：

$$\boxed{\text{Actor 负责行动，Critic 负责评价}}$$

---

## 3. Reward、Return、Value、Advantage

这四个概念一定要区分。

### 3.1 Reward：当前一步奖励

$$r_t$$

表示：

> 当前时间 $t$ 做完动作以后，环境立即给出的奖励。

例如四足机器人：

```text
跟踪目标速度       +2.0
保持身体稳定       +1.0
动作变化太大       -0.2
脚部滑动           -0.1
```

最终可能：

$$r_t=2.7$$

Reward 只描述**当前一步**。

---

### 3.2 Return：未来累计回报

Reward-to-go：

$$\hat R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

表示：

> 从当前时间 $t$ 开始，未来累计能够获得多少奖励。

其中 $\gamma$ 是 Discount Factor（折扣因子）。

例如：

```python
gamma = 0.99
```

表示未来奖励仍然比较重要。

---

### 3.3 Value：Critic 的预测

$$V(s_t)$$

是 Critic 对 Return 的预测。

例如：

```text
Critic 预测：
V(sₜ) = 15

实际运行以后：
Rₜ = 20
```

说明实际结果比 Critic 原本预计得好。

---

### 3.4 Advantage：优势

Advantage 表示：

> 当前动作相比"正常情况下的预期表现"到底好多少。

最简单可以理解：

$$A_t \approx R_t - V(s_t)$$

例如：

$$R_t = 20,\quad V(s_t) = 15$$

那么：

$$A_t = 5$$

所以 $A_t > 0$ 说明这个动作比预期好。

PPO 应该：

$$\pi(a_t|s_t) \uparrow$$

以后增加这个动作出现的可能性。

反过来，$A_t < 0$ 说明动作比预期差，应降低它的可能性。

因此最重要的是：

$$\boxed{A > 0 \Rightarrow \text{好动作}}$$

$$\boxed{A < 0 \Rightarrow \text{坏动作}}$$

---

## 4. PPO 的概率比 Ratio

PPO 中非常重要的一个量：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

它表示：

> **新策略相比旧策略，对当前这个动作的偏好发生了多少变化。**

例如：

旧策略：$\pi_{old}(a|s) = 0.2$

新策略：$\pi_{new}(a|s) = 0.24$

那么：

$$r = \frac{0.24}{0.2} = 1.2$$

含义：

| Ratio | 含义          |
| ----- | ----------- |
| $r=1$ | 新旧策略基本没变化   |
| $r>1$ | 新策略更倾向这个动作  |
| $r<1$ | 新策略更不倾向这个动作 |

> ⚠️ 注意：这里的 $r_t(\theta)$ 是 **Probability Ratio（概率比）**，不是前面的 Reward $r_t$。两个地方经常使用相同字母，一定根据上下文判断。

---

## 5. PPO-Clip 核心公式

PPO-Clip 的目标函数：

$$L^{CLIP} = \min\left(r_t(\theta)A_t,\ \text{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,A_t\right)$$

其中：

* $r_t(\theta)$：新旧策略概率比
* $A_t$：Advantage
* $\epsilon$：Clip Ratio
* `clip()`：限制数值范围

PPO-Clip 的作用是移除策略发生过大改变时继续获得优化收益的动力，从而使新策略不会因为一次优化而过度远离旧策略。

---

## 6. Clip 到底在干什么

例如：

```python
clip_ratio = 0.2
```

所以 $\epsilon = 0.2$，限制范围是 $[1-\epsilon,\ 1+\epsilon] = [0.8,\ 1.2]$，即 $\text{clip}(r,\ 0.8,\ 1.2)$。

例如：

```text
r = 1.1 → clip = 1.1
r = 1.5 → clip = 1.2
r = 0.5 → clip = 0.8
```

---

### 6.1 Advantage > 0：好动作

假设 $A = 2$。

如果 $r = 1$，目标：

$$L = 2$$

如果提高动作概率使 $r = 1.1$，那么：

$$L = 2.2$$

说明 PPO 鼓励：

> 好动作以后多做。

但是 $r = 1.5$ 时，经过 Clip 后最多按照 $1.2$ 计算：

$$L = 1.2 \times 2 = 2.4$$

即使继续把 $r$ 提高到 $2,\,5,\,10$，目标也不会因为这种过度变化继续变好。

---

### 6.2 Advantage < 0：坏动作

假设 $A = -2$。

如果 $r = 1$，那么：

$$L = -2$$

降低这个动作概率使 $r = 0.9$，得到：

$$L = -1.8$$

因为 PPO 最大化目标函数，$-1.8 > -2$，所以这种改变是好的。

但是如果 $r = 0.5$，PPO 不希望策略一下改变过大，Clip 会限制收益。

---

### 6.3 一句话理解 PPO-Clip

$$\boxed{\text{好动作多做，坏动作少做，但一次不要改太猛}}$$

这就是 PPO 最核心的思想。

---

## 7. GAE：Advantage 怎么算得更稳定

PPO 需要 Advantage，但直接：

$$A_t = R_t - V(s_t)$$

可能因为完整轨迹中的随机性产生较大波动。

因此常使用 **GAE（Generalized Advantage Estimation）**。

---

### 7.1 TD Error

首先计算：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

其中：

* $r_t$：当前 reward
* $V(s_t)$：当前状态价值
* $V(s_{t+1})$：下一状态价值

可以理解：

$$\delta_t = \underbrace{r_t + \gamma V(s_{t+1})}_{\text{执行动作后的新评价}} - \underbrace{V(s_t)}_{\text{执行动作前的预期}}$$

---

### 7.2 GAE 公式

$$\hat A_t = \delta_t + \gamma\lambda\,\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

也可以递归写：

$$\boxed{A_t = \delta_t + \gamma\lambda A_{t+1}}$$

其中 $\lambda$ 控制 Advantage 估计参考多远的未来。

---

### 7.3 $\lambda$ 的含义

如果 $\lambda = 0$，那么 $A_t = \delta_t$，只看一步，更依赖 Critic。

如果 $\lambda \rightarrow 1$，会考虑更多未来的信息。

简单记：

```text
λ 小
↓
更依赖 Critic
↓
通常 variance 小、bias 大

λ 大
↓
更多参考未来轨迹
↓
通常 variance 大、bias 小
```

Spinning Up 的 PPO 默认：

```python
lam = 0.97
```

---

### 7.4 `gamma` 和 `lambda` 不要混

```text
gamma
↓
未来 Reward 有多重要

lambda
↓
计算 Advantage 时参考多远
```

即：

$$\boxed{\gamma = \text{未来奖励折扣}}$$

$$\boxed{\lambda = \text{GAE 的时间范围 / bias-variance 折中}}$$

---

## 8. Exploration vs. Exploitation

PPO 训练的是 **Stochastic Policy（随机策略）**，并以 On-Policy 方式根据当前策略采样动作，因此本身具有探索能力。随着训练进行，策略通常会逐渐利用已经找到的高奖励行为，因此随机程度可能降低，也可能因此陷入局部最优。

### Exploration：探索

尝试暂时不知道好不好的动作：

```text
腿抬高一点
步幅大一点
身体低一点
```

可能发现新的更优步态。

---

### Exploitation：利用

已经发现某种动作效果很好：

```text
这个步态 Reward 高
↓
提高这种动作的概率
↓
以后更多使用
```

---

### Local Optimum：局部最优

例如：

```text
小碎步走路
Reward = 80
```

机器人发现之后不断强化。

但实际上可能还存在：

```text
更好的奔跑步态
Reward = 120
```

如果探索越来越少，就可能一直停留在 80。

这就是：

$$\boxed{\text{Local Optimum = 局部最优}}$$

---

## 9. PPO 完整训练流程

Spinning Up 给出的 PPO-Clip 伪代码可以概括成下面几个步骤：先用当前策略收集 trajectories，然后计算 rewards-to-go 和 advantage，再优化 PPO-Clip 策略目标，同时通过均方误差训练 Value Function。

```text
① 当前 Actor πθ
        ↓
② 与环境交互
        ↓
③ 收集轨迹
(sₜ, aₜ, rₜ, ...)
        ↓
④ 计算 Return R̂ₜ
        ↓
⑤ Critic 计算 V(sₜ)
        ↓
⑥ GAE 计算 Advantage Âₜ
        ↓
   ┌───────────────┐
   ↓               ↓
⑦ 更新 Actor      ⑧ 更新 Critic
PPO-Clip           MSE
θ → θnew           φ → φnew
   ↓               ↓
   └───────┬───────┘
           ↓
      用新策略重新采数据
           ↓
          重复
```

---

## 10. Critic 怎么训练

Critic 的目标：

$$V_\phi(s_t)$$

尽量接近实际 Return：

$$\hat R_t$$

因此使用均方误差：

$$L_V = \left(V_\phi(s_t) - \hat R_t\right)^2$$

例如：

```text
Critic 预测：
V(s) = 10

实际 Return：
R = 15
```

Loss：

$$(10-15)^2 = 25$$

不断训练以后：

```text
10 → 12 → 14 → 14.8 → 15
```

因此：

$$\boxed{\text{Critic 学习如何更准确地预测状态价值}}$$

而 Critic 越准确，Advantage 的估计通常也会更可靠。

---

## 11. PPO 主要参数

Spinning Up 的 PyTorch PPO 接口包含下面这些参数。

```python
ppo_pytorch(
    env_fn,
    actor_critic,
    ac_kwargs={},
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=0.0003,
    vf_lr=0.001,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    logger_kwargs={},
    save_freq=10
)
```

### 11.1 `env_fn`

创建训练环境的函数。

可以理解：

```text
env_fn
↓
创建 Environment
↓
Actor 与环境交互
```

以后在 Isaac Lab 中就是你的机器人训练环境。

---

### 11.2 `actor_critic`

创建 Actor + Critic 网络。

需要包含：

```text
Actor  → pi
Critic → v
```

并提供：

```python
step()
act()
```

---

### 11.3 `ac_kwargs`

给 Actor-Critic 网络传额外配置。

例如：

```python
ac_kwargs = {
    "hidden_sizes": [64, 64]
}
```

可以理解：

$$\boxed{\text{Actor/Critic 网络结构配置}}$$

---

### 11.4 `seed=0`

随机种子。

影响：

* 网络初始化
* 动作采样
* 环境随机性

主要用于提高实验可复现性。

---

### 11.5 `steps_per_epoch=4000`

每轮 PPO 更新之前，与环境交互多少个 step。

```text
采 4000 step
↓
计算 Return / Advantage
↓
更新 Actor / Critic
↓
重新采数据
```

所以：

$$\boxed{\text{steps_per_epoch = 每轮采多少数据}}$$

---

### 11.6 `epochs=50`

执行多少轮：

```text
采数据
+
训练
```

例如：

```text
Epoch 1 → 采4000步 → 更新
Epoch 2 → 采4000步 → 更新
...
Epoch 50
```

---

### 11.7 `gamma=0.99`

Discount Factor：

$$\boxed{\gamma = \text{未来奖励的重要程度}}$$

数值越接近 1，越重视长期表现。

---

### 11.8 `clip_ratio=0.2`

就是 PPO 中的 $\epsilon = 0.2$，对应：

$$[1-\epsilon,\ 1+\epsilon] = [0.8,\ 1.2]$$

作用：

$$\boxed{\text{不鼓励新策略一次变化过大}}$$

---

### 11.9 `pi_lr=0.0003`

Actor 的学习率：

```python
lr = 3e-4
```

控制：

> 每一次梯度更新 Actor 参数走多大一步。

---

### 11.10 `vf_lr=0.001`

Critic 的学习率：

```python
lr = 1e-3
```

所以：

```text
pi_lr → Actor 学习率
vf_lr → Critic 学习率
```

---

### 11.11 `train_pi_iters=80`

同一批数据最多对 Actor 进行多少次梯度更新。

```text
采集一批数据
↓
Actor update 1
Actor update 2
...
最多 Actor update 80
```

> ⚠️ 不是重新采 80 次数据，而是**同一批 On-Policy 数据重复优化多次**。

---

### 11.12 `train_v_iters=80`

同一批数据训练 Critic 的次数。

优化：

$$(V(s)-R)^2$$

---

### 11.13 `lam=0.97`

GAE 中的 $\lambda$，控制：

> Advantage 估计考虑多远的未来。

---

### 11.14 `max_ep_len=1000`

一个 Episode 最多运行多少 step。

例如控制频率 50 Hz：

$$1000/50 = 20\text{ s}$$

意味着最多运行约 20 秒。

如果机器人提前摔倒，则可以提前 Termination。

---

### 11.15 `target_kl=0.01`

KL Divergence 可以先理解成：

> **新策略和旧策略差了多少。**

如果 $KL \approx 0$，说明新旧策略比较接近。

如果 KL 越来越大，说明策略已经改变很多。

Spinning Up 的 PPO 使用 approximate KL 做 Early Stopping：如果策略更新偏离旧策略过多，就提前结束该轮 Policy 优化。

因此：

```text
clip_ratio
↓
第一层：不鼓励改变太大

target_kl
↓
第二层：真的变化太大就提前停止
```

---

### 11.16 `logger_kwargs`

训练日志相关配置。

例如：

* 实验名称
* 日志保存位置
* 输出信息

现在知道用途即可。

---

### 11.17 `save_freq=10`

每隔多少 Epoch 保存一次 Actor 和 Critic。

例如：

```text
Epoch 10 → Save
Epoch 20 → Save
Epoch 30 → Save
```

---

## 12. PyTorch Actor-Critic 接口

Spinning Up 的 PyTorch Actor-Critic 中，`step()` 接收一批 Observation，并返回 Action、Value Estimate 和动作的 Log Probability。

```python
a, v, logp_a = ac.step(obs)
```

---

### 12.1 `a`

Action：

```text
shape = (batch, act_dim)
```

例如：

```text
64 个环境
每个机器人 12 个动作
```

那么 `a.shape` 可能是：

```text
(64, 12)
```

---

### 12.2 `v`

Critic 的 Value：

```text
shape = (batch,)
```

例如：

```text
机器人1 → 12.4
机器人2 → 9.8
机器人3 → 16.1
...
```

每个 Observation 对应一个 Value。

---

### 12.3 `logp_a`

表示：

$$\log \pi_\theta(a|s)$$

即：

> 当前策略认为动作 $a$ 出现的 Log Probability。

PPO 代码里非常重要。

---

## 13. 为什么 PPO 使用 `log_prob`

PPO 需要计算：

$$\text{ratio} = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$$

但实际代码通常保存：

```text
logp_old
logp
```

利用：

$$\log\frac{a}{b} = \log a - \log b$$

得到：

$$\text{ratio} = \exp(\log\pi_{new} - \log\pi_{old})$$

PyTorch 中经常写：

```python
ratio = torch.exp(logp - logp_old)
```

所以看到 `log_prob` 要立刻想到：

$$\boxed{\text{后面要计算 PPO Ratio}}$$

---

## 14. PPO-Clip 对应 PyTorch 代码

核心数学公式：

$$L^{CLIP} = \min\left(\text{ratio}\times A,\ \text{clip}(\text{ratio})\times A\right)$$

代码大致是：

```python
ratio = torch.exp(logp - logp_old)

clip_adv = torch.clamp(
    ratio,
    1.0 - clip_ratio,
    1.0 + clip_ratio
) * adv

objective = torch.min(
    ratio * adv,
    clip_adv
)

loss_pi = -objective.mean()
```

这里为什么有负号？

因为数学公式希望 $\max L^{CLIP}$，而 PyTorch Optimizer 通常执行 Gradient Descent（$\min \text{Loss}$）。

所以 $\max L$ 等价于 $\min(-L)$，因此：

```python
loss_pi = -objective.mean()
```

---

## 15. Critic Loss 对应 PyTorch

Critic $V(s)$ 需要逼近 $R$，所以：

```python
loss_v = ((value - returns) ** 2).mean()
```

就是：

$$\boxed{L_V = (V(s)-R)^2}$$

然后：

```python
value_optimizer.zero_grad()
loss_v.backward()
value_optimizer.step()
```

和之前学习普通 PyTorch 神经网络训练完全一样。

---

## 16. GAE 对应代码

数学公式：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$A_t = \delta_t + \gamma\lambda A_{t+1}$$

代码思路：

```python
delta = reward + gamma * next_value - value

advantage = (
    delta
    + gamma * lam * next_advantage
)
```

实际处理 Episode Termination 时通常还需要利用 `done` / mask 防止把不同 Episode 的数据串在一起。

---

## 17. PPO 参数分类记忆

不要把所有参数当成十几个独立数字背。

### 数据采集

```text
steps_per_epoch
epochs
max_ep_len
```

控制：

$$\boxed{\text{采多少数据、训练多少轮}}$$

---

### Return / Advantage

```text
gamma
lam
```

控制：

```text
gamma → 多重视未来 Reward
lam   → GAE 看多远
```

---

### Actor 更新

```text
clip_ratio
pi_lr
train_pi_iters
target_kl
```

可以记：

```text
pi_lr
↓
每次梯度更新走多大

train_pi_iters
↓
一批数据更新多少次

clip_ratio
↓
不要鼓励策略变化太大

target_kl
↓
真的变化太大就停止
```

---

### Critic 更新

```text
vf_lr
train_v_iters
```

表示：

```text
Critic 学习率
+
Critic 每批数据训练次数
```

---

## 18. PPO 训练好的模型怎么使用

Spinning Up 的 PyTorch 文档中，训练好的 Actor-Critic 可以重新加载，然后通过 `act()` 根据 Observation 获得动作。

概念上：

```python
obs_tensor = torch.as_tensor(
    obs,
    dtype=torch.float32
)

action = ac.act(obs_tensor)
```

流程：

```text
机器人传感器数据
↓
构造成 Observation
↓
转换成 Tensor
↓
Actor
↓
Action
↓
发送给机器人控制器
```

这也是以后四足机器人 **Sim-to-Real 部署** 最核心的数据流之一。

---

## 19. PPO 与四足机器人的对应关系

以后在 Isaac Lab / HIMLoco 中，可以把 PPO 对应到：

```text
机器人状态
joint_pos
joint_vel
base_ang_vel
gravity
command
previous_action
        ↓
    Observation
        ↓
      Actor
        ↓
      Action
        ↓
关节位置 / 力矩目标
        ↓
      Robot
        ↓
Reward Function
        ↓
速度跟踪、稳定性、
能量、足端滑动……
        ↓
Return + Critic
        ↓
       GAE
        ↓
   Advantage
        ↓
    PPO-Clip
        ↓
更新 Actor / Critic
```

因此在 Isaac Lab 中学习：

```python
ObservationsCfg
ActionsCfg
RewardsCfg
TerminationsCfg
```

实际上就是在定义 PPO 与机器人交互时的：

$$s_t,\quad a_t,\quad r_t,\quad \text{done}$$

---

## 20. 本节小结

**核心要点（需要记住）：**

1. PPO = **好动作多做，坏动作少做，但策略一次不能改变太大**。
2. PPO 是 **On-Policy**：当前策略采数据，更新后重新采。
3. Actor：$\pi_\theta(a|s)$，负责选择动作。
4. Critic：$V_\phi(s)$，负责预测状态价值。
5. Reward $r_t$ = 当前一步奖励。
6. Return $R_t$ = 从当前开始未来累计奖励。
7. Advantage：$A_t \approx R_t - V(s_t)$，表示动作比预期好还是差。
8. GAE 使用多个 TD Error 更稳定地估计 Advantage：$A_t = \delta_t + \gamma\lambda A_{t+1}$。
9. PPO Ratio：$\text{ratio} = \frac{\pi_{new}}{\pi_{old}}$，表示新旧策略对动作的偏好变化。
10. PPO-Clip：$\min\left(\text{ratio}\cdot A,\ \text{clip}(\text{ratio})\cdot A\right)$，防止策略更新过猛。
11. `gamma` = 多重视未来 Reward。
12. `lam` = GAE 参考多远。
13. `clip_ratio` = PPO 的主要限制范围。
14. `target_kl` = 策略变化过大时 Early Stopping。
15. `pi_lr` = Actor 学习率，`vf_lr` = Critic 学习率。
16. `train_pi_iters` / `train_v_iters` = 同一批数据重复更新 Actor / Critic 的次数。
17. PPO 代码常使用：

    ```python
    ratio = torch.exp(logp - logp_old)
    ```

    来计算新旧策略概率比。

---

## 21. 最重要的一条知识链

以后忘了 PPO，可以先把这一条恢复出来：

```text
Observation sₜ
↓
Actor 产生 Action aₜ
↓
Environment
↓
Reward rₜ
↓
计算 Return Rₜ
↓
Critic 预测 V(sₜ)
↓
GAE
↓
Advantage Aₜ
↓
A > 0 → 增加动作概率
A < 0 → 降低动作概率
↓
PPO Clip 防止改得太猛
↓
更新 Actor
↓
MSE 更新 Critic
↓
重新采集数据
```

---

## 22. 自测小练习

### ① Probability Ratio

```python
import torch

logp_new = torch.tensor([-0.8])
logp_old = torch.tensor([-1.0])

ratio = torch.exp(logp_new - logp_old)

print(ratio)
```

思考：

```text
ratio > 1
```

意味着新策略更喜欢这个动作还是更不喜欢？

---

### ② PPO Clip

```python
ratio = torch.tensor([1.5])
adv = torch.tensor([2.0])

clip_ratio = 0.2

normal = ratio * adv

clipped = torch.clamp(
    ratio,
    1 - clip_ratio,
    1 + clip_ratio
) * adv

objective = torch.min(normal, clipped)

print("normal:", normal)
print("clipped:", clipped)
print("objective:", objective)
```

应该观察到：

```text
ratio = 1.5
```

已经超过 $1.2$，因此正 Advantage 情况下目标不会继续因为 Ratio 增大而提高。

---

### ③ Critic Loss

```python
value = torch.tensor([10.0])
returns = torch.tensor([15.0])

loss_v = ((value - returns) ** 2).mean()

print(loss_v)
```

结果：

```text
25
```

表示 Critic 的预测和实际 Return 存在较大误差。

---

### ④ TD Error

```python
reward = 2.0
value = 10.0
next_value = 11.0
gamma = 0.99

delta = reward + gamma * next_value - value

print(delta)
```

思考：

```text
delta > 0
```

说明实际情况比 Critic 原来的预期更好还是更差？

---

### ⑤ GAE 递归

```python
delta = 2.0
next_advantage = 1.0

gamma = 0.99
lam = 0.97

advantage = (
    delta
    + gamma * lam * next_advantage
)

print(advantage)
```

理解：

```text
当前 Advantage
=
当前 TD Error
+
未来 Advantage 的一部分
```

---

## 23. 当前掌握程度

学到这里，PPO 暂时不要求自己从头推导所有数学公式。

现阶段应该能够回答：

1. Actor 和 Critic 分别干什么？
2. Reward、Return、Value、Advantage 有什么区别？
3. 为什么 Advantage 正数要增加动作概率？
4. Probability Ratio 表示什么？
5. PPO 为什么需要 Clip？
6. `gamma` 和 `lambda` 有什么区别？
7. GAE 解决什么问题？
8. `log_prob` 为什么会出现在 PPO 代码中？
9. `clip_ratio` 和 `target_kl` 为什么都在限制策略更新？
10. PPO 一轮训练的数据流是什么？

如果这些问题基本能够自己说出来，就可以继续进入：

**PPO PyTorch 实现 → Isaac Lab PPO/RSL-RL 配置 → 四足机器人强化学习任务。**
