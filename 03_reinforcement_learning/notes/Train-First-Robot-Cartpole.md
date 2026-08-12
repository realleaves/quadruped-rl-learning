# Isaac Lab 第一个机器人教程学习笔记（Train Your First Robot in Isaac Lab）

> 📖 来源：[NVIDIA Docs — Train Your First Robot in Isaac Lab](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-first-robot-with-isaac-lab/index.html)
> 🗓️ 学习日期：2026-08-12
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → PPO → Isaac Lab → HIMLoco → 四足机器人部署
> 📌 本次覆盖范围：Overview(0) → Analyzing the Code(7)，**含 Running the Training**

---

## 0. 先搞清楚：这次学的是什么

### 0.1 模块学习目标（原话）

- 了解 **Physical AI**，以及它如何改变机器人学习和执行任务的方式。
- 用 Isaac Lab + Isaac Sim 描述强化学习（RL）的核心原理及其与机器人的关系。
- 在 Isaac Lab 中**识别并配置一个 RL 任务的各个基本组件**。
- 用 Isaac Lab 工作流在仿真里**训练、评估、调优**一个机器人控制策略（本模块用 cartpole）。
- **分析训练结果**。

### 0.2 三个关键定位

- **Physical AI**：RL 是其中的核心主题，教程原话称它"定义了一种动态的、可适应的训练机器人大脑的策略"。
- **Isaac Sim 和 Isaac Lab 的分工**（后面第 2 节细讲）：
  - **Isaac Sim** = 物理仿真 + 渲染（提供环境和机器人）。
  - **Isaac Lab** = 强化学习框架（训练机器人"大脑"）。
- 一句话总流程：**在 Isaac Sim 里搭机器人 → 在 Isaac Lab 里训练 → 回到 Isaac Sim 里测试**。

### 0.3 学完的产出

一个能**平衡倒立摆**的策略，训练时同时在仿真里跑 **4096 个 cartpole**（"a fleet of cartpoles balancing in simulation"）。后续模块会延伸到更复杂机器人和 **sim-to-real（仿真到真机迁移）**。

> 💡 为什么要先在仿真里训练？"Physical AI is born in simulation"——训练早期产生的行为不可预测，在真机上练既危险又烧钱；仿真可以控制重力、摩擦力，还能**比真实时间快很多**地运行。

---

## 1. 强化学习入门（RL for Robots）

### 1.1 核心定义

RL 是"**从交互中学习**"（learning from interaction）。教程强调：我们**定义目标，而不是实现目标的显式步骤**——机器人不是被程序直接"写死"怎么平衡杆子，而是靠与环境交互自己学会。

### 1.2 Agent–Environment 交互循环

这是整个 RL 的地基（对应 MDP，第 3 节展开）：

```
          环境 state（当前状态）
                ↓
   agent 收到 observations（观测，不完美的传感器信息）
                ↓
   policy 策略把"看到什么"映射成"下一步做什么"
                ↓
   agent 输出 actions（动作，改变环境）
                ↓
   环境返回 rewards（奖励，标量，评价这步的好坏）
                ↓
   循环…… 策略不断进化，目标：最大化累积奖励
```

### 1.3 关键概念

| 概念 | 含义 |
|------|------|
| **State（状态）** | 环境在某一时刻的**完整**描述 |
| **Observation（观测）** | agent 传感器**实际感知到**的（不完整、有噪声） |
| **Policy（策略）** | "看到什么 → 该做什么"的映射。训练完就是一个**存到文件的神经网络**。**"deep" in deep RL 就指"RL + 深度神经网络"** |
| **Action（动作）** | agent 做出的运动/行为 |
| **Reward（奖励）** | 标量反馈，告诉 agent 这一步"好"还是"坏" |

> ⚠️ 分清 State 和 Observation：agent 拿不到完美的状态，只能拿到观测。仿真里能看到真实状态（叫 **privileged data**，见 3.3 节），真机上只有传感器读数。

### 1.4 三种奖励类型（重点）

| 类型 | 定义 | 例子 |
|------|------|------|
| **Sparse（稀疏）** | 反馈稀少但有决定性 | 下棋赢的那一下才算数 |
| **Dense（稠密）** | 反馈频繁、渐进 | 机器人每靠近目标一点就给一点 |
| **Shaped（塑形）** | 额外注入领域知识的引导奖励 | 下棋时吃到对方棋子就给奖励 |

> 💡 塑形奖励是把**领域知识**塞进奖励的抓手：光有稀疏奖励，学习太慢；光有稠密奖励，可能被"钻空子"。第 1.5 节就是活例子。

### 1.5 奖励设计的例子：教机器人走路（本节精华）

**反面教材**：如果只用"越接近目标奖励越大"的稠密奖励，机器人可能学会**往前跳然后摔过去**，而不是走路。

**更稳健的奖励集**（每项都在约束目标的一个方面）：
- 躯干基本保持竖直；
- 脚在时间阈值内着地（防止滑行）；
- 身体朝向目标方向前进；
- 一个躯干离地高度传感器；
- 躯干压得太低要惩罚。

> 💡 这就是"**定义目标的约束，而不是定义步骤**"：每一句奖励都描述"什么样算走得好"，而不是"腿该怎么抬"。

### 1.6 为什么用 RL / 什么时候用

传统编程"很难泛化、重新编程成本高"（仓库机器人面对不同尺寸/摩擦的物体、自动驾驶的不可预测路况、人形机器人的手/走路/跑跳都很难硬编码）。

RL 合适的情形：
- 结果**部分随机、部分可控**；
- 需要**探索**和适应不确定环境；
- 动力学复杂/部分可观测，传统控制方法失效；
- 有**高保真仿真**可以先训策略。

### 1.7 为什么在仿真里训练（再次强调，并给出数字）

- 需要大量相同机器人、完美的回合重置、真机碰撞代价高；
- 仿真里可自由控制重力/摩擦等物理参数，**能比真实时间快很多**运行：例子 `Isaac-Velocity-Flat-Spot-v0` 任务在 RTX A6000 + RSL-RL 库下约 **90,000 FPS**；
- 可以安全练习危险场景（搬运热物、自动驾驶边缘情况）；
- 训练时改变物理参数（如摩擦）→ 学到更**鲁棒**的策略，利于 sim-to-real。

教程给的 Isaac Lab 真实案例：Shadow Hand 灵巧操作（手内重定向）、ANYmal-D 四足机器人在崎岖地形跟踪速度指令、齿轮装配（接触丰富的操作任务）。

---

## 2. Isaac Lab 如何加速 RL（How Isaac Lab Accelerates RL）

### 2.1 核心思路：大规模并行（重点）

传统上逐个环境串行训练很慢。Isaac Lab 的思路是：**在同一个仿真里同时跑很多个环境**——"一支机器人军队在共享世界里走路/做任务"。每个机器人是共享舞台上的一个**隔离子场景**（相互不影响）。

- 训练被**并行到 GPU** 上，而不是一步步顺序跑。
- 教程称这让 RL"在消费级硬件上也能跑很多任务"（对你的 4060 是好消息）。

### 2.2 技术栈四层（记下来）

```
OpenUSD  （定义/交互 3D 资产）
   ↓
Omniverse（仿真工具底座）
   ↓
Isaac Sim（机器人专用工具：资产、物理仿真、渲染）
   ↓
Isaac Lab（RL 工具：训练机器人大脑）
```

### 2.3 Isaac Sim vs Isaac Lab 分工

| | Isaac Sim | Isaac Lab |
|---|---|---|
| 管什么 | 资产、物理仿真、渲染 | 强化学习训练流程 |
| 有没有 UI | 有视口（Viewport） | 没有自己的 UI，纯 Python 文件 |
| 输出 | 物理状态 + 视觉输出 | 策略文件 |

Isaac Lab 处理数据的三段流水线：**states（仿真原始数据）→ noise addition（加噪声/随机化，为 sim-to-real 准备）→ observations（处理成可用观测）**。

Isaac Lab 驱动 Isaac Sim 的具体例子：按配置实例化 N 个环境（舞台上的微场景）、每回合给机器人一个随机物体抓取、训练中改变摩擦、按事件停仿真、变速运行、改重力。

### 2.4 其它加速/实用手段

- **Headless（无头训练）**：跳过 Isaac Sim 视口，把算力全部给训练。加 `--headless` 参数：
  ```bash
  python scripts/skrl/train.py --task MyRobotTask-v0 --num_envs 4096 --headless
  ```
  去掉 `--headless` 就显示视口；加 `--livestream 2` 可网络串流画面。
- **模块化、开源**：RL 库、机器人/环境资产、你自己的集成都可以替换，不用重复造轮子。
- **领域随机化（Domain Randomization）**：上面说的 noise addition，通过加随机化/扰动提升 sim-to-real 迁移。

### 2.5 预集成的 RL 库

Isaac Lab 内置了：**RSL-RL、RL-Games、SKRL、Stable Baselines**（还可以接自己的）。本教程的项目用 **SKRL + PPO**。

### 2.6 训完之后

策略稳定后可以**导出文件**，拿回 Isaac Sim 里进一步测试验证，或准备 sim-to-real。教程提到 TensorBoard 日志（学习率、policy loss）配合训练录像（1300% 加速）观察训练过程。

---

## 3. 任务设计与 MDP（Task Design and the MDP）

### 3.1 MDP 是什么

MDP（马尔可夫决策过程）"用于**决策建模**：agent 通过与环境交互来学习做决策"，环境**部分受 agent 控制**。核心循环：
- agent 在环境里**行动** → 环境用**状态**描述 → 每步后给**奖励** → 目标是训练一个**最大化奖励的策略**。

### 3.2 概念 → Isaac Lab 代码的映射（这一节的关键）

| MDP 概念 | Isaac Lab 里的对应 |
|----------|-------------------|
| **Agent** | 机器人（一个已有的 USD 资产） |
| **Environment** | 环境（USD 资产，或程序化生成如随机地形） |
| **Action** | 动关节、闭夹爪等 |
| **Observation** | 角度、速度、场景组件——"对完整状态的部分观测" |

每步仿真中，任务还需要三个函数：**怎么算 reward（奖励）、怎么算 done（结束条件）、怎么 reset（重置）**。

> 💡 换句话说，Isaac Lab 里"设计一个任务"= 选一个 agent 资产 + 一个环境 + 写 reward/done/reset 三个函数。第 7 节看代码就是看这三个函数。

### 3.3 真机 vs 仿真的数据（privileged data）

- **真机**：观测来自传感器（位置编码器、距离传感器、IMU、力传感器、相机等）。
- **仿真**：能拿到 **privileged data（特权数据）**——真机上很难甚至**不可能**测到的量。用途：拿它当 ground truth 评估训练效果。（真机部署时不能依赖它，这是 sim-to-real 的一个难点。）

### 3.4 补充：形式化定义（教程没写，但 RL 必背）

MDP 是一个五元组 $(S, A, T, R, \gamma)$：
- $S$：状态集合；$A$：动作集合；
- $T(s'|s,a)$：转移概率；
- $R(s,a)$：奖励函数；
- $\gamma \in [0,1)$：折扣因子；
- 目标：最大化**期望折扣回报** $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$。

> ⚠️ 教程第 3 页只给了概念框架，没给这套数学符号——我把它们补进来是因为第 4 页的讨论题、以及后面 PPO 章节都会用到。

---

## 4. Cartpole 问题（Training the Cartpole）

### 4.1 任务描述

- 一辆小车**左右移动**，把一根杆子**竖直立稳**（只在一个轴向上运动）。
- 类比：用手指尖把一支笔立起来来回找平衡，但**限制在一个方向**。
- 目标产出：一个策略，让**一队 cartpole 在仿真里同时保持平衡**。

### 4.2 教程留的讨论题（值得自己先想想）

1. 不用强化学习，你会怎么控制 cartpole？
2. 你会用什么算法？
3. 如果要适配**不同杆重、不同小车摩擦、不同电机功率**，你要怎么改？

> 💡 第 3 题的答案就是"传统控制很难泛化"的最好例证——这也是 RL 存在的意义之一。学完第 7 节再回头看这题会更有感觉。

---

## 5. 创建并安装外部项目（Get Started In Isaac Lab）

### 5.1 两种起步方式

1. **云 Launchable**（Brev 一键部署）：省事，但教程提醒——**当前 Launchable 用的是 Isaac Lab 3.0，比本课程测试的版本新**。
2. **本地工作站**：按 [Isaac Lab 官方安装文档](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 装，先查好[系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)。本教程教学用的是**本地 + 外部项目**这条路。

### 5.2 用模板生成器创建项目

Isaac Lab 自带模板生成脚本（不是独立的 cookiecutter）：

```bash
# 1. 激活环境（先装好 Isaac Lab）
conda activate env_isaaclab
# 或 Linux venv: source env_isaaclab/bin/activate

# 2. 进入 Isaac Lab 目录
cd ~/IsaacLab

# 3. 启动模板生成器
./isaaclab.sh --new          # Windows: .\isaaclab.bat --new
```

### 5.3 交互菜单选项（按教程推荐选）

| 提示 | 推荐值 |
|------|--------|
| Task type（任务类型） | **External**（项目建在 Isaac Lab 仓库之外） |
| Project Path | `/home/{你的用户名}/Cartpole` |
| Project name | **Cartpole** |
| Isaac Lab workflow | **Manager-based**（基于管理器） |
| RL library | **skrl** |
| RL algorithms | **PPO**（近端策略优化） |

> ⚠️ 选 Manager-based 时代码按"配置管理器"组织；**Direct-based** 会把代码组织成另一种结构（本教程讲的是 Manager-based）。

### 5.4 安装与验证

在生成的项目根目录里：

```bash
python -m pip install -e source/Cartpole   # "本质上就是把它注册到 Isaac Lab"
python scripts/list_envs.py                 # 列表里能看到 Cartpole = 安装成功
```

> 💡 教程小技巧：用 IDE 打开项目根目录（如 `~/Cartpole`），终端会自动定位到正确位置。

---

## 6. Running the Training（运行并观察训练）

### 6.1 训练命令

```bash
python scripts/skrl/train.py --task=Template-Cartpole-v0
```

- `scripts/skrl/train.py`：SKRL 库的训练脚本；
- `--task=Template-Cartpole-v0`：就是上一步生成并注册的任务名；
- 默认 **4096 个环境并行**；
- 电脑带不动就显式减小：`--num_envs=1000`；
- 训练结束后 **Isaac Sim 会自动关闭，这是正常现象**。

### 6.2 观察训练过程

- **RL = 从交互中学习**：刚开始小车乱跑、杆子根本立不住——很正常；
- 随着训练推进，策略变好，杆子越来越稳；
- 能看到**几千个机器人同时**在训练（教程录像加速 500%）。

### 6.3 视口相机操作（Isaac Sim）

| 操作 | 方法 |
|------|------|
| 移动 | 按住右键 + WASD（Q/E 升降） |
| 旋转 | 按住右键拖动 |
| 缩放 | 滚轮，或 Alt + 右键 |
| 平移 | 按住中键拖动 |

### 6.4 在你电脑上跑（补充）

你的机器是 **RTX 4060（8G）+ 15G 内存**。默认 4096 环境在内存上会偏紧，直接按教程提示用：

```bash
python scripts/skrl/train.py --task=Template-Cartpole-v0 --num_envs=1000
```

> 💡 补充：训练过程一般会写日志到项目 `logs/` 目录，可用 `tensorboard --logdir logs` 打开看奖励曲线和 loss（第 2 节页面里的录像就是配合 TensorBoard 数据做的）。

---

## 7. Analyzing the Code（核心：把 MDP 翻译成 Isaac Lab 代码）

### 7.1 文件位置与总体结构

Manager-based 工作流里，**每个"管理器"就是一个 Python `@configclass`**。本页分析的唯一文件：

```
source/Cartpole/Cartpole/tasks/manager_based/cartpole/cartpole_env_cfg.py
```

整个任务配置的骨架（所有管理器在此组装）：

```python
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    # 场景
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    # 各管理器：观测 / 动作 / 事件 / 奖励 / 结束
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
```

对应第 3 节的映射：**观测管理器 = observation、动作管理器 = action、奖励管理器 = reward、结束管理器 = done、事件管理器 = reset/随机化**。

### 7.2 结束条件（TerminationsCfg）——"done"

```python
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        joint_ids=["slider_to_cart"],   # 小车滑块关节
        bounds=(-3.0, 3.0),             # 滑出 ±3 就算失败
    )
```

两个结束条件：
- `time_out`：回合**超时**（该次采样作废，进入下一回合）；
- `cart_out_of_bounds`：小车**滑出 ±3.0 边界**。

教程原话理由："这回合的结果已经差到离谱，不如早点结束继续下一个。"（训练时间别浪费在垃圾回合上。）

### 7.3 动作（ActionsCfg）

```python
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,       # 输出的力先乘 100
    )
```

只有一个动作：给滑块关节施加**力矩**，让小车左右动。`scale=100.0` 把策略输出放大到实际力矩量级。

### 7.4 观测（ObservationsCfg.PolicyCfg）

```python
class ObservationsCfg:
    class PolicyCfg(ObsGroup):          # 策略看到的观测组
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)   # 相对关节位置
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)   # 相对关节速度
```

两个观测项（顺序固定）：关节相对位置 + 关节相对速度。这就是策略"看到的世界"。

### 7.5 奖励（RewardsCfg）——5 项表（重点）

| 项 | 函数 | 权重 | 目标关节 |
|----|------|------|----------|
| `alive` | `mdp.is_alive`（活着） | **+1.0** | — |
| `terminating` | `mdp.is_terminated`（结束） | **-2.0** | — |
| `pole_pos` | `mdp.joint_pos_target_l2`（杆位置 L2） | **-1.0** | `cart_to_pole`，目标 0.0 |
| `cart_vel` | `mdp.joint_vel_l1`（车速度 L1） | **-0.01** | `slider_to_cart` |
| `pole_vel` | `mdp.joint_vel_l1`（杆角速度 L1） | **-0.005** | `cart_to_pole` |

**设计思路（教程引导的物理思考）：**
- 杆子竖直 = 杆关节角度**贴近 0**（`pole_pos`，-1.0）；
- 结束要重罚（`terminating`，-2.0，比每步活着给的 +1.0 还要重——鼓励别失败）；
- 小车和杆子的速度都压低（`cart_vel` / `pole_vel`）→ 别乱动，行为更稳。

> 💡 权重的相对关系很有讲究：活着 +1、失败 -2，意味着"安稳活两步 ≈ 失败一次"，策略有强烈动机保持不倒下。这就是**把"物理目标"翻译成"数值奖励"**的示范。

### 7.6 奖励函数的实现（教程唯一展开的一个）

`mdp` 包已经内置了大部分奖励函数，教程只展开 `joint_pos_target_l2`。**关键：它对所有并行环境一次性用 PyTorch 张量算**：

```python
def joint_pos_target_l2(env, asset_cfg, target):
    asset = env.scene[asset_cfg.asset_name]                       # 1. 从场景取机器人
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])  # 2. 关节角折到 (-pi, pi)
    return torch.sum(torch.square(joint_pos - target), dim=1)     # 3. 平方距离，按关节维求和
```

三步：
1. 从场景里取 **Articulation**（教程定义："构成 cartpole 机构的物理关节的集合"）；
2. 把关节位置**折到 (-π, π)**（角度周期性的处理）；
3. 算到目标的**平方距离**，沿关节维求和，返回 `(num_envs,)` 的向量。

> ⚠️ 注意 `dim=1`：返回的是**每个环境各一个标量**，不是把所有环境加成一个数。这是 Isaac Lab 并行仿真的习惯——奖励、done 都是按环境维对齐的。

### 7.7 场景配置（CartpoleSceneCfg）

```python
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0))
```

- **地面**（100×100 平面）+ **机器人**（复用 Cartpole 资产，放到每个环境的命名空间 `{ENV_REGEX_NS}/Robot`）+ **穹顶灯**；
- 关键理念：**训练相关元素（地面、光照）放在环境里，不放进行机器人的 USD 文件**——让机器人资产保持"干净"。

### 7.8 Isaac Lab 如何控制仿真（重要概念）

```python
# CartpoleEnvCfg 后置配置
decimation = 2              # 每个策略决策对应 2 次物理步
episode_length_s = 5        # 回合最长 5 秒
sim.dt = 1 / 120            # 物理步长 1/120 秒
sim.render_interval = self.decimation
```

**"Isaac Lab is controlling the simulator"**：仿真不是自由乱跑，而是按 Isaac Lab 的节奏步进——**暂停 → 算观测/决策 → 执行一小段物理 → 重复**。`decimation=2` + `sim.dt=1/120` 意味着策略**每秒只决策 60 次**（动作频率 60Hz），每决策执行 2 个物理子步。

### 7.9 连接关系总结（代码怎么串起来的）

- `ObservationsCfg.PolicyCfg` 继承 `ObsGroup`（策略观测组）；
- `TerminationsCfg` / `ActionsCfg` / `ObservationsCfg` / `RewardsCfg` 都是普通 `@configclass`；
- `CartpoleEnvCfg` 继承 **`ManagerBasedRLEnvCfg`**，把上面所有 + `CartpoleSceneCfg` 组装起来；
- 奖励函数在 `mdp/rewards.py` 里，可查 Isaac Lab API 文档看全部可用函数。

### 7.10 补充：任务是怎么注册的（`__init__.py`，页面没展开但要知道）

项目里 `tasks/__init__.py` 会用 `gym.register` 把环境注册成 gym 任务，这就是 `Template-Cartpole-v0` 这个名字的由来（skrl 脚本按这个名字找到环境）：

```python
gym.register(
    id="Template-Cartpole-v0",
    entry_point="...:CartpoleEnv",
    disable_env_checker=True,
)
```

> 💡 记住这条链路：**菜单选 External + Manager-based + skrl + PPO → 生成 `cartpole_env_cfg.py` → `pip install -e` 注册 → `list_envs.py` 验证 → `train.py --task=Template-Cartpole-v0` 开始训练**。整个模块就是这条链路。

---

## 8. 本节小结（背下来）

1. **分工**：Isaac Sim 管仿真/渲染，Isaac Lab 管 RL 训练；技术栈 OpenUSD → Omniverse → Isaac Sim → Isaac Lab。
2. **RL 定义**：从交互中学习；循环 = state → observation → policy → action → reward。
3. **三种奖励**：sparse / dense / shaped；塑形奖励=注入领域知识引导学习。
4. **奖励设计原则**：定义目标约束而非步骤（走路例子：躯干竖直、脚落地、朝向目标、惩罚过低）。
5. **MDP**：agent/state/action/reward/policy；映射到 Isaac Lab = 选 agent 资产 + 环境 + 写 reward/done/reset 函数。
6. **加速秘诀**：GPU 大规模并行（4096 环境）、headless 省资源、模块化换组件、领域随机化助 sim-to-real。
7. **外部项目流程**：`./isaaclab.sh --new` 按菜单选（External / Manager-based / skrl / PPO）→ `pip install -e source/Cartpole` → `list_envs.py` 验证。
8. **训练命令**：`python scripts/skrl/train.py --task=Template-Cartpole-v0`，带不动就 `--num_envs=1000`。
9. **cartpole 奖励 5 项**：活着 +1、失败 -2、杆位置 L2 -1、车速度 -0.01、杆角速度 -0.005。
10. **代码骨架**：`CartpoleEnvCfg(ManagerBasedRLEnvCfg)` 组装 scene + 5 个管理器（obs/act/events/reward/term）。

## 9. 自测小练习

1. **概念题**：State 和 Observation 的区别是什么？为什么真机上没有 privileged data？
2. **设计题**：如果杆子只能倒一次（杆太长），你会怎么改 7.5 的奖励权重？
3. **代码题**：`joint_pos_target_l2` 的返回值形状是什么？为什么（联系 `dim=1` 和 4096 个环境）？
4. **命令题**：写出从头创建项目到开始训练的完整命令序列（菜单选项除外）。
5. **连线题**：MDP 的 agent / environment / action / reward / done 分别对应 cartpole_env_cfg.py 里的哪一块？

## 10. 下一步预告

- **08 Running our Policy**：把训好的策略拿出来，在 Isaac Sim 里跑一遍看效果（还剩这个 + Conclusion 没学，学完本模块就结束）。
- 之后是 **Train Your Second Robot** 和 **sim-to-real** 模块。
- 与你的主线衔接：本模块是 Isaac Lab 入门，之后 **HIMLoco** 章节会接触真正四足机器人的训练代码。
