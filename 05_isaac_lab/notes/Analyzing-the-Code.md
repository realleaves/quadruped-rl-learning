# Isaac Lab 教程 07 —— 分析训练代码（Analyzing the Code）学习笔记

> 📖 来源：[NVIDIA Docs — Train Your First Robot in Isaac Lab → 07 Analyzing the Code](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/train-your-first-robot-with-isaac-lab/07-analyzing-the-code.html)
> 🗓️ 学习日期：2026-08-13
> 🧭 学习路径：PyTorch → 深度学习 → 强化学习 → PPO → Isaac Lab → HIMLoco → 四足机器人部署
> 📌 本次范围：Section 07「分析训练代码」—— 核心文件是 `cartpole_env_cfg.py`
> 🔗 相关笔记：[[Train-First-Robot-Cartpole]]（同教程，讲 Overview + RL 基础 + 运行训练）

---

## 0. 这个页面在讲什么（一句话）

把上一节的 **MDP 框架**，翻译成 Isaac Lab **Manager-based（管理器）工作流**的实际代码：

> 强化学习里的每个 MDP 元素，都被定义成一个 Python 类（用 `@configclass` 装饰），各自管理环境的一部分。

| MDP 元素 | Manager 类 | 管的什么 |
|---|---|---|
| 奖励 | `RewardsCfg` | 每一步给多少分 |
| 观测 | `ObservationsCfg` | agent 能"看到"什么 |
| 动作 | `ActionsCfg` | agent 能"动"什么 |
| 终止 | `TerminationsCfg` | 一局什么时候结束 |

这四个 + 场景/环境配置，全部定义在**同一个文件**里：

```
source/Cartpole/Cartpole/tasks/manager_based/cartpole/cartpole_env_cfg.py
```

> 💡 `@configclass` 是 Isaac Lab 对 Python `dataclass` 的扩展：除了像 dataclass 一样存配置，还支持 `__post_init__`（初始化后回调）等自定义行为。后面 `CartpoleEnvCfg` 里的 `__post_init__` 就是靠它触发的。

---

## 1. 终止条件 `TerminationsCfg`

**逻辑**：小车上到哪一步这局就可以放弃了？两条件，"达其一即终止"：

1. **超时**（`time_out`）——跑满时长，正常结束
2. **小车出界**（`cart_out_of_bounds`）——滑出范围 (-3.0, 3.0)，说明已经失控

> 原文："At some point, the results from this episode are so far off it's best to move on."（有些时候，这一局的结果已经差得离谱，不如直接换下一局。）

```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) 超时
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) 小车出界
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "bounds": (-3.0, 3.0)},
    )
```

要点：
- `func` 指定"判定函数"（来自 Isaac Lab 内置的 `mdp` 包）
- `params` 传参：用 `SceneEntityCfg` 指定**作用于哪个关节**（这里是 `slider_to_cart`，即小车那根导轨）
- `bounds=(-3.0, 3.0)` 是位置阈值，单位米

---

## 2. 动作 `ActionsCfg`

**逻辑**：唯一的动作就是给小车导轨一个**力**（effort），让它左右移动。`scale=100.0` 表示把策略输出的动作数值放大 100 倍作为实际施加的力（单位 N）。

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    )
```

---

## 3. 观测 `ObservationsCfg`

**逻辑**：agent 观测小车和杆的状态——位置 + 速度（共 2 维）。注意 `PolicyCfg` 嵌套在 `ObsGroup` 里，表示这一组观测是**喂给策略网络**的（Policy group）。

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 观测项（顺序即维度顺序，很重要）
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)   # 关节相对位置
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)   # 关节相对速度
```

> 💡 为什么观测能这么"精简"？因为环境里其实有两个关节：`slider_to_cart`（小车平动）和 `cart_to_pole`（杆子转动）。`joint_pos_rel` 等函数会返回**所有**关节的 pos/vel，这里只列了两个 `ObsTerm`，实际观测向量就是"小车位置、小车速度、杆角度、杆角速度"拼起来。教程只展示了写法，真正的函数来自 `mdp` 包。

---

## 4. 奖励 `RewardsCfg`（本页重点）

**逻辑**：倒立摆是**连续任务**（没有确定终点），所以奖励设计用"常驻生存奖励 + 失败惩罚 + 塑形项（shaping）"的组合。共 5 项，全部按权重加权求和：

| # | 名称 | 函数 | 权重 | 含义 |
|---|---|---|---|---|
| 1 | `alive` | `mdp.is_alive` | **+1.0** | 活着就有奖励（常驻生存奖励） |
| 2 | `terminating` | `mdp.is_terminated` | **-2.0** | 提前终止 = 惩罚（失败惩罚） |
| 3 | `pole_pos` | `mdp.joint_pos_target_l2` | **-1.0** | 杆没立正 = 惩罚（**主任务**） |
| 4 | `cart_vel` | `mdp.joint_vel_l1` | **-0.01** | 小车动太快 = 轻微惩罚（塑形） |
| 5 | `pole_vel` | `mdp.joint_vel_l1` | **-0.005** | 杆转太快 = 轻微惩罚（塑形） |

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) 常驻生存奖励
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) 失败惩罚
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) 主任务：杆子立直（目标角度 0）
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "target": 0.0},
    )
    # (4) 塑形：压低小车速度
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) 塑形：压低杆子角速度
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

### 4.1 理解奖励的 3 个关键点

1. **函数通过 `func` 传入**——Manager 工作流里，奖励计算函数不是写死逻辑，而是作为参数传进去。
2. **一次算所有环境**——奖励函数用 PyTorch tensor，**同时对 4096 个并行环境**算奖励（批量并行训练）。
3. **绝大多数函数来自 `mdp` 内置包**——只有 `joint_pos_target_l2` 是教程自己写的自定义函数，其余全是 `isaaclab.envs.mdp` 里的现成货（完整列表见 mdp API 文档）。

### 4.2 唯一的自定义奖励函数 `joint_pos_target_l2`

```python
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float,
                        asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # 从场景里取出机器人（关节集合体）
    asset: Articulation = env.scene[asset_cfg.name]
    # 把关节位置 wrap 到 (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # 平方误差：(joint_pos - target)^2，沿 dim=1 求和
    return torch.sum(torch.square(joint_pos - target), dim=1)
```

**逐步拆解**（教程原文逻辑）：

1. `env.scene[asset_cfg.name]` → 拿到 `Articulation`（一组物理关节，构成整个 cartpole 机构）
2. `wrap_to_pi(...)` → 把关节角 wrap 到 **-π ~ π**(角度周期化,避免角度跳变导致奖励突变)
3. `torch.square(joint_pos - target)` → 平方距离（L2 范数）
4. `torch.sum(..., dim=1)` → 对每个环境内的关节求和，输出形状 `(num_envs,)`——**每个环境一个标量奖励**

---

## 5. 场景与环境配置

### 5.1 场景 `CartpoleSceneCfg`

**逻辑**：搭一个"训练舞台"——地面、机器人、灯光。好处：训练才需要的元素（灯光、道具）**不用塞进机器人的 USD 文件**，跟机器人本体解耦；以后想加道具（比如抓取物体）也在这里加。

```python
@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    # 机器人（CARTPOLE_CFG 是机器人的定义，用 replace() 把 prim 路径换到本环境的命名空间）
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 半球灯
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

> 💡 `{ENV_REGEX_NS}` 是 Isaac Lab 的**环境命名空间占位符**——4096 个环境各有自己的命名空间，`replace()` 让每个环境都有一份独立的机器人实例。

### 5.2 环境配置 `CartpoleEnvCfg`（所有 manager 的集合点）

```python
@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    # 场景设置
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4096, env_spacing=4.0)
    # 基本设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP 设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # 常规设置
        self.decimation = 2          # 物理步数 / 策略步
        self.episode_length_s = 5    # 每局 5 秒
        # viewer 设置
        self.viewer.eye = (8.0, 0.0, 5.0)   # 相机位置
        # 仿真设置
        self.sim.dt = 1 / 120        # 物理步长 1/120 秒
        self.sim.render_interval = self.decimation
```

### 5.3 关键的时序参数关系（必须搞懂）

```
策略频率  = 物理频率 / decimation
         = (1 / 120 s)⁻¹ / 2
         = 120 Hz / 2 = 60 Hz

每局总步数 ≈ 5 s × 60 Hz = 300 步
```

- `sim.dt = 1/120`：物理仿真步长（Isaac Sim 每步推进 1/120 秒）
- `decimation = 2`：策略每 1 步，物理仿真走 2 步 → **策略控制频率 60 Hz**
- `episode_length_s = 5`：一局 5 秒
- 教程强调：因为 **Isaac Lab 接管了仿真器**，它可以暂停 → 做计算 → 一次推固定时间，而不是让仿真自由地持续跑——这是能批量训练的基础。

### 5.4 参数速查表

| 元素 | 参数 | 值 |
|---|---|---|
| Scene | `num_envs` | 4096（并行环境数） |
| Scene | `env_spacing` | 4.0（环境间距） |
| Termination | `cart_out_of_bounds` bounds | (-3.0, 3.0) |
| Action | `joint_effort` scale | 100.0 |
| Reward | `alive` | +1.0 |
| Reward | `terminating` | -2.0 |
| Reward | `pole_pos` | -1.0, target 0.0 |
| Reward | `cart_vel` | -0.01 |
| Reward | `pole_vel` | -0.005 |
| Env | `decimation` | 2 |
| Env | `episode_length_s` | 5 |
| Env | `sim.dt` | 1/120 |
| Env | `viewer.eye` | (8.0, 0.0, 5.0) |

---

## 6. 本页没讲、由后续页面覆盖的内容（别在这页找）

- `train.py` / `scripts/` 的工程结构 → 教程其他页面
- `CartpolePPORunnerCfg`（PPO 超参，如 `num_epochs`、`minibatch_size`、`learning_rate`）→ **Running the Training 页面**
- 训练命令及 `--task` / `--num_envs` 等 CLI 参数含义 → **Running the Training 页面**

> 这些内容在 [[Train-First-Robot-Cartpole]] 笔记里已有整理，可以直接回看。

---

## 7. 一句话总结

> **这一个文件 = 整个 RL 环境的"宪法"**：观测、动作、奖励、终止、场景、仿真时序全在这里用 `@configclass` 声明式定义。策略只管把观测映射成动作，环境的"规则"由这些 config 决定——这就是 Manager-based 工作流的顶层设计：**规则与学习分离**，换任务只需改配置，不用改训练代码。
