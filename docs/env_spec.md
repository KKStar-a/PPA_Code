# `env_spec.md`：三连杆高杆-腾空-低杆 Gym 环境规格说明

> 这份文档是给 Codex/IDE agent 直接读取的实现规格说明。目标不是写论文，而是指导代码落地。

---

## 1. 目标

实现一个 Gymnasium 风格环境 `ThreeLinkHighLowBarEnv`，描述三连杆机器人从高杆摆动、脱手腾空、抓握低杆的全过程。

要求：

- 具有固定维度 observation；
- 内部存在离散模式 `HIGH_BAR / FLIGHT / LOW_BAR`；
- 支持 `reset()` 与 `step(action)`；
- 支持事件切换与 reset map；
- 第一版可训练，不要求一开始就完全复现论文级动力学。

---

## 2. 环境类型

- API：Gymnasium
- 时间：离散仿真步长 `dt`
- 内部积分：可以先用显式 Euler / semi-implicit Euler
- 动力学计算：NumPy 优先
- 第一版训练算法目标：PPO

---

## 3. 离散模式

```python
HIGH_BAR = 0
FLIGHT = 1
LOW_BAR = 2
```

可选终止标签：

```python
TERMINAL_SUCCESS = 10
TERMINAL_FAIL = 11
```

环境内部保留 `self.mode`。

---

## 4. 观测空间

为了兼容 Gym，所有模式下 observation 维度必须固定。

建议第一版 observation 为：

```python
obs = [
    mode_flag,
    q1, q2, q3,
    dq1, dq2, dq3,
    x0, y0,
    dx0, dy0,
]
```

共 11 维。

说明：

- 在 `HIGH_BAR` 和 `LOW_BAR` 模式下，`x0,y0,dx0,dy0` 虽然不是独立自由度，仍保留在 observation 中；
- 这些量可通过当前模式下的几何关系计算得到；
- 保持维度固定比物理上最简更重要。

后续如果需要，可扩展额外特征：

- `distance_to_low_bar`
- `catch_window_margin`
- `release_ready_flag`
- `impact_estimate`

---

## 5. 动作空间

建议第一版使用连续动作：

```python
action = [tau2, tau3, release_cmd]
```

其中：

- `tau2`：关节 2 力矩
- `tau3`：关节 3 力矩
- `release_cmd`：脱手指令

动作范围示例：

```python
Box(low=[-tau_max, -tau_max, -1.0], high=[tau_max, tau_max, 1.0])
```

脱手逻辑：

- 当 `mode == HIGH_BAR` 且 `release_cmd > release_threshold` 时，触发切换到 `FLIGHT`
- 后续可升级为：只有满足物理 guard 时才允许脱手

---

## 6. 状态定义

内部连续状态建议统一存为：

```python
state = {
    "mode": int,
    "q": np.ndarray shape (3,),
    "dq": np.ndarray shape (3,),
    "p": np.ndarray shape (2,),
    "dp": np.ndarray shape (2,),
}
```

其中：

- `q = [q1,q2,q3]`
- `dq = [dq1,dq2,dq3]`
- `p = [x0,y0]`
- `dp = [dx0,dy0]`

在不同模式下的解释：

### `HIGH_BAR`

- `q,dq` 为真实主状态
- `p,dp` 由高杆几何关系推导得到，主要用于统一 observation 和切换初始化

### `FLIGHT`

- `q,dq,p,dp` 都是独立状态

### `LOW_BAR`

- `q,dq` 为真实主状态
- `p,dp` 由低杆几何关系推导得到

---

## 7. 参数组织

参数统一放在 `params.py` 中，使用 dataclass：

```python
@dataclass
class EnvParams:
    dt: float
    g: float
    tau_max: float
    release_threshold: float
    max_steps: int

    # geometry
    l1: float
    l2: float
    l3: float
    lc1: float
    lc2: float
    lc3: float

    # masses / inertias
    m1: float
    m2: float
    m3: float
    I1: float
    I2: float
    I3: float
    b1: float

    # bar positions
    high_bar_pos: np.ndarray  # [0, 0]
    low_bar_pos: np.ndarray   # [Dx, Dy]
    catch_radius: float

    # catch thresholds
    kappa: float
    dq_jump_max: np.ndarray   # shape (3,)
    q_catch_min: np.ndarray   # shape (3,)
    q_catch_max: np.ndarray   # shape (3,)
```

---

## 8. 动力学接口

### 8.1 高杆阶段

文件：`dynamics_high_bar.py`

函数接口：

```python
def f_high_bar(q: np.ndarray, dq: np.ndarray, u: np.ndarray, params: EnvParams):
    """Return ddq for fixed-base high-bar dynamics."""
```

第一版要求：

- 接受 `q,dq,u=[tau2,tau3]`
- 返回 `ddq`
- 可先用简化版占位实现，但函数接口必须稳定

升级版要求：

- 使用完整 `M(q)`, `H(q,dq)`, `G(q)`
- 解线性方程得到 `ddq`

### 8.2 腾空阶段

文件：`dynamics_flight.py`

建议接口：

```python
def f_flight(state: dict, u: np.ndarray, params: EnvParams):
    """Return derivatives for floating-base flight mode."""
```

第一版简化要求：

- `p_dot = dp`
- `dp_dot = [0, -g]`
- `q_dot = dq`
- `dq_dot` 可先采用简化内部姿态模型

第二版升级：

- 使用 5 自由度浮动基座模型或 Schur 补降维模型

### 8.3 低杆阶段

文件：`dynamics_low_bar.py`

接口与高杆阶段一致：

```python
def f_low_bar(q: np.ndarray, dq: np.ndarray, u: np.ndarray, params: EnvParams):
    """Return ddq for fixed-base low-bar dynamics."""
```

第一版可以先与 `f_high_bar` 共用同一动力学骨架。

---

## 9. 几何辅助函数

建议新建 `kinematics.py`，提供：

```python
def hand_pos_from_high_bar(q, params) -> np.ndarray

def hand_vel_from_high_bar(q, dq, params) -> np.ndarray

def hand_pos_from_low_bar(q, params) -> np.ndarray

def hand_vel_from_low_bar(q, dq, params) -> np.ndarray

def center_of_mass(q, dq, params) -> tuple[np.ndarray, np.ndarray]
```

说明：

- 第一版即使实现不完全严格，也应保证接口齐全；
- 切换逻辑与奖励都需要这些函数。

---

## 10. 模式切换逻辑

文件：`events.py`

### 10.1 `HIGH_BAR -> FLIGHT`

第一版：

```python
def should_release(release_cmd: float, env_state, params) -> bool
```

逻辑：

- 当 `release_cmd > threshold` 时允许脱手

升级版：

- 加入径向约束力过零 guard
- 即 `g_r(q,dq,ddq) ~ 0`

切换动作：

- `mode = FLIGHT`
- 用高杆阶段几何映射得到当前 `p, dp`
- 保留 `q, dq`

### 10.2 `FLIGHT -> LOW_BAR`

```python
def check_low_bar_contact(p: np.ndarray, dp: np.ndarray, params: EnvParams):
    """Return contact flag and geometric info."""
```

判定：

```python
phi = ||p - p_L|| - r_c
contact = (phi <= 0) and ((p - p_L) @ dp < 0)
```

### 10.3 抓稳判定

```python
def check_catch_success(q, dq, p, dp, params, aux) -> dict:
    """Return flags: pre_ok, impact_ok, catch_ok, impulse, dq_jump."""
```

要求检查：

1. 接触成立
2. 姿态处于 `Q_catch`
3. 冲量不超过阈值
4. 角速度跃变不超过阈值

---

## 11. Reset Map

文件：`reset_maps.py`

关键接口：

```python
def reset_map_flight_to_low_bar(q, dq, p, dp, params):
    """Return q_plus, dq_plus, impulse, dq_jump."""
```

核心公式：

```python
q_plus = q_minus

# dq_plus = dq_minus + M^{-1}(q_minus) J_cm(q_minus)^T dp_minus
```

输出至少包含：

- `q_plus`
- `dq_plus`
- `impulse`
- `dq_jump`

环境逻辑：

- 若抓稳成功：切入 `LOW_BAR`
- 否则：`done=True`, `terminated_by="catch_failed"`

---

## 12. 奖励函数

建议采用分项奖励，而不是纯稀疏奖励。

### 12.1 通用结构

```python
reward = (
    w_progress * progress_reward
    + w_pose * pose_reward
    + w_contact * contact_reward
    + w_success * success_bonus
    - w_ctrl * control_cost
    - w_impact * impact_penalty
    - w_fail * fail_penalty
)
```

### 12.2 第一版建议

#### `HIGH_BAR`

- 鼓励摆动能量增长
- 鼓励质心速度有利于飞向低杆
- 惩罚过大控制

#### `FLIGHT`

- 奖励手部靠近低杆
- 奖励姿态接近抓握窗口
- 惩罚过快下落、过大角速度

#### `LOW_BAR`

- 奖励成功抓杠
- 奖励抓后保持若干步稳定
- 惩罚冲击过大

### 12.3 终止奖励

- 成功抓稳：大正奖励
- 未命中或抓握失败：大负奖励
- 状态越界：负奖励

---

## 13. 终止条件

环境应返回 `terminated` / `truncated`。

### `terminated=True` 的情况

- 成功抓稳并保持达到阈值
- 低杆抓握失败
- 超出几何边界
- 关节角 / 角速度发散
- 身体触地或显著不物理

### `truncated=True` 的情况

- 达到 `max_steps`

---

## 14. `reset()` 设计

建议默认初始模式为 `HIGH_BAR`。

初始状态：

```python
q ~ around hanging posture

dq ~ small random noise
```

并计算：

- `p = hand_pos_from_high_bar(q)`
- `dp = hand_vel_from_high_bar(q, dq)`

随机化建议：

- `q1,q2,q3` 在小范围内随机扰动
- `dq` 小范围随机扰动
- 后续可做 domain randomization

---

## 15. `step()` 伪代码

```python
def step(action):
    tau2, tau3, release_cmd = action

    if mode == HIGH_BAR:
        ddq = f_high_bar(q, dq, [tau2, tau3], params)
        q, dq = integrate(q, dq, ddq)
        p = hand_pos_from_high_bar(q)
        dp = hand_vel_from_high_bar(q, dq)
        if should_release(release_cmd, state, params):
            mode = FLIGHT

    elif mode == FLIGHT:
        state_dot = f_flight(state, [tau2, tau3], params)
        integrate full state
        if check_low_bar_contact(p, dp, params):
            catch_info = check_catch_success(q, dq, p, dp, params, aux=None)
            if catch_info["catch_ok"]:
                q, dq = reset_map_flight_to_low_bar(q, dq, p, dp, params)
                p = hand_pos_from_low_bar(q)
                dp = hand_vel_from_low_bar(q, dq)
                mode = LOW_BAR
            else:
                terminated = True

    elif mode == LOW_BAR:
        ddq = f_low_bar(q, dq, [tau2, tau3], params)
        q, dq = integrate(q, dq, ddq)
        p = hand_pos_from_low_bar(q)
        dp = hand_vel_from_low_bar(q, dq)
        if stable_for_enough_steps:
            terminated = True
            success = True

    obs = build_obs(...)
    reward = compute_reward(...)
    return obs, reward, terminated, truncated, info
```

---

## 16. `info` 字典建议

```python
info = {
    "mode": mode,
    "distance_to_low_bar": float,
    "contact": bool,
    "catch_ok": bool,
    "impulse_norm": float,
    "dq_jump": np.ndarray,
    "released": bool,
    "success": bool,
}
```

这样便于调试和训练可视化。

---

## 17. 训练脚本要求

`scripts/train_ppo.py` 至少应做到：

- 创建环境
- 使用 Stable-Baselines3 PPO
- 定期保存 checkpoint
- 使用 tensorboard 日志
- 训练后保存最终模型

`scripts/random_rollout.py` 至少应做到：

- 随机动作 rollout 一条轨迹
- 打印 mode 切换时刻
- 打印是否命中低杆

`scripts/visualize_episode.py` 至少应做到：

- 可视化高杆、低杆位置
- 可视化手部轨迹
- 可视化各关节角时间历程

---

## 18. 单元测试要求

至少包含以下测试：

### `test_event_switch.py`

- 高杆脱手后 mode 从 `HIGH_BAR` 切为 `FLIGHT`
- 腾空接触低杆时触发接触事件

### `test_reset_map.py`

- `reset_map_flight_to_low_bar()` 输出维度正确
- `q_plus == q_minus`
- `dq_plus` 与公式一致

### `test_dynamics.py`

- `f_high_bar` / `f_flight` / `f_low_bar` 返回维度正确
- 数值不会立即 NaN

---

## 19. 第一版不做的事

为了尽快得到可训练原型，第一版明确不要求：

- 完整复现所有显式长公式
- 一开始就做 OCP
- 一开始就严格实现径向反力过零脱手
- 一开始就处理复杂正握/反握模式
- 一开始就做材料接触刚度与摩擦模型

---

## 20. 开发顺序

推荐让 Codex 按以下顺序实现：

1. 建环境骨架与参数文件
2. 实现 observation / action / reset / step 框架
3. 实现占位版高杆与低杆动力学
4. 实现简化版腾空动力学
5. 实现 `HIGH_BAR -> FLIGHT`
6. 实现 `FLIGHT -> LOW_BAR`
7. 实现 reset map
8. 实现奖励与终止条件
9. 加测试
10. 再逐步替换为完整动力学

---

## 21. 成功标准

第一版环境达到以下条件即可视为完成：

- 可以 `import` 和实例化
- `reset()` 正常返回 observation
- `step()` 连续调用不崩溃
- 观测维度固定
- 存在真实的 mode 切换
- 随机 rollout 能从高杆进入腾空
- 至少有概率接触低杆
- 抓握成功/失败都有明确逻辑
- PPO 训练脚本可启动

