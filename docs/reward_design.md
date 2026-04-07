# `reward_design.md`：三连杆高低杠环境奖励设计建议

## 1. 设计原则

这个任务本质上是一个长时序、混杂切换、稀疏成功事件问题。

如果只给“抓住低杆 +1，其他 0”的稀疏奖励，训练大概率会非常慢。因此第一版奖励必须满足：

- 让智能体知道“往哪里飞”
- 让智能体知道“什么姿态更容易抓”
- 让智能体避免过大冲击
- 让智能体逐渐学会完整流程

建议从**分阶段 shaping reward** 开始，后期再逐渐减弱 shaping。

---

## 2. 总体形式

建议统一使用：

```python
reward = (
    r_mode_progress
    + r_pose
    + r_contact
    + r_success
    - r_ctrl
    - r_impact
    - r_fail
)
```

其中每一项都可以按模式分别定义。

---

## 3. 高杆阶段奖励 `HIGH_BAR`

目标不是长期停在高杆上，而是：

- 获得有利于腾空的能量状态
- 在合适时机脱手
- 让质心和手部运动趋势朝向低杆

### 3.1 可选项

#### A. 摆动能量奖励

```python
r_energy = w_energy * clip(E - E_ref, min_val, max_val)
```

作用：鼓励机器人摆起，而不是一直在低能量区小幅振荡。

#### B. 朝向低杆的速度奖励

令：

- `bar_to_low = low_bar_pos - high_bar_pos`
- `v_cm` 为质心速度

可用：

```python
r_direction = w_dir * cosine_similarity(v_cm, bar_to_low)
```

作用：鼓励形成朝向低杆的腾空初速度。

#### C. 脱手时机奖励

如果动作中有 `release_cmd`：

- 在极不合理状态下脱手，给负奖励
- 在高能量、较合理姿态下脱手，给小正奖励

---

## 4. 腾空阶段奖励 `FLIGHT`

这是最关键阶段，建议把奖励重点放在：

- 手部接近低杆
- 姿态进入抓握窗口
- 飞行不发散

### 4.1 距离奖励

```python
r_dist = -w_dist * np.linalg.norm(hand_pos - low_bar_pos)
```

这是最直接、最重要的一项。

也可以改成势函数差分形式：

```python
r_progress = w_prog * (prev_dist - curr_dist)
```

这样更稳定。

### 4.2 姿态窗口奖励

对抓握窗口 `Q_catch`，定义离窗口距离：

```python
def window_violation(q, q_min, q_max):
    # inside -> 0, outside -> positive
```

奖励：

```python
r_pose = -w_pose * window_violation(q, q_min, q_max)
```

作用：让机器人即使还没碰到低杆，也先学会“用什么姿态接近”。

### 4.3 角速度惩罚

```python
r_spin = -w_spin * np.linalg.norm(dq)
```

作用：防止空中出现数值上很快但物理上不可抓的极端翻转。

### 4.4 高度或坠落惩罚

如果已经明显低于合理区域：

```python
r_drop = -w_drop
```

作用：缩短失败轨迹，提高训练效率。

---

## 5. 低杆接触时奖励

接触是一个离散事件，建议显式给奖励。

### 5.1 接触奖励

只要发生低杆接触：

```python
r_contact = +w_contact
```

作用：先让 agent 学会“碰到低杆”。

### 5.2 成功抓握奖励

若 `catch_ok == True`：

```python
r_success = +w_success_large
```

这是整个任务中最大的单次正奖励。

### 5.3 抓握失败惩罚

若接触但 `catch_ok == False`：

```python
r_fail = -w_fail_large
```

这样可以明确区分：

- “没碰到低杆”
- “碰到了但没抓稳”

---

## 6. 冲击惩罚

原始建模里已经给出接触冲量与角速度跃变判定，这些非常适合直接转成奖励。

### 6.1 冲量惩罚

```python
r_impact = -w_impact * np.linalg.norm(impulse)
```

或只在超阈值时惩罚：

```python
r_impact = -w_impact * max(0.0, impulse_norm - impulse_limit)
```

### 6.2 角速度跃变惩罚

```python
r_jump = -w_jump * np.linalg.norm(dq_jump)
```

作用：鼓励更“柔和”的抓握。

---

## 7. 控制代价

所有模式下都应加入控制代价：

```python
r_ctrl = w_ctrl * (tau2**2 + tau3**2)
```

总奖励中减去它：

```python
reward -= r_ctrl
```

作用：

- 避免输出过大力矩
- 提高策略平滑性
- 与原建模中的能耗优化方向一致

---

## 8. 终止奖励设计

建议明确区分：

### 成功终止

```python
reward += +100.0
```

### 抓握失败终止

```python
reward += -50.0
```

### 状态越界 / 落地 / 发散终止

```python
reward += -30.0
```

数值只是示例，后续需调参。

---

## 9. 推荐的第一版奖励模板

下面给出一个非常适合第一版起步的模板。

### `HIGH_BAR`

```python
reward = (
    0.5 * energy_progress
    + 0.2 * direction_score
    - 0.001 * (tau2**2 + tau3**2)
)
```

### `FLIGHT`

```python
reward = (
    2.0 * (prev_distance_to_low_bar - curr_distance_to_low_bar)
    - 0.2 * pose_window_violation
    - 0.001 * (tau2**2 + tau3**2)
    - 0.01 * np.linalg.norm(dq)
)
```

### 接触瞬间

```python
if contact:
    reward += 10.0
if catch_ok:
    reward += 100.0
else:
    reward -= 20.0
```

### 冲击惩罚

```python
reward -= 0.05 * impulse_norm
reward -= 0.02 * np.linalg.norm(dq_jump)
```

---

## 10. 训练阶段化建议

### 阶段 1：只学会接近低杆

奖励重点：

- 距离低杆越来越近
- 空中姿态不过分发散

这一阶段甚至可以先不要求严格抓握成功。

### 阶段 2：学会进入抓握窗口

奖励重点：

- 距离低杆近
- 姿态进入 `Q_catch`
- 接触时冲击不要太大

### 阶段 3：学会抓稳

奖励重点：

- 成功抓握
- 冲击较小
- 抓后稳定保持

这是比较稳的 curriculum learning 路线。

---

## 11. 不建议的做法

第一版尽量不要：

- 只用最终成功奖励
- 一开始就把惩罚项设得非常重
- 一开始就把所有物理约束做成 hard constraint
- 奖励项过多、过复杂

原因是：环境本身已经很复杂，过早引入太多约束会让 agent 几乎探索不到正样本。

---

## 12. 推荐调参顺序

调奖励时建议按这个顺序：

1. 先确保 `distance_to_low_bar` 的 shaping 足够强
2. 再加入 `pose_window_violation`
3. 再加入 `contact / catch` 离散奖励
4. 最后加入 `impulse` 与 `dq_jump` 惩罚
5. 最后微调控制代价

---

## 13. 一句话总结

第一版奖励最重要的不是“绝对物理完美”，而是让策略先学会：

1. 摆起来；
2. 飞过去；
3. 以更像能抓住的姿态接近低杆；
4. 抓住时别撞得太狠。

