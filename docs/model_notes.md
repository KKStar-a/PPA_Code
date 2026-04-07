# 三连杆高低杠 Park 空翻建模笔记（面向代码实现）

## 1. 项目目标

本文档从原始建模稿中提炼出**对代码实现最重要的动力学与混杂切换信息**，用于后续在 Gymnasium / Stable-Baselines3 / Codex 中实现一个可训练的强化学习环境。

目标任务为：

- 机器人初始时连接在高杆上摆动；
- 在合适时机脱手，进入腾空阶段；
- 在空中完成姿态调整；
- 触及低杆并完成抓握；
- 切换到低杆固定基座模型继续运动。

该系统本质上是一个**连续—离散混杂系统（hybrid system）**。

---

## 2. 模式划分

建议在代码中显式区分三个连续模式：

1. `HIGH_BAR`：高杆固定基座阶段
2. `FLIGHT`：双手离杠腾空阶段
3. `LOW_BAR`：低杆抓握后的固定基座阶段

此外可在环境逻辑中设置两个吸收终止状态：

- `SUCCESS`：成功抓住低杆并满足抓稳条件
- `FAIL`：脱手失败、未命中、冲击过大、状态越界等

---

## 3. 高杆阶段：固定基座 PAA 模型

### 3.1 广义坐标

高杆阶段采用平面三连杆欠驱动模型：

\[
q = \begin{bmatrix} q_1 & q_2 & q_3 \end{bmatrix}^T,
\qquad
q_a = \begin{bmatrix} q_2 & q_3 \end{bmatrix}^T
\]

其中：

- \(q_1\)：被动关节角（抓握高杆处）
- \(q_2, q_3\)：主动关节相对角

### 3.2 控制输入

\[
U = \begin{bmatrix}
-b_1 \dot q_1 \\
\tau_2 \\
\tau_3
\end{bmatrix}
\]

其中第 1 关节为被动关节，仅考虑粘滞摩擦；第 2、3 关节由电机驱动。

### 3.3 动力学方程

高杆阶段动力学统一写为：

\[
M(q)\ddot q + H(q,\dot q) + G(q) = U
\]

实现时可先按如下思路处理：

- 第一版代码直接写出 `M(q)`, `H(q,dq)`, `G(q)`；
- 或先搭接口，用占位版动力学保证环境先跑通；
- 后续再将原文中的显式公式完整替换进去。

### 3.4 工程备注

原稿同时给出了：

- 质心在杆轴上的标准 PAA 模型；
- 存在质心偏置角 \(\gamma_i\) 的一般模型。

代码实现建议：

- **第一版环境先用标准 PAA 模型**；
- 参数稳定后再升级到一般质心偏置模型。

---

## 4. 腾空阶段：浮动基座模型

当双手离开高杆后，系统从受约束的欠驱动系统切换为平面内浮动基座系统。

原稿给出了两种数学上等价的写法。

### 4.1 方案 A：5 自由度浮动基座模型

广义坐标取为：

\[
q_f = \begin{bmatrix} x_0 & y_0 & q_1 & q_2 & q_3 \end{bmatrix}^T
= \begin{bmatrix} p^T & q^T \end{bmatrix}^T
\]

其中 \(p = [x_0, y_0]^T\) 表示手爪在惯性系中的平移坐标。

动力学可写成分块形式：

\[
\begin{bmatrix}
M_{total}I_{2\times2} & J_{cm}(q) \\
J_{cm}^T(q) & M(q)
\end{bmatrix}
\begin{bmatrix}
\ddot p \\
\ddot q
\end{bmatrix}
+
\begin{bmatrix}
H_p(q,\dot q) \\
H(q,\dot q)
\end{bmatrix}
+
\begin{bmatrix}
0 \\
M_{total} g \\
G(q)
\end{bmatrix}
=
\begin{bmatrix}
0 \\
0 \\
0 \\
\tau_2 \\
\tau_3
\end{bmatrix}
\]

这套形式最适合代码实现，因为它与抓杠事件切换最自然。

### 4.2 方案 B：Schur 补降维姿态模型

通过消去平移变量，可得到 3 自由度降维模型：

\[
\bar M(q)\ddot q + \bar H(q,\dot q) =
\begin{bmatrix}
0 \\
\tau_2 \\
\tau_3
\end{bmatrix}
\]

其中

\[
\bar M(q) = M(q) - \frac{1}{M_{total}}J_{cm}^T(q)J_{cm}(q)
\]

\[
\bar H(q,\dot q) = H(q,\dot q) - \frac{1}{M_{total}}J_{cm}^T(q)H_p(q,\dot q)
\]

而且腾空阶段重力项在降维形式中被完全抵消。

### 4.3 代码建议

- **环境主实现优先采用方案 A**，因为状态统一、切换方便；
- **规划与分析模块可另行使用方案 B**，因为维度更低、物理意义更清晰。

---

## 5. 低杆抓握事件

### 5.1 低杆位置

设低杆在惯性系中的位置为：

\[
p_L = \begin{bmatrix} D_x \\ D_y \end{bmatrix}
\]

### 5.2 Guard 条件

定义接触事件函数：

\[
\phi(p) = \|p - p_L\| - r_c
\]

其中 \(r_c\) 为有效抓握半径。理想点接触时可取 \(r_c = 0\)，数值仿真中可用很小正数提高鲁棒性。

从腾空阶段切换到低杆抓握阶段的严格触发条件为：

\[
\phi(p^-) = 0,
\qquad
(p^- - p_L)^T \dot p^- < 0
\]

其含义是：

- 手爪已到达接触面；
- 且碰撞前速度朝向接触面内部。

---

## 6. 抓握冲击与状态重置

### 6.1 冲击模型

碰撞时忽略重力、科里奥利项和有界驱动力矩在无穷小时间内的积分，仅保留惯性项与接触冲量。

设接触冲量为：

\[
\Lambda = \begin{bmatrix} \Lambda_x \\ \Lambda_y \end{bmatrix}
\]

广义动量守恒写成：

\[
M_f(q^-)\left(\dot q_f^+ - \dot q_f^-\right) = J_c^T \Lambda
\]

其中

\[
M_f(q)=
\begin{bmatrix}
M_{total}I_{2\times2} & J_{cm}(q) \\
J_{cm}^T(q) & M(q)
\end{bmatrix}
\]

而接触雅可比在当前建模下可直接写为：

\[
J_c =
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0
\end{bmatrix}
\]

### 6.2 碰撞后速度约束

理想抓握假设下：

\[
\dot p^+ = 0
\]

### 6.3 姿态角速度跃变

由分块展开可得：

\[
\dot q^+ = \dot q^- + M^{-1}(q^-) J_{cm}^T(q^-) \dot p^-
\]

该式是后续代码中 `reset_map_flight_to_low_bar(...)` 的核心。

### 6.4 接触冲量闭式表达

\[
\Lambda = -\left[M_{total}I_{2\times2} - J_{cm}(q^-)M^{-1}(q^-)J_{cm}^T(q^-)\right] \dot p^-
\]

可用于：

- 冲击强度评估；
- 奖励函数惩罚项；
- 终止条件设计；
- 抓握是否过载的判定。

### 6.5 Reset Map

腾空阶段状态：

\[
X_f = \begin{bmatrix} p^T & q^T & \dot p^T & \dot q^T \end{bmatrix}^T
\]

低杆阶段状态：

\[
X_{PAA} = \begin{bmatrix} q^T & \dot q^T \end{bmatrix}^T
\]

状态重置映射：

\[
X_{PAA}^+ = \Delta(X_f^-)
\]

其中

\[
q^+ = q^-
\]

\[
\dot q^+ = \dot q^- + M^{-1}(q^-) J_{cm}^T(q^-) \dot p^-
\]

也就是说，抓住低杆后：

- 丢弃平移自由度 \((x_0,y_0)\) 及其速度；
- 保留碰撞前姿态；
- 用跃变后的角速度作为低杆阶段初值。

---

## 7. 抓稳成功判定

原稿中抓稳成功不是单纯“碰到低杆”，而是一个二值判定。

### 7.1 预抓稳条件

定义可抓取姿态窗口集合：

\[
\mathcal{Q}_{\mathrm{catch}} =
\left\{
q^- \in \mathbb{R}^3 \mid
q_i^{\min} \le q_i^- \le q_i^{\max},\ i=1,2,3
\right\}
\]

预抓稳判定：

\[
\chi_{\mathrm{pre}}=
\begin{cases}
1, & \phi(p^-)=0,\ (p^- - p_L)^T\dot p^- < 0,\ q^-\in\mathcal{Q}_{\mathrm{catch}} \\
0, & \text{否则}
\end{cases}
\]

### 7.2 冲击不过载条件

定义冲量大小：

\[
I_\Lambda = \|\Lambda\|_2
\]

采用比例型冲量阈值：

\[
\Lambda_{\max}^{(0)} = \kappa M_{total}\|\dot p^-\|_2,
\qquad 0<\kappa<1
\]

并约束角速度跃变：

\[
|\Delta \dot q_i| \le \Delta \dot q_{i,\max},\qquad i=1,2,3
\]

定义冲击不过载判定：

\[
\chi_\Lambda=
\begin{cases}
1, & \|\Lambda\|_2\le \Lambda_{\max}^{(0)}\ \text{且}\ |\Delta\dot q_i|\le \Delta\dot q_{i,\max}} \\
0, & \text{否则}
\end{cases}
\]

### 7.3 最终抓稳判定

\[
\chi_{\mathrm{catch}} = \chi_{\mathrm{pre}} \cdot \chi_\Lambda
\]

代码上可解释为：

- 若 \(\chi_{\mathrm{catch}} = 1\)，执行 reset map 并切入 `LOW_BAR`；
- 若 \(\chi_{\mathrm{catch}} = 0\)，记为抓握失败，可直接 `done=True` 或切入 `FAIL`。

---

## 8. 高杆脱手时机

原稿将高杆最佳脱手描述为“无重状态”事件，即径向约束力过零。

### 8.1 总质心映射

系统总质心位置与速度：

\[
p_c(q) = \frac{1}{M_{total}} \begin{bmatrix} S_x(q) \\ S_y(q) \end{bmatrix}
\]

\[
v_c(q,\dot q) = \frac{1}{M_{total}} J_{cm}(q)\dot q
\]

### 8.2 高杆约束反力

\[
F_{bar}(q,\dot q,\ddot q) = J_{cm}(q)\ddot q + H_p(q,\dot q) + \begin{bmatrix}0 \\ M_{total}g\end{bmatrix}
\]

### 8.3 脱手 guard

定义脱手触发事件：

\[
g_r(q,\dot q,\ddot q) = x_c(q)F_x(q,\dot q,\ddot q) + y_c(q)F_y(q,\dot q,\ddot q) = 0
\]

在强化学习第一版环境中，可以有两种实现方式：

1. **动作触发型**：增加一个 `release_cmd` 动作，超过阈值则脱手；
2. **物理触发型**：只有在 `g_r` 过零附近才允许脱手。

建议先做动作触发型，后续再升级。

---

## 9. 对 Gym 环境实现最重要的结论

如果只保留最必要的信息，代码实现时应抓住以下 8 点：

1. 系统是三模式混杂系统：`HIGH_BAR / FLIGHT / LOW_BAR`
2. 高杆与低杆都可用固定基座 PAA 模型
3. 腾空阶段优先用 5 自由度浮动基座模型
4. `FLIGHT -> LOW_BAR` 的事件由 `phi(p)=0` 和入射速度方向共同决定
5. 低杆抓握后需要执行 `reset map`
6. 冲量和角速度跃变可直接作为抓稳判定与奖励项
7. 抓稳成功不是“接触”而是 `chi_catch = 1`
8. 高杆脱手第一版可先做成动作控制，后续再加入径向力过零 guard

---

## 10. 建议的代码分层

建议将环境代码拆分为：

- `dynamics_high_bar.py`
- `dynamics_flight.py`
- `dynamics_low_bar.py`
- `events.py`
- `reset_maps.py`
- `params.py`
- `three_link_env.py`

这样可以把：

- 连续动力学
- 模式切换条件
- 状态跳变映射
- 参数管理

彻底分离，方便后续让 Codex 分步实现。

---

## 11. 实现优先级建议

第一阶段只做：

- 固定维度 observation
- 三个 mode
- 高杆简化动力学
- 腾空简化动力学
- 低杆接触检测
- reset map
- 成功 / 失败终止逻辑

第二阶段再补：

- 原稿中的完整 `M/H/G` 显式公式
- `J_cm(q)` 和 `H_p(q,dq)` 的严格实现
- 抓稳阈值的工程化标定
- 脱手 guard 的物理判定
- 轨迹规划或 OCP 约束

---

## 12. 一句话总结

这个项目不是单纯的摆杆问题，而是一个**三连杆欠驱动 + 腾空浮动基座 + 低杆冲击抓握 + 混杂切换**的 Gym 环境。最稳妥的路线不是一次写满全部物理细节，而是先把**模式、事件、重置、终止和奖励**搭起来，再逐步把原稿中的严格公式替换进去。
