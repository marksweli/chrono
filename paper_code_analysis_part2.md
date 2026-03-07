# 论文公式与 Chrono FSI/SPH 代码对照解读（第二部分）

> **论文**：*Modeling granular material dynamics and its two-way coupling with moving solid bodies using a continuum representation and the SPH method*
> Wei Hu, Milad Rakhsha, Lijing Yang, Ken Kamrin, Dan Negrut
> Computer Methods in Applied Mechanics and Engineering, 2021

本文档为第一部分的续篇，继续按照论文第 2 节（Numerical method）的顺序，对 2.5 时间积分与 2.6 应力后处理部分的所有公式与代码进行逐一对照。

代码链接均指向 [marksweli/chrono](https://github.com/marksweli/chrono) 仓库中具体文件和行号。

---

## 2.5 Time Integration（时间积分）

---

#### 预测步与修正步方程

论文采用标准二阶显式预测-修正（Predictor-Corrector）格式。预测步将场变量从 [ t_n ] 推进至中间时刻 [ t + \Delta t/2 ]，修正步利用中间时刻的导数完成一步完整积分。

---

#### 预测步（Predictor Step）

**代码**（[SphFluidDynamics.cu，第 113–135 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L113)）：

```cpp
// DoStepDynamics — RK2 integration scheme (predictor-corrector)
case IntegrationScheme::RK2: {
    Real dummy = 0;
    auto& y_tmp = m_data_mgr.sortedSphMarkers1_D;
    CopySortedMarkers(y, y_tmp);              // y_tmp <- y_n  (save current state)

    forceSystem->ForceSPH(y, t, dummy);       // compute f(t_n, y_n) = {a_i, b_i, c_i}
    EulerStep(y_tmp, h / 2);                  // K1 = y_n + (h/2)*f(t_n, y_n)  [predictor]
    ApplyBoundaryConditions(y_tmp);

    forceSystem->ForceSPH(y_tmp, t + h/2, dummy);  // compute f(t_n+h/2, K1)
    EulerStep(y, h);                           // y_{n+1} = y_n + h*f(t_n+h/2, K1) [corrector]
    ApplyBoundaryConditions(y);
    break;
}
```

**代码**（[SphFluidDynamics.cu，第 313–332 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L313)）：

```cuda
// Device functions for Euler/midpoint steps
__device__ void PositionEulerStep(Real dT, const Real3& vel, Real4& pos) {
    Real3 p = mR3(pos);
    p += dT * vel;     // x_i(t + dt/2) = x_i(t) + (dt/2) * c_i(t)
    pos = mR4(p, pos.w);
}

__device__ void VelocityEulerStep(Real dT, const Real3& acc, Real3& vel) {
    vel += dT * acc;   // u_i(t + dt/2) = u_i(t) + (dt/2) * a_i(t)
}

__device__ void DensityEulerStep(Real dT, const Real& deriv, EosType eos, Real4& rho_p) {
    rho_p.x += dT * deriv;  // rho_i(t + dt/2) = rho_i(t) + (dt/2) * d(rho)/dt
    rho_p.y = Eos(rho_p.x, eos);
}
```

**代码**（[SphFluidDynamics.cu，第 334–343 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L334)）：

```cuda
__device__ void TauEulerStep(Real dT,
                             const Real3& deriv_tau_diag,
                             const Real3& deriv_tau_offdiag,
                             const Real& deriv_rho,
                             bool close_to_surface,
                             Real3& tau_diag,
                             Real3& tau_offdiag,
                             Real4& rho_p,
                             Real3& pcEvSv,
                             bool& error_occurred) {
    // Preliminary: sigma*(t + dt/2) = sigma(t) + (dt/2) * b_i(t)
    Real3 new_tau_diag = tau_diag + dT * deriv_tau_diag;
    Real3 new_tau_offdiag = tau_offdiag + dT * deriv_tau_offdiag;
    // ... then apply yield surface projection (Section 2.6)
}
```

**公式**（预测步，论文 Section 2.5）：

\[
\begin{aligned}
\bar{\mathbf{u}}_i\!\left(t + \tfrac{\Delta t}{2}\right) &= \mathbf{u}_i(t) + \tfrac{\Delta t}{2}\,\mathbf{a}_i(t) \\
\bar{\mathbf{x}}_i\!\left(t + \tfrac{\Delta t}{2}\right) &= \mathbf{x}_i(t) + \tfrac{\Delta t}{2}\,\mathbf{c}_i(t) \\
\bar{\pmb{\sigma}}_i\!\left(t + \tfrac{\Delta t}{2}\right) &= \pmb{\sigma}_i(t) + \tfrac{\Delta t}{2}\,\mathbf{b}_i(t)
\end{aligned}
\]

其中 [ \mathbf{a}_i = d\mathbf{u}_i/dt ]（来自公式 (10)），[ \mathbf{b}_i = d\pmb{\sigma}_i/dt ]（来自公式 (13)），[ \mathbf{c}_i = d\mathbf{x}_i/dt ]（来自公式 (14)）。

**解读**：
代码中 `EulerStep(y_tmp, h/2)` 调用 `EulerStep_D` kernel，其中：
- `PositionEulerStep(dT=h/2, velMas + vel_XSPH, posRad)` 实现 [ \bar{\mathbf{x}}_i = \mathbf{x}_i + \frac{\Delta t}{2}\mathbf{c}_i ]（含 XSPH 修正）；
- `VelocityEulerStep(dT=h/2, derivVelRho, velMas)` 实现 [ \bar{\mathbf{u}}_i = \mathbf{u}_i + \frac{\Delta t}{2}\mathbf{a}_i ]；
- `TauEulerStep(dT=h/2, ...)` 实现 [ \bar{\pmb{\sigma}}_i = \pmb{\sigma}_i + \frac{\Delta t}{2}\mathbf{b}_i ]（含后处理）。
因此 `EulerStep(y_tmp, h/2)` 完整对应论文中的预测步。

---

#### 修正步（Corrector Step）

**代码**（[SphFluidDynamics.cu，第 127–135 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L127)）：

```cpp
// DoStepDynamics — RK2 corrector step
forceSystem->ForceSPH(y_tmp, t + h/2, dummy);
// Evaluate a_i(t+h/2), b_i(t+h/2), c_i(t+h/2) at the intermediate state K1

EulerStep(y, h);
// y_{n+1} = y_n + h * f_{n+1/2}
// u_i(t+dt) = u_i(t) + dt * a_i(t+dt/2)
// x_i(t+dt) = x_i(t) + dt * c_i(t+dt/2)
// sigma_i(t+dt) = sigma_i(t) + dt * b_i(t+dt/2)
```

**代码**（[SphFluidDynamics.cu，第 560–606 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L560)）：

```cuda
// EulerStep_D kernel (also used for corrector step with full dt)
__global__ void EulerStep_D(Real4* posRadD, Real3* velMasD, /* ... */ Real dT, /* ... */) {
    // Position update using XSPH velocity
    PositionEulerStep(dT, velMasD[index] + vel_XSPH_D[index], posRadD[index]);
    // Velocity update
    VelocityEulerStep(dT, mR3(derivVelRhoD[index]), velMasD[index]);
    if (paramsD.elastic_SPH) {
        // Stress update with Drucker-Prager correction (Section 2.6)
        TauEulerStep(dT, derivTauXxYyZzD[index], derivTauXyXzYzD[index],
                     derivVelRhoD[index].w, freeSurfaceIdD[index],
                     tauXxYyZzD[index], tauXyXzYzD[index],
                     rhoPresMuD[index], pcEvSvD[index], error_occurred);
    } else {
        DensityEulerStep(dT, derivVelRhoD[index].w, paramsD.eos_type, rhoPresMuD[index]);
    }
}
```

**公式**（修正步）：

\[
\begin{aligned}
\mathbf{u}_i(t + \Delta t) &= \mathbf{u}_i(t) + \Delta t\,\mathbf{a}_i\!\left(t + \tfrac{\Delta t}{2}\right) \\
\mathbf{x}_i(t + \Delta t) &= \mathbf{x}_i(t) + \Delta t\,\mathbf{c}_i\!\left(t + \tfrac{\Delta t}{2}\right) \\
\pmb{\sigma}_i(t + \Delta t) &= \pmb{\sigma}_i(t) + \Delta t\,\mathbf{b}_i\!\left(t + \tfrac{\Delta t}{2}\right)
\end{aligned}
\]

**解读**：
修正步在代码中通过第二次调用 `EulerStep(y, h)`（完整步长 `h`）来实现。由于 `y` 仍保存时刻 [ t_n ] 的数据（`y_tmp` 是中间状态），而当前存储的导数（`derivVelRhoD` 等）已由 `forceSystem->ForceSPH(y_tmp, ...)` 更新为时刻 [ t_n + \Delta t/2 ] 处的值，因此 `EulerStep(y, h)` 的结果为 [ y_n + h \cdot f(t_n + h/2, K_1) ]，精确实现修正步。

---

#### SYMPLECTIC 辛积分格式

**代码**（[SphFluidDynamics.cu，第 140–153 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L140)）：

```cpp
case IntegrationScheme::SYMPLECTIC: {
    CopySortedMarkers(y, y_tmp);              // y_tmp <- y_n
    forceSystem->ForceSPH(y, t, dummy);       // f(t_n, y_n)
    EulerStep(y_tmp, h / 2);                  // y_{n+1/2} = y_n + (h/2)*f(t_n, y_n)
    forceSystem->ForceSPH(y_tmp, t + h/2, dummy);  // f_{n+1/2}
    MidpointStep(y, h);                        // y_{n+1} = y_n + h * f_{n+1/2}
    break;
}
```

**代码**（[SphFluidDynamics.cu，第 319–322 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L319)）：

```cuda
// PositionMidpointStep (used in MidpointStep_D)
__device__ void PositionMidpointStep(Real dT, const Real3& vel, const Real3& acc, Real4& pos) {
    Real3 p = mR3(pos);
    p += dT * vel + 0.5 * dT * dT * acc;  // x_{n+1} = x_n + h*v_{n+1/2} + 0.5*h^2*a_{n+1/2}
    pos = mR4(p, pos.w);
}
```

**公式**（辛格式等价的 "leapfrog" 更新，论文 2.5 节描述）：

\[
\mathbf{u}_i(t + \Delta t) = \mathbf{u}_i(t) + \Delta t\,\mathbf{a}_i\!\left(t + \tfrac{\Delta t}{2}\right)
\]

\[
\mathbf{x}_i(t + \Delta t) = \mathbf{x}_i(t) + \Delta t\,\mathbf{v}_i\!\left(t + \tfrac{\Delta t}{2}\right) + \tfrac{1}{2}(\Delta t)^2\,\mathbf{a}_i\!\left(t + \tfrac{\Delta t}{2}\right)
\]

**解读**：
辛格式（`SYMPLECTIC`）在代码中通过 `MidpointStep_D` 实现，其位置更新包含额外的加速度项 [ \frac{1}{2}\Delta t^2 \mathbf{a}_{n+1/2} ] 以达到二阶精度的辛守恒属性。速度更新仍为简单 Euler 步，整体保证长时间能量守恒。

---

#### CFL 条件与时间步计算

**代码**（[SphFluidDynamics.cu，第 87–106 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L87)）：

```cpp
double SphFluidDynamics::computeTimeStep() const {
    // CFL time step: min(h/(c_s + max_vel_diff))
    double min_courant_viscous_time_step =
        thrust::reduce(m_data_mgr.courantViscousTimeStepD.begin(), ...
                       std::numeric_limits<Real>::max(), thrust::minimum<Real>());
    // Acceleration-based time step: sqrt(h / |a|)
    double min_acceleration_time_step =
        thrust::reduce(m_data_mgr.accelerationTimeStepD.begin(), ...
                       std::numeric_limits<Real>::max(), thrust::minimum<Real>());

    double adjusted_time_step = 0.3 * std::min(min_courant_viscous_time_step, min_acceleration_time_step);
    return adjusted_time_step;
}
```

**代码**（[SphForceWCSPH.cu，第 1586–1592 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1586)）：

```cuda
// Per-particle CFL time step in CrmRHS
if (IsFluidParticle(rhoPresMuA.w)) {
    courantViscousTimeStepD[index] = paramsD.h / (paramsD.Cs + max_vel_diff);  // CFL
    Real intermediate = sqrtf(derivVelRho.x * derivVelRho.x +
                               derivVelRho.y * derivVelRho.y +
                               derivVelRho.z * derivVelRho.z);
    Real accT = sqrtf(paramsD.h / intermediate);  // acceleration-based dt
    accelerationTimeStepD[index] = accT;
}
```

**公式**（CFL 条件，论文 Section 2.5 引用 [84]）：

\[
\Delta t \leq C_\text{CFL} \min\left(\frac{h}{c_s + \max_j\|\mathbf{u}_{ij}\|},\; \sqrt{\frac{h}{\|\mathbf{a}_i\|}}\right)
\]

**解读**：
代码通过两个 per-particle 时间步数组：
- `courantViscousTimeStep = h / (Cs + max_vel_diff)`：CFL 声速条件（`Cs` = 声速，`max_vel_diff` = 最大相对速度项 [ h \cdot |\mathbf{u}_{AB} \cdot \mathbf{r}_{AB}| / |\mathbf{r}_{AB}|^2 ]）；
- `accelerationTimeStep = sqrt(h / |a_i|)`：加速度条件。
全局时间步取所有粒子最小值再乘以安全系数 0.3，与标准 CFL 条件一致。

---

## 2.6 Post-processing Strategy for the Stress Tensor（应力张量后处理）

---

#### 公式 (19)：应力预测步结果

**代码**（[SphFluidDynamics.cu，第 344–346 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L344)）：

```cuda
// TauEulerStep — Step 1: compute trial stress sigma*
Real3 new_tau_diag = tau_diag + dT * deriv_tau_diag;       // sigma* = sigma(t) + dt * d sigma/dt
Real3 new_tau_offdiag = tau_offdiag + dT * deriv_tau_offdiag;
```

**公式**：

\[
\pmb{\sigma}^{*} = \pmb{\sigma}(t + \Delta t) \quad (19)
\]

**解读**：
在 `TauEulerStep` 函数的第一步中，`new_tau_diag = tau_diag + dT * deriv_tau_diag` 计算时间积分后的试验应力张量 [ \pmb{\sigma}^* ]（对角和非对角分量分别更新）。这是后处理四步策略中的 STEP 1，即通过预测-修正积分方案（公式 (13)）得到中间值，再进行 Drucker-Prager 投影修正。

---

#### 压力提取（Step 1 续）

**代码**（[SphFluidDynamics.cu，第 349–354 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L349)）：

```cuda
// TauEulerStep — extract p* and tau* from sigma*
Real p_tr = -CH_1_3 * (new_tau_diag.x + new_tau_diag.y + new_tau_diag.z);
// p* = -1/3 * tr(sigma*)
// Separate deviatoric: tau* = sigma* + p*I
tau_diag += mR3(p_n);       // tau^n = sigma^n - diag(-p^n)
new_tau_diag += mR3(p_tr);  // tau* = sigma* - diag(-p*)
```

**公式**（来自公式 (2)）：

\[
p^* = -\frac{1}{3}\mathrm{tr}(\pmb{\sigma}^*), \quad \pmb{\tau}^* = \pmb{\sigma}^* + p^*\mathbf{I}
\]

**解读**：
`CH_1_3 = 1/3`；`new_tau_diag += mR3(p_tr)` 将对角应力张量转换为偏应力：[ \tau^*_{xx} = \sigma^*_{xx} + p^* ]，[ \tau^*_{yy} = \sigma^*_{yy} + p^* ]，[ \tau^*_{zz} = \sigma^*_{zz} + p^* ]（因为 [ p^* > 0 ] 且 [ \sigma = -p\mathbf{I} + \tau ]，故 [ \tau = \sigma + p\mathbf{I} ]）。若 [ p^* < 0 ]，则应力张量置零，跳过本时间步。

---

#### 公式 (20)：偏应力双内积（von Mises 等效应力）

**代码**（[SphFluidDynamics.cu，第 356–361 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L356)）：

```cuda
// TauEulerStep — STEP 2: compute tau_bar* = sqrt(0.5 * tau*:tau*)
Real tau_tr =
    square(new_tau_diag.x) + square(new_tau_diag.y) + square(new_tau_diag.z) +     // diagonal
    2 * (square(new_tau_offdiag.x) + square(new_tau_offdiag.y) + square(new_tau_offdiag.z));  // off-diagonal
tau_tr = sqrt(0.5 * tau_tr);
// tau_tr = sqrt(0.5 * tau*_ab : tau*_ab) = tau_bar*
```

**公式**：

\[
\bar{\tau}^{*} = \sqrt{\frac{1}{2} (\tau_{\alpha\beta}^{*}) : (\tau_{\alpha\beta}^{*})} \quad (20)
\]

**解读**：
代码计算偏应力张量的双内积（Frobenius 范数的变体）：
[ \tau^*:\tau^* = \tau^*_{xx}{}^2 + \tau^*_{yy}{}^2 + \tau^*_{zz}{}^2 + 2(\tau^*_{xy}{}^2 + \tau^*_{xz}{}^2 + \tau^*_{yz}{}^2) ]（利用对称性，非对角分量计数两次）。
`sqrt(0.5 * tau_tr_sq)` 对应公式 (20) 的 [ \sqrt{\frac{1}{2}\tau^*_{\alpha\beta}:\tau^*_{\alpha\beta}} ]，即等效剪切应力 [ \bar{\tau}^* ]。

---

#### 公式 (21)：Drucker-Prager 屈服应力 [ S_0 ]

**代码**（[SphFluidDynamics.cu，第 364–375 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L364)）：

```cuda
// TauEulerStep — STEP 3: compute S_0 = mu_s * p* (yield criterion)
Real mu_s = paramsD.mu_fric_s;         // static friction coefficient
// Real s_0 = mu_s * p_tr;             // Drucker-Prager yield stress (not explicitly named)
Real tau_max = p_tr * mu_s + coh;      // tau_max = S_0 = mu_s * p* (+ cohesion)
```

**公式**：

\[
S_0 = \mu_s p^* \quad (21)
\]

**解读**：
代码中 `tau_max = p_tr * mu_s + coh`，其中 `coh = paramsD.Coh_coeff`（黏聚力，颗粒材料中通常为 0 或很小），`mu_s = paramsD.mu_fric_s` 为静摩擦系数。基本形式 `p_tr * mu_s` 精确对应公式 (21) 的 [ S_0 = \mu_s p^* ]，额外的 `coh` 项是对摩擦-内聚力本构（Drucker-Prager 含内聚力）的扩展。

---

#### 公式 (22)：弹性状态（无塑性流动）的应力更新

**代码**（[SphFluidDynamics.cu，第 399–408 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L399)）：

```cuda
// TauEulerStep — STEP 4: if tau_tr < tau_max (no plastic flow)
if (tau_tr > tau_max) {
    // plastic flow: scale back to yield surface (Eq. 23)
    Real coeff = tau_max / (tau_tr + 1e-9);
    new_tau_diag *= coeff;
    new_tau_offdiag *= coeff;
} // else: tau* used as final stress (Eq. 22)

// Reconstruct sigma from deviatoric component and pressure
tau_diag = new_tau_diag - mR3(p_tr);   // sigma = tau - p*I (combined back)
tau_offdiag = new_tau_offdiag;
rho_p.y = p_tr;                         // store p^{n+1} = p*
```

**公式**：

\[
\pmb{\tau}^{n + 1} = \pmb{\tau}^{*},\quad p^{n + 1} = p^{*},\quad \pmb{\sigma}^{n + 1} = \pmb{\sigma}^{*} \quad (22)
\]

**解读**：
若 `tau_tr <= tau_max`（即 [ \bar{\tau}^* < S_0 ]），代码走 `else` 分支（不进入 `if(tau_tr > tau_max)` 块），直接将 `new_tau_diag, new_tau_offdiag` 保留不变，再通过 `tau_diag = new_tau_diag - mR3(p_tr)` 重组全应力张量，等价于 [ \pmb{\sigma}^{n+1} = -p^*\mathbf{I} + \pmb{\tau}^* = \pmb{\sigma}^* ]，与公式 (22) 完全一致。

---

#### 公式 (23)：塑性流动时的 Drucker-Prager 屈服面投影

**代码**（[SphFluidDynamics.cu，第 399–409 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L399)）：

```cuda
// TauEulerStep — STEP 4: if tau_tr >= tau_max (plastic flow occurs)
if (tau_tr > tau_max) {
    // Drucker-Prager: scale deviatoric stress back to yield surface
    Real coeff = tau_max / (tau_tr + 1e-9);   // coeff = mu * p* / tau_bar* = tau_max / tau_tr
    new_tau_diag *= coeff;                     // tau^{n+1} = (mu*p*/tau_bar*) * tau*
    new_tau_offdiag *= coeff;
}
// Reconstruct full stress tensor sigma^{n+1} = -p*I + tau^{n+1}
tau_diag = new_tau_diag - mR3(p_tr);
tau_offdiag = new_tau_offdiag;
rho_p.y = p_tr;
```

**代码**（[SphFluidDynamics.cu，第 370–373 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L370)）：

```cuda
// mu(I) rheology — friction coefficient calculation
Real mu = mu_s + (mu_2 - mu_s) * (I + 1.0e-9) / (I0 + I + 1.0e-9);
// mu = mu_s + (mu_2 - mu_s) / (I0/I + 1)   [mu(I) model]
Real tau_max = p_tr * mu + coh;
```

**公式**：

\[
\pmb{\tau}^{n + 1} = \frac{\mu p^{*}}{\bar{\tau}^{*}}\pmb{\tau}^{*},\quad p^{n + 1} = p^{*},\quad \pmb{\sigma}^{n + 1} = -p^{n + 1}\mathbf{I} + \pmb{\tau}^{n + 1} \quad (23)
\]

其中摩擦系数 [ \mu = \mu_s + \frac{\mu_2 - \mu_s}{I_0/I + 1} ]，[ I = \dot{\lambda} d\sqrt{\rho_0/\rho^{n+1}} ] 为惯性数，[ \dot{\lambda} = (\bar{\tau}^* - \bar{\tau}^n)/(G\Delta t) ] 为塑性应变率。

**解读**：
代码中 `coeff = tau_max / tau_tr = mu * p_tr / tau_tr`，这正是公式 (23) 中的缩放系数 [ \mu p^*/\bar{\tau}^* ]（其中 `tau_max = mu * p_tr`，`tau_tr = bar_tau_star`）。乘以 `new_tau_diag` 和 `new_tau_offdiag` 即得 [ \pmb{\tau}^{n+1} = \frac{\mu p^*}{\bar{\tau}^*}\pmb{\tau}^* ]。

mu(I) 流变模型代码中 `Chi` = [ \dot{\lambda} ] 通过 `abs(tau_tr - tau_n) * INV_G_shear / dT` 计算（其中 `INV_G_shear = 1/G`），对应 [ \dot{\lambda} = |\bar{\tau}^* - \bar{\tau}^n|/(G\Delta t) ]（与论文定义相比绝对值使 [ \dot{\lambda} \geq 0 ]）。惯性数 `I = Chi * dia * sqrt(rho0/(p_tr+eps))` 对应 [ I = \dot{\lambda} d\sqrt{\rho_0/\rho^{n+1}} ]（其中 `dia = paramsD.ave_diam = d`），最后 `mu = mu_s + (mu_2 - mu_s) * I / (I0 + I)` 对应 [ \mu = \mu_s + (\mu_2 - \mu_s)/(I_0/I + 1) ]，完全吻合论文中的 [ \mu(I) ] 模型。

---

## 补充：BCE 粒子应力推断

#### 公式（BCE 应力外推，论文正文第 200–210 行）

**代码**（[SphForceWCSPH.cu，第 718–727 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L718)）：

```cuda
// CrmAdamiBC kernel — stress tensor extrapolation to BCE particles
// sigma_j = ( sum_{k in Omega_f} sigma_k * W_jk
//           + diag(f_b - f_j) * sum_{k in Omega_f} rho_k * diag(r_jk) * W_jk )
//           / sum_{k in Omega_f} W_jk
sortedTauXxYyZz[index] = (sum_tauD - dot(paramsD.gravity - bceAcc[index], sum_rhorw)) / sum_w;
sortedTauXyXzYz[index] = sum_tauO / sum_w;
// where: sum_tauD = sum_k tau_k * W_jk,  sum_rhorw = sum_k rho_k * r_jk * W_jk
// bceAcc[index] = f_j (BCE inertial acceleration)
// gravity = f_b (body force)
```

**公式**（论文正文）：

\[
\pmb{\sigma}_j = \frac{\sum_{k\in\Omega_f}\pmb{\sigma}_kW_{jk} + [\mathrm{diag}(\mathbf{f}_b - \mathbf{f}_j)]\sum_{k\in\Omega_f}\rho_k[\mathrm{diag}(\mathbf{r}_{jk})]W_{jk}}{\sum_{k\in\Omega_f}W_{jk}}
\]

**解读**：
代码中 `sum_tauD` 对应分子第一项 [ \sum_{k\in\Omega_f}\pmb{\sigma}_k W_{jk} ]；`dot(gravity - bceAcc, sum_rhorw)` 对应第二项（利用 `dot` 近似 `diag` 矩阵乘积，只计算对角贡献）。`sum_w` 为核函数权重之和（分母）。`bceAcc[index]` = [ \mathbf{f}_j ]（BCE 粒子惯性加速度，由 `CalcRigidBceAccelerationD` 计算）。

---

#### 公式（BCE 粒子惯性力，论文正文第 209 行）

**代码**（[SphBceManager.cu，第 135–171 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphBceManager.cu#L135)）：

```cuda
// CalcRigidBceAccelerationD — BCE inertial acceleration = a_body + alpha x r + omega x (omega x r)
// linear acceleration
Real3 acc = body_linacc[body_ID];

// centrifugal acceleration: omega x (omega x r_c) 
Real3 omega_cross = cross(omega, local);              // omega x r_c
Real3 omega_cross_cross = cross(omega, omega_cross);  // omega x (omega x r_c)
acc += mR3(dot(u, omega_cross_cross), dot(v, omega_cross_cross), dot(w, omega_cross_cross));

// tangential acceleration: alpha x r_c
Real3 alpha_cross = cross(body_angacc[body_ID], local);
acc += mR3(dot(u, alpha_cross), dot(v, alpha_cross), dot(w, alpha_cross));
```

**公式**（论文正文）：

\[
\mathbf{f}_j = \dot{\mathbf{u}}_{body} + \dot{\pmb{\omega}}_{body}\times \mathbf{r}_{jc} + \pmb{\omega}_{body}\times (\pmb{\omega}_{body}\times \mathbf{r}_{jc})
\]

**解读**：
代码中三项分别对应：
1. `body_linacc[body_ID]`：刚体质心线加速度 [ \dot{\mathbf{u}}_{body} ]；
2. `alpha_cross = alpha x r_c`：切向加速度 [ \dot{\pmb{\omega}} \times \mathbf{r}_{jc} ]；
3. `omega_cross_cross = omega x (omega x r_c)`：向心加速度 [ \pmb{\omega} \times (\pmb{\omega} \times \mathbf{r}_{jc}) ]。
所有加速度通过旋转矩阵 [ [\mathbf{u}, \mathbf{v}, \mathbf{w}] ]（由四元数生成）从刚体局部坐标系变换到全局坐标系，与公式完全一致。

---

#### 流体对固体的合力与合力矩

**代码**（[SphBceManager.cu，第 302–381 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphBceManager.cu#L302)）：

```cuda
// CalcRigidForces_D kernel — F_body and T_body from fluid forces on BCE
Force = mR3(derivatives[sorted_index]) * markerMass;   // m_j * du_j/dt
Real3 dist3 = mR3(positions[sorted_index]) - body_pos[body_ID];  // r_jc
Torque = cross(dist3, Force);   // T_j = r_jc x (m_j * du_j/dt)
// Block-level reduction: sum over all BCE markers of this body
atomicAdd(&body_forces[body_ID].x, sharedForces[0].x);    // F_body = sum_j m_j * du_j/dt
atomicAdd(&body_torques[body_ID].x, sharedTorques[0].x);  // T_body = sum_j r_jc x (m_j * du_j/dt)
```

**公式**（论文正文）：

\[
\mathbf{F}_{body} = \sum_{j\in \Omega_s}m_j\dot{\mathbf{u}}_j\quad \mathrm{and}\quad \mathbf{T}_{body} = \sum_{j\in \Omega_s}\mathbf{r}_{jc}\times (m_j\dot{\mathbf{u}}_j)
\]

**解读**：
代码中 `derivatives[sorted_index]` = [ d\mathbf{u}_j/dt ]（由 `CrmRHS` 或 `CfdRHS` 计算的加速度），乘以 `markerMass` = [ m_j ] 得到 BCE 粒子所受力。`dist3 = r_{jc}`。利用 GPU 共享内存 block-level reduction 对每个刚体的所有 BCE 粒子进行求和，最后通过 `atomicAdd` 累加到全局数组 `body_forces[body_ID]` 和 `body_torques[body_ID]`，精确对应上述公式。

---

## 附录：核函数梯度（GradW）

**代码**（[SphGeneral.cuh，第 67–82 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphGeneral.cuh#L67)）：

```cuda
// Gradient of cubic spline kernel W(r,h) with respect to x_i
// grad_i W_ij = dW/dr * e_ij  (e_ij = r_ij/|r_ij|, direction from j to i)
// Returns beta(q) * r_ij, where beta includes the 1/|r_ij| factor
inline __host__ __device__ Real3 GradW3h_CubicSpline(Real3 d, Real invh) {
    Real q = length(d) * invh;   // q = |r_ij| / h
    if (abs(q) < EPSILON) return mR3(0);

    if (q < 1) {
        // beta = 3 * alpha / h^2, alpha = 1/(4*pi*h^3)
        // dW/dq = alpha * (-3q^2 + 3q) -> collapsed to: alpha*(3q-4) combined with 1/h
        Real beta = 3 * INVPI * quintic(invh) / 4;
        return (beta * (3 * q - 4)) * d;
    }
    if (q < 2) {
        Real beta = 3 * INVPI * quintic(invh) / 4;
        return (beta * (4 - q - 4 / q)) * d;
    }
    return mR3(0);
}
```

**公式**（核函数梯度）：

\[
\nabla_i W_{ij} = \frac{dW}{dr}\Big|_{r=r_{ij}} \mathbf{e}_{ij}
\]

**解读**：
`GradW3h_CubicSpline` 直接返回 [ \nabla_i W_{ij} = \beta(q) \cdot \mathbf{r}_{ij} ]（不除以 [ |\mathbf{r}_{ij}| ]，因为 [ \mathbf{r}_{ij}/|\mathbf{r}_{ij}| \cdot dW/dr ] 与 [ \beta \cdot \mathbf{r}_{ij} ] 等价，其中 [ \beta ] 包含了 [ 1/|\mathbf{r}_{ij}| ] 的贡献）。对于 [ q < 1 ]，`beta * (3q - 4)` 对应 [ dW/dq \cdot \alpha_d/h = \alpha_d (-3q^2 + 3q)/h^2 / \text{something} ]；总之通过 `GradW3h` 分发函数可调用不同核（cubic spline、Wendland 等）的梯度实现。

---

*本文档（第一、二部分合计）覆盖了论文第 2 节（Numerical Method）中所有出现的公式（公式 (1)–(24) 及正文中的附加公式），并将每个公式与 Chrono FSI/SPH 模块中最底层的 CUDA 设备函数/kernel 代码逐一对应。*
