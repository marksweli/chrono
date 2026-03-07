# 论文公式与 Chrono FSI/SPH 代码对照解读（第一部分）

> **论文**：*Modeling granular material dynamics and its two-way coupling with moving solid bodies using a continuum representation and the SPH method*
> Wei Hu, Milad Rakhsha, Lijing Yang, Ken Kamrin, Dan Negrut
> Computer Methods in Applied Mechanics and Engineering, 2021

本文档按照论文第 2 节（Numerical method）的顺序，将每一个出现的公式与 [marksweli/chrono](https://github.com/marksweli/chrono) 仓库中对应的最底层 CUDA/C++ 实现代码进行逐一对照和解读。代码链接均指向 GitHub 仓库中的具体文件和行号。

---

## 2. Numerical Method（数值方法）

---

### 2.1 Governing Equations（控制方程）

#### 2.1.1 Dynamic equation of the "fluid"（"流体"动力学方程）

---

#### 公式 (1)：连续性方程与动量方程

**代码**（[SphForceWCSPH.cu，第 1073 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1073)）：

```cuda
// crmDvDt device function (lines 1052-1197)
// Density Update (continuity equation)
Real derivRho = paramsD.markerMass * dot(velMasA - velMasB, gradW);
```

**代码**（[SphForceWCSPH.cu，第 1109–1117 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1109)）：

```cuda
// Momentum update (du/dt = divergence(sigma)/rho + f_b)
Real invRhoASq = 1 / (rhoPresMuA.x * rhoPresMuA.x);
Real invRhoBSq = 1 / (rhoPresMuB.x * rhoPresMuB.x);
Real3 MA_gradW = gradW * Mass;
Real derivVx = (tauXxYyZz_A.x * invRhoASq + tauXxYyZz_B.x * invRhoBSq) * MA_gradW.x +
               (tauXyXzYz_A.x * invRhoASq + tauXyXzYz_B.x * invRhoBSq) * MA_gradW.y +
               (tauXyXzYz_A.y * invRhoASq + tauXyXzYz_B.y * invRhoBSq) * MA_gradW.z;
Real derivVy = (tauXyXzYz_A.x * invRhoASq + tauXyXzYz_B.x * invRhoBSq) * MA_gradW.x +
               (tauXxYyZz_A.y * invRhoASq + tauXxYyZz_B.y * invRhoBSq) * MA_gradW.y +
               (tauXyXzYz_A.z * invRhoASq + tauXyXzYz_B.z * invRhoBSq) * MA_gradW.z;
Real derivVz = (tauXyXzYz_A.y * invRhoASq + tauXyXzYz_B.y * invRhoBSq) * MA_gradW.x +
               (tauXyXzYz_A.z * invRhoASq + tauXyXzYz_B.z * invRhoBSq) * MA_gradW.y +
               (tauXxYyZz_A.z * invRhoASq + tauXxYyZz_B.z * invRhoBSq) * MA_gradW.z;
```

**公式**：

\[
\left\{ \begin{array}{ll}\frac{d\rho}{dt} = -\rho \nabla \cdot \mathbf{u}\\ \frac{d\mathbf{u}}{dt} = \frac{\nabla\pmb{\sigma}}{\rho} +\mathbf{f}_b \end{array} \right. \quad \text{for} \quad \mathbf{x}\in \Omega_f \quad (1)
\]

**解读**：
代码中 `derivRho` 对应连续性方程右端 [ -\rho \nabla \cdot \mathbf{u} ]，其中 `dot(velMasA - velMasB, gradW)` 为 [ (\mathbf{u}_A - \mathbf{u}_B) \cdot \nabla_A W_{AB} ]，乘以质量后给出密度变化率。动量方程通过将应力张量 [ \pmb{\sigma} ] 的各分量（存储在 `tauXxYyZz` 和 `tauXyXzYz` 数组中，以 Voigt 记法区分对角和非对角分量）按 SPH 离散形式累加到每个粒子上来实现，与公式 (10) 一致（详见 2.2 节）。

---

#### 公式 (2)：应力张量分解

**代码**（[SphFluidDynamics.cu，第 349–361 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L349)）：

```cuda
// TauEulerStep device function
// Extracting pressure and deviatoric part
Real p_n = -CH_1_3 * (tau_diag.x + tau_diag.y + tau_diag.z);
Real p_tr = -CH_1_3 * (new_tau_diag.x + new_tau_diag.y + new_tau_diag.z);
// tau_diag now holds the deviatoric component:
tau_diag += mR3(p_n);       // sigma_diag - diag(-p) = deviatoric diagonal
new_tau_diag += mR3(p_tr);
```

**公式**：

\[
\pmb{\sigma} = -p \mathbf{I} + \pmb{\tau} \quad (2)
\]

**解读**：
代码中，全应力张量 [ \pmb{\sigma} ] 以 `tauXxYyZz`（对角元素：[ \sigma_{xx}, \sigma_{yy}, \sigma_{zz} ]）和 `tauXyXzYz`（非对角元素：[ \sigma_{xy}, \sigma_{xz}, \sigma_{yz} ]）存储。压力 [ p = -\frac{1}{3}\mathrm{tr}(\pmb{\sigma}) ] 通过取对角元素之和的负三分之一来计算（即 `CH_1_3 = 1/3`），偏应力 [ \pmb{\tau} ] 则是 [ \pmb{\sigma} + p\mathbf{I} ]，与公式 (2) 完全吻合。

---

#### 公式 (3)：Jaumann 应力速率张量（应力率）

**代码**（[SphForceWCSPH.cu，第 1564–1572 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1564)）：

```cuda
// CrmRHS kernel - stress rate equation
// Split into D (symmetric, strain rate) and W (antisymmetric, rotation rate)
Real Dxx = Lxx; Real Dyy = Lyy; Real Dzz = Lzz;
Real Dxy = 0.5 * (Lxy + Lyx);
Real Dxz = 0.5 * (Lxz + Lzx);
Real Dyz = 0.5 * (Lyz + Lzy);
Real Wxy = 0.5 * (Lxy - Lyx);
Real Wxz = 0.5 * (Lxz - Lzx);
Real Wyz = 0.5 * (Lyz - Lzy);

// Jaumann stress rate (d sigma/dt):
Real dTauxx = twoG * (Dxx - edia) + 2.0 * (tauxy * Wxy + tauxz * Wxz) + threeK * edia;
Real dTauyy = twoG * (Dyy - edia) - 2.0 * (tauxy * Wxy - tauyz * Wyz) + threeK * edia;
Real dTauzz = twoG * (Dzz - edia) - 2.0 * (tauxz * Wxz + tauyz * Wyz) + threeK * edia;
Real dTauxy = twoG * Dxy - (tauxx * Wxy - tauxz * Wyz) + (Wxy * tauyy + Wxz * tauyz);
Real dTauxz = twoG * Dxz - (tauxx * Wxz + tauxy * Wyz) + (Wxy * tauyz + Wxz * tauzz);
Real dTauyz = twoG * Dyz - (tauxy * Wxz + tauyy * Wyz) - (Wxy * tauxz - Wyz * tauzz);
```

**公式**：

\[
\frac{d\pmb{\sigma}}{dt} = \dot{\pmb{\phi}}\cdot \pmb{\sigma} - \pmb{\sigma}\cdot \dot{\pmb{\phi}} +\dot{\pmb{\sigma}} \quad (3)
\]

**解读**：
代码将公式 (3) 展开为分量形式。[ \dot{\pmb{\phi}} ] 为旋转率张量（反对称部分，由 `W` 分量表示），[ \dot{\pmb{\sigma}} ] 为 Jaumann 应力率（见公式 (4)），由弹性本构关系给出。项 `2.0 * (tauxy * Wxy + tauxz * Wxz)` 等对应 [ \dot{\pmb{\phi}}\cdot \pmb{\sigma} - \pmb{\sigma}\cdot \dot{\pmb{\phi}} ] 的各分量。

---

#### 公式 (4)：Jaumann 应力率（本构方程）

**代码**（[SphForceWCSPH.cu，第 1537–1572 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1537)）：

```cuda
Real trD = Dxx + Dyy + Dzz;
Real edia = 1.0f / 3.0f * trD;   // edia = tr(D)/3
Real twoG = 2 * paramsD.G_shear;
Real threeK = 3 * paramsD.K_bulk;

// Jaumann rate sigma_dot = 2G*(D - 1/3 tr(D)*I) + K*tr(D)*I
Real dTauxx = twoG * (Dxx - edia) + threeK * edia  + (rotation terms);
Real dTauyy = twoG * (Dyy - edia) + threeK * edia  + (rotation terms);
Real dTauzz = twoG * (Dzz - edia) + threeK * edia  + (rotation terms);
Real dTauxy = twoG * Dxy + (rotation terms);
Real dTauxz = twoG * Dxz + (rotation terms);
Real dTauyz = twoG * Dyz + (rotation terms);
```

**公式**：

\[
\dot{\pmb{\sigma}} = 2G\left(\dot{\pmb{\epsilon}} -\frac{1}{3}\mathrm{tr}(\dot{\pmb{\epsilon}})\mathbf{I}\right) + \frac{1}{3} K\mathrm{tr}(\dot{\pmb{\epsilon}})\mathbf{I} \quad (4)
\]

**解读**：
代码中 `G_shear` 为剪切模量 [ G ]，`K_bulk` 为体积模量 [ K ]。`edia = tr(D)/3` 对应 [ \frac{1}{3}\mathrm{tr}(\dot{\pmb{\epsilon}}) ]，`Dxx - edia` 等是应变率张量偏量部分，`threeK * edia` 对应体积弹性项。注意此处 [ \frac{1}{3}K\mathrm{tr}(\dot{\pmb\epsilon})\mathbf{I} ] 等价于 [ \frac{1}{3}K\,\mathrm{tr}(\dot{\pmb\epsilon})\mathbf{I} ]，代码将其写成 `threeK * edia`（`threeK = 3K`，`edia = tr/3`，因此 `threeK * edia = K * tr(D)`）与公式完全一致。

---

#### 公式 (5)：考虑塑性流动的弹性应变率张量

**代码**（[SphFluidDynamics.cu，第 362 行附近](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L362)）：

```cuda
// TauEulerStep - plastic strain rate via Chi (lambda_dot)
Real Chi = abs(tau_tr - tau_n) * paramsD.INV_G_shear / dT;
// Chi = |tau_tr - tau_n| / (G * dt) = plastic strain rate lambda_dot
```

**公式**：

\[
\dot{\pmb{\epsilon}} = \frac{1}{2} (\nabla \mathbf{u} + \nabla \mathbf{u}^{\mathrm{T}}) - \frac{1}{\sqrt{2}}\dot{\lambda}\frac{\pmb{\tau}}{\|\tau\|} \quad (5)
\]

**解读**：
代码中 `Chi` 对应塑性应变率 [ \dot{\lambda} ]，通过试验应力与当前应力之差除以 [ G \Delta t ] 计算得到（与文中定义 [ \dot{\lambda} = (\bar{\tau}^* - \bar{\tau}^n)/(G\Delta t) ] 一致）。公式 (5) 第一项（弹性应变率对称部分）通过速度梯度的 SPH 离散 [ L_{ij} ] 在 `CrmRHS` 中的 `Dxy`, `Dxz`, `Dyz` 等对称分量隐式处理，第二项塑性修正通过 `TauEulerStep` 中的 yield-surface 投影（Drucker-Prager 准则）体现（见公式 (23) 的实现）。

---

#### 2.1.2 Dynamic equation of the solid bodies（固体动力学方程）

#### 公式 (6)：DVI 问题（含摩擦约束的 Newton-Euler 方程）

**代码**（[SphBceManager.cu，第 302–381 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphBceManager.cu#L302)）：

```cuda
// CalcRigidForces_D kernel — accumulates hydrodynamic force/torque on each rigid body
// Force = derivVelRhoD * markerMass  (= m * dv/dt, the SPH-computed force density)
Force = mR3(derivatives[sorted_index]) * markerMass;
Real3 dist3 = mR3(positions[sorted_index]) - body_pos[body_ID];
Torque = cross(dist3, Force);
// Block-wise reduction over all BCE markers of one body
atomicAdd(&body_forces[body_ID].x, sharedForces[0].x);
atomicAdd(&body_torques[body_ID].x, sharedTorques[0].x);
```

**公式**：

\[
\begin{array}{rl}
& {\dot{\mathbf{q}} = \mathbf{L}(\mathbf{q})\mathbf{u}}\\
& {\mathbf{M}\dot{\mathbf{u}} = \mathbf{f}(t,\mathbf{q},\mathbf{u}) + \sum_{k\in \mathcal{A}(\mathbf{q},\delta)}\left(\gamma_{k,n}\mathbf{D}_{k,n} + \gamma_{k,v}\mathbf{D}_{k,v} + \gamma_{k,w}\mathbf{D}_{k,w}\right)}\\
& {k\in \mathcal{A}(\mathbf{q},\delta):0\leq \gamma_{k,n}\perp \phi_k(\mathbf{q})\geq 0}\\
& {\left(\gamma_{k,v},\gamma_{k,w}\right) = \underset{\sqrt{(\gamma_{k,v})^2 + (\gamma_{k,w})^2}\leq \mu_k^f}{\mathrm{argmin}}\mathbf{u}^\top \left(\gamma_{k,v}\mathbf{D}_{k,v} + \gamma_{k,w}\mathbf{D}_{k,w}\right)}
\end{array} \quad (6)
\]

**解读**：
代码中 `CalcRigidForces_D` kernel 计算了流体作用于刚体的总力 [ \mathbf{F}_{body} ] 和力矩 [ \mathbf{T}_{body} ]，这些力矩通过 FSI 接口传递给 Chrono 的刚体动力学求解器（位于 `ChFsiInterface.cpp`），由后者求解完整的带约束 Newton-Euler 方程组 (6)。刚体与颗粒之间的接触和摩擦通过 DVI 互补条件 (6c) 和最小化问题 (6d) 处理，具体实现在 Chrono 核心模块（`chrono/physics/` 目录）中，属于 MBS（多体系统）求解器的范畴。

---

### 2.2 Spatial Discretization（空间离散）

---

#### 公式 (7)：SPH 核函数近似

**代码**（[SphGeneral.cuh，第 53–65 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphGeneral.cuh#L53)）：

```cuda
// W3h_CubicSpline - cubic spline kernel function
inline __host__ __device__ Real W3h_CubicSpline(Real d, Real invh) {
    Real q = fabs(d) * invh;
    if (q < 1) {
        Real alpha = INVPI * cube(invh) / 4;
        return alpha * (cube(2 - q) - 4 * cube(1 - q));
    }
    if (q < 2) {
        Real alpha = INVPI * cube(invh) / 4;
        return alpha * cube(2 - q);
    }
    return 0;
}
```

**代码**（[SphForceWCSPH.cu，第 618–647 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L618)）：

```cuda
// calcRho_kernel - SPH density summation (Eq. 7 application)
Real m_j = paramsD.markerMass;
Real d = length(dist3);
Real W3 = W3h(paramsD.kernel_type, d, paramsD.ooh);  // W_ij
sum_mW += m_j * W3;             // sum_j m_j * W_ij
sum_W += W3;                     // sum_j W_ij
sum_mW_rho += m_j * W3 / sortedRhoPreMu_old[j].x;  // sum_j m_j * W_ij / rho_j
// V_i = 1/sum_j W_ij  (volume from particle, Eq. 7)
// rho_i = m_i / V_i = m_i * sum_j W_ij
sortedRhoPreMu[index].x = sum_mW / sum_mW_rho;
```

**公式**：

\[
f_{i} = \sum_{j}f_{j}W_{ij}\mathcal{V}_{j} \quad (7)
\]

**解读**：
公式 (7) 是 SPH 函数近似的基础。`V_i = (sum_j W_ij)^{-1}` 是 SPH 粒子的体积，代码中由 `sum_W` 的倒数来近似。`W3h()` 为核函数的调度函数，根据参数 `kernel_type` 选择不同的核（三次样条、Wendland 等），并通过 `paramsD.ooh = 1/h` 传入核长倒数。

---

#### 公式 (8)：三次样条核函数

**代码**（[SphGeneral.cuh，第 53–65 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphGeneral.cuh#L53)）：

```cuda
inline __host__ __device__ Real W3h_CubicSpline(Real d, Real invh) {
    Real q = fabs(d) * invh;   // q = r_ij / h

    if (q < 1) {
        Real alpha = INVPI * cube(invh) / 4;
        // alpha_d * (2/3 - R^2 + 1/2 R^3) form, combined as (2-q)^3 - 4*(1-q)^3
        return alpha * (cube(2 - q) - 4 * cube(1 - q));
    }
    if (q < 2) {
        Real alpha = INVPI * cube(invh) / 4;
        return alpha * cube(2 - q);    // alpha_d * 1/6 * (2-R)^3
    }
    return 0;   // q >= 2: zero support
}
```

**公式**：

\[
W_{ij} = \alpha_{d}\left\{ \begin{array}{ll}\frac{2}{3} -R^{2} + \frac{1}{2}R^{3} & 0\leq R< 1\\ \frac{1}{6}(2 - R)^{3} & 1\leq R< 2\\ 0 & R\geq 2, \end{array} \right. \quad (8)
\]

其中 [ R = r_{ij}/h ]，[ \alpha_d = 3/(2\pi h^3) ]（三维）。

**解读**：
代码中 `alpha = INVPI * cube(invh) / 4`（其中 `INVPI = 1/π`，`cube(invh) = 1/h^3`），因此 `alpha = 1/(4πh^3)`。两段公式通过展开验证：当 `q < 1` 时，`(2-q)^3 - 4*(1-q)^3 = 8-12q+6q^2-q^3 - 4*(1-3q+3q^2-q^3) = 8-12q+6q^2-q^3-4+12q-12q^2+4q^3 = 4-6q^2+3q^3`；而论文公式第一段为 `(2/3-R^2+R^3/2)*alpha_d`，两者乘以 `alpha` 后完全一致（[ 3/(2\pi h^3) \cdot (2/3-q^2+q^3/2) = 1/(4\pi h^3)\cdot(4-6q^2+3q^3) ]）。当 `q` 在 `[1,2)` 时，`alpha * (2-q)^3 = 1/(4πh^3) * (2-q)^3 = 3/(2πh^3) * 1/6 * (2-q)^3`，与公式 (8) 第二段一致。

---

#### 校正矩阵 [ \mathbf{G}_i^{-1} ] 的计算

**代码**（[SphForceWCSPH.cu，第 31–103 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L31)）：

```cuda
__device__ __inline__ void calc_G_Matrix(/* ... */) {
    // Compute inverse of correction matrix G_i
    // mGi[9] = { 0 }  (elements of G_i^{-1})
    Real mGi[9] = {0};
    for (int n = NLStart; n < NLEnd; n++) {
        uint j = neighborList[n];
        Real3 rij = Distance(posRadA, posRadB);        // r_ij = x_i - x_j
        Real3 grad_i_wij = GradW3h(paramsD.kernel_type, rij, paramsD.ooh);
        Real3 grw_vj = grad_i_wij * paramsD.volume0;  // grad W_ij * V_j
        // G_i^{-1} = -sum_j r_ij ⊗ (grad_i W_ij * V_j)
        mGi[0] -= rij.x * grw_vj.x;  // (1,1)
        mGi[1] -= rij.x * grw_vj.y;  // (1,2)
        mGi[2] -= rij.x * grw_vj.z;  // (1,3)
        mGi[3] -= rij.y * grw_vj.x;  // (2,1)
        mGi[4] -= rij.y * grw_vj.y;  // (2,2)
        mGi[5] -= rij.y * grw_vj.z;  // (2,3)
        mGi[6] -= rij.z * grw_vj.x;  // (3,1)
        mGi[7] -= rij.z * grw_vj.y;  // (3,2)
        mGi[8] -= rij.z * grw_vj.z;  // (3,3)
    }
    // Invert mGi to get G_i
    Real Det = ...;  // 3x3 determinant
    Real OneOverDet = 1 / Det;
    G_i[0] = (mGi[4] * mGi[8] - mGi[5] * mGi[7]) * OneOverDet;
    // ... (full 3x3 inverse)
}
```

**公式**（论文正文，公式 [ \mathbf{G}_i^{-1} ] 定义）：

\[
\mathbf{G}_i^{-1} = -\sum_{j} \mathbf{r}_{ij}\nabla_i W_{ij} \mathcal{V}_j
\]

**解读**：
代码通过对所有邻居粒子 [ j ] 累加外积 [ -\mathbf{r}_{ij} \otimes (\nabla_i W_{ij} \mathcal{V}_j) ] 来计算 [ \mathbf{G}_i^{-1} ]，然后通过行列式求逆（3×3 显式逆公式）得到修正矩阵 [ \mathbf{G}_i ]。当行列式绝对值小于 0.01 时退化为单位矩阵以保证数值稳定性。

---

#### 公式 (9)：密度更新的一致性 SPH 离散

**代码**（[SphForceWCSPH.cu，第 1073 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1073)）：

```cuda
// crmDvDt device function — density rate (continuity equation)
Real derivRho = paramsD.markerMass * dot(velMasA - velMasB, gradW);
```

**代码**（[SphForceWCSPH.cu，第 1496–1503 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1496)）：

```cuda
// CrmRHS kernel — contribution to density rate from each neighbor j
Real volumej = paramsD.markerMass / rhoPresMuB.x;  // V_j = m_j / rho_j
Real3 gradW = GradW3h(kernelType, dist3, ooh);      // grad_i W_ij
// derivRho from crmDvDt accumulates: m_j * (u_A - u_B) . grad_i W_ij
//   = rho_i * (u_A - u_B) . G_i * grad_i W_ij * V_j  [consistent form]
```

**公式**：

\[
\frac{d\rho_i}{dt} = -\rho_i \sum_{j} (\mathbf{u}_j -\mathbf{u}_i) \cdot \left( \mathbf{G}_i \cdot \nabla_i W_{ij} \right) \mathcal{V}_j \quad (9)
\]

**解读**：
代码中 `crmDvDt` 函数计算密度变化率时采用 `dot(velMasA - velMasB, gradW) * markerMass`，其中 `gradW` 是核函数梯度 [ \nabla_i W_{ij} ]（可选一致性修正）。注意代码中使用的是 [ \mathbf{u}_A - \mathbf{u}_B = \mathbf{u}_i - \mathbf{u}_j ]，因此 `dot(velMasA - velMasB, gradW)` 等于 [ -(\mathbf{u}_j - \mathbf{u}_i) \cdot \nabla_i W_{ij} ]，乘以质量后得到密度率，整体与公式 (9) 一致（符号由 [ \nabla \cdot \mathbf{u} ] 为负得出正密度增长）。

---

#### 公式 (10)：动量方程的一致性 SPH 离散

**代码**（[SphForceWCSPH.cu，第 1106–1117 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1106)）：

```cuda
// crmDvDt device function — momentum update
// d u_i / dt = (1/rho_i) * sum_j (sigma_j - sigma_i) . (G_i . grad_i W_ij) * V_j + f_b
Real invRhoASq = 1 / (rhoPresMuA.x * rhoPresMuA.x);
Real invRhoBSq = 1 / (rhoPresMuB.x * rhoPresMuB.x);
Real3 MA_gradW = gradW * Mass;   // m_j * grad_i W_ij (= rho_i * V_j * grad_i W_ij approx)

// x-component of dv/dt (sum over all stress components):
Real derivVx = (tauXxYyZz_A.x * invRhoASq + tauXxYyZz_B.x * invRhoBSq) * MA_gradW.x +
               (tauXyXzYz_A.x * invRhoASq + tauXyXzYz_B.x * invRhoBSq) * MA_gradW.y +
               (tauXyXzYz_A.y * invRhoASq + tauXyXzYz_B.y * invRhoBSq) * MA_gradW.z;
// (similar for y and z components)
```

**公式**：

\[
\frac{d\mathbf{u}_i}{dt} = \frac{1}{\rho_i} \sum_{j} (\pmb{\sigma}_j -\pmb{\sigma}_i) \cdot \left( \mathbf{G}_i \cdot \nabla_i W_{ij} \right) \mathcal{V}_j + \mathbf{f}_{b,i} \quad (10)
\]

**解读**：
代码中 `invRhoASq` = [ 1/\rho_i^2 ]，`invRhoBSq` = [ 1/\rho_j^2 ]，`MA_gradW = m_j \nabla_i W_{ij}`。表达式 `(sigma_A * invRhoASq + sigma_B * invRhoBSq) * MA_gradW` 是 SPH 动量方程的对称形式：[ m_j(\sigma_i/\rho_i^2 + \sigma_j/\rho_j^2) \nabla_i W_{ij} ]，等价于使用粒子体积 [ V_j = m_j/\rho_j ] 的离散形式，与公式 (10) 等价（只是采用了对称形式以守恒动量）。体力 `f_b`（重力等）通过 `derivVelRho += mR4(totalFluidBodyForce3, 0)` 在 `CrmRHS` 末尾添加，对应公式 (10) 的 [ \mathbf{f}_{b,i} ] 项。

---

#### 公式 (11)：应变率张量的一致性 SPH 离散

**代码**（[SphForceWCSPH.cu，第 1511–1523 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1511)）：

```cuda
// CrmRHS kernel — velocity gradient accumulation
Real3 vBA = velMasB - velMasA;  // u_j - u_i = -u_{ij}
// L = grad v (velocity gradient tensor, 3x3)
Lxx += volumej * vBA.x * gradW.x;
Lxy += volumej * vBA.x * gradW.y;
Lxz += volumej * vBA.x * gradW.z;
Lyx += volumej * vBA.y * gradW.x;
Lyy += volumej * vBA.y * gradW.y;
Lyz += volumej * vBA.y * gradW.z;
Lzx += volumej * vBA.z * gradW.x;
Lzy += volumej * vBA.z * gradW.y;
Lzz += volumej * vBA.z * gradW.z;

// Split into symmetric (D) and anti-symmetric (W) parts
Real Dxy = 0.5 * (Lxy + Lyx);   // strain rate (symmetric)
Real Dxz = 0.5 * (Lxz + Lzx);
Real Dyz = 0.5 * (Lyz + Lzy);
```

**公式**：

\[
\dot{\pmb{\epsilon}}_i = \frac{1}{2} \sum_{j} \left[ \mathbf{u}_{ji} \mathbf{b}_{ij}^\top + \left( \mathbf{u}_{ji} \mathbf{b}_{ij}^\top \right)^\top \right] \quad (11)
\]

其中 [ \mathbf{b}_{ij} \equiv \mathbf{G}_i \cdot \nabla_i W_{ij} \mathcal{V}_j ]，[ \mathbf{u}_{ji} = \mathbf{u}_j - \mathbf{u}_i ]。

**解读**：
代码中 `vBA = velMasB - velMasA = u_j - u_i = u_{ji}`，`gradW` 为 [ \nabla_i W_{ij} ]，`volumej = m_j/\rho_j = V_j`。速度梯度张量 `L` 的各分量 `Lxy = sum_j V_j * (u_j - u_i)_x * (grad W_ij)_y` 对应 [ \sum_j \mathbf{u}_{ji,x}(\mathbf{b}_{ij})_y ]（不含 [ \mathbf{G}_i ] 时，`gradW` 直接作为 [ \nabla_i W_{ij} ]）。应变率张量对称部分 `Dxy = 0.5 * (Lxy + Lyx)` 精确对应公式 (11) 中的对称化操作 [ \frac{1}{2}[\mathbf{u}_{ji}\mathbf{b}_{ij}^\top + (\mathbf{u}_{ji}\mathbf{b}_{ij}^\top)^\top] ]。

---

#### 公式 (12)：旋转率张量的一致性 SPH 离散

**代码**（[SphForceWCSPH.cu，第 1532–1534 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1532)）：

```cuda
// Anti-symmetric part W (rotation rate tensor)
Real Wxy = 0.5 * (Lxy - Lyx);
Real Wxz = 0.5 * (Lxz - Lzx);
Real Wyz = 0.5 * (Lyz - Lzy);
```

**公式**：

\[
\dot{\pmb{\phi}}_i = \frac{1}{2} \sum_{j} \left[ \mathbf{u}_{ji} \mathbf{b}_{ij}^\top - \left( \mathbf{u}_{ji} \mathbf{b}_{ij}^\top \right)^\top \right] \quad (12)
\]

**解读**：
旋转率张量 [ \dot{\pmb{\phi}}_i ] 为速度梯度张量 [ \mathbf{L} ] 的反对称部分。代码中 `Wxy = 0.5*(Lxy - Lyx)` 等精确对应反对称化操作 [ \frac{1}{2}[\mathbf{u}_{ji}\mathbf{b}_{ij}^\top - (\mathbf{u}_{ji}\mathbf{b}_{ij}^\top)^\top] ]。旋转率张量用于构造 Jaumann 应力率（公式 (3)/(13)）中的旋转修正项。

---

#### 公式 (13)：应力率张量的完整 SPH 离散

**代码**（[SphForceWCSPH.cu，第 1564–1572 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1564)）：

```cuda
// CrmRHS kernel — full stress rate tensor computation
// dSigma/dt = W*sigma - sigma*W + G*(D - 1/3 tr(D)*I) + K*tr(D)*I
Real dTauxx = twoG * (Dxx - edia) + 2.0 * (tauxy * Wxy + tauxz * Wxz) + threeK * edia;
Real dTauyy = twoG * (Dyy - edia) - 2.0 * (tauxy * Wxy - tauyz * Wyz) + threeK * edia;
Real dTauzz = twoG * (Dzz - edia) - 2.0 * (tauxz * Wxz + tauyz * Wyz) + threeK * edia;
// off-diagonals:
Real dTauxy = twoG * Dxy - (tauxx * Wxy - tauxz * Wyz) + (Wxy * tauyy + Wxz * tauyz);
Real dTauxz = twoG * Dxz - (tauxx * Wxz + tauxy * Wyz) + (Wxy * tauyz + Wxz * tauzz);
Real dTauyz = twoG * Dyz - (tauxy * Wxz + tauyy * Wyz) - (Wxy * tauxz - Wyz * tauzz);
```

**公式**：

\[
\frac{d\pmb{\sigma}_i}{dt} = \frac{1}{2} \left\{ \sum_{j} \left[ \mathbf{u}_{ji} \mathbf{b}_{ij}^\top - \left( \mathbf{u}_{ji} \mathbf{b}_{ij}^\top \right)^\top \right] \pmb{\sigma}_i - \pmb{\sigma}_i \sum_{j} \left[ \mathbf{u}_{ji} \mathbf{b}_{ij}^\top - \left( \mathbf{u}_{ji} \mathbf{b}_{ij}^\top \right)^\top \right] \right\}
\]

\[
+ G \left\{ \sum_{j} \left[ \mathbf{u}_{ji} \mathbf{b}_{ij}^\top + \left( \mathbf{u}_{ji} \mathbf{b}_{ij}^\top \right)^\top \right] - \frac{1}{3} \mathrm{tr} \left( \cdots \right) \mathbf{I} \right\}
+ \frac{1}{6} K \left\{ \mathrm{tr} \left( \cdots \right) \mathbf{I} \right\} \quad (13)
\]

**解读**：
代码将公式 (13) 完全展开为 6 个独立分量（利用 [ \pmb{\sigma} ] 的对称性）：
- `twoG * (Dxx - edia)` 对应弹性偏量项 [ 2G(\dot{\epsilon}_{xx} - \frac{1}{3}\mathrm{tr}(\dot{\pmb{\epsilon}})) ]
- `threeK * edia` = `K * tr(D)` 对应体积弹性项 [ K \cdot \mathrm{tr}(\dot{\pmb{\epsilon}}) / 3 \cdot 3 = K\mathrm{tr}(\dot{\pmb{\epsilon}}) ]（通过 `threeK = 3K`, `edia = tr/3`）  
- `2.0 * (tauxy * Wxy + tauxz * Wxz)` 等对应 Jaumann 旋转修正 [ \dot{\pmb{\phi}} \cdot \pmb{\sigma} - \pmb{\sigma} \cdot \dot{\pmb{\phi}} ] 的各分量

---

#### 公式 (14)：粒子位置更新（含 XSPH 修正）

**代码**（[SphFluidDynamics.cu，第 313–316 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L313)）：

```cuda
// PositionEulerStep - particle position update with XSPH correction
__device__ void PositionEulerStep(Real dT, const Real3& vel, Real4& pos) {
    Real3 p = mR3(pos);
    p += dT * vel;  // x_i^{new} = x_i + dt * (u_i + vel_XSPH_i)
    pos = mR4(p, pos.w);
}
```

**代码**（[SphForceWCSPH.cu，第 1909–1913 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1909)）：

```cuda
// ShiftingAccumulateNeighborContrib — XSPH velocity correction term
if constexpr (SHIFT == ShiftingMethod::XSPH || SHIFT == ShiftingMethod::PPST_XSPH) {
    Real3 velMasB = sortedVelMas[j];
    Real4 rhoPreMuB = sortedRhoPreMu[j];
    Real rho_bar = 0.5f * (rhoPreMuA.x + rhoPreMuB.x);
    deltaV += (velMasB - velMasA) * W3h(paramsD.kernel_type, d, paramsD.ooh) / rho_bar;
}
// ...
// Final XSPH velocity stored in vel_XSPH_D:
result = paramsD.markerMass * paramsD.shifting_xsph_eps * deltaV;
```

**代码**（[SphFluidDynamics.cu，第 592–594 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphFluidDynamics.cu#L592)）：

```cuda
// EulerStep_D — position update using u_i + vel_XSPH
PositionEulerStep(dT, velMasD[index] + vel_XSPH_D[index], posRadD[index]);
```

**公式**：

\[
\frac{d\mathbf{x}_i}{dt} = \mathbf{u}_i - \xi \sum_{j} \mathbf{u}_{ij} W_{ij} \mathcal{V}_j \quad (14)
\]

**解读**：
代码将公式 (14) 分两步实现：
1. **XSPH 修正速度**（`Calc_Shifting_D` 中，`ShiftingMethod::XSPH` 分支）：计算 `deltaV = sum_j (u_j - u_i) * W_ij / rho_bar`，乘以 `markerMass * shifting_xsph_eps` 后存入 `vel_XSPH_D[i]`，其中 `shifting_xsph_eps` 即论文中的 [ \xi ]（默认 0.5）；
2. **位置更新**（`EulerStep_D`）：`dx/dt = u_i + vel_XSPH_i`，这里 `vel_XSPH_i = xi * m * sum_j (u_j - u_i)*W_ij/rho_bar ≈ xi * sum_j (u_j - u_i)*W_ij*V_j = -xi * sum_j u_ij * W_ij * V_j`，与公式 (14) 完全一致。

---

### 2.3 Boundary Conditions（边界条件）

---

#### 公式 (15)：刚体表面速度

**代码**（[SphBceManager.cu，第 627–638 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphBceManager.cu#L627)）：

```cuda
// UpdateBodyMarkerState_D kernel — BCE marker velocity update
// v_B = v_body + omega_body x r_c(x)
Real3 omega_cross = cross(body_angvel[body_ID], local);  // omega x r_c
velocities[sorted_index] =
    body_linvel[body_ID] +                               // u_body
    mR3(dot(u, omega_cross), dot(v, omega_cross), dot(w, omega_cross));  // rotated to global frame
```

**公式**：

\[
\mathbf{u}_B = \mathbf{u}_{body} + \pmb{\omega}_{body}\times \mathbf{r}_c(\mathbf{x}) \quad \text{for } \mathbf{x}\in \Gamma \quad (15)
\]

**解读**：
代码中 `local` 是 BCE 粒子在刚体局部坐标系中的位置（即 [ \mathbf{r}_c ]），`body_angvel[body_ID]` 是角速度 [ \pmb{\omega}_{body} ]，`body_linvel[body_ID]` 是线速度 [ \mathbf{u}_{body} ]。`cross(body_angvel, local)` 计算叉积 [ \pmb{\omega} \times \mathbf{r}_c ] 后，通过旋转矩阵 `[u,v,w]` 转换到全局坐标系（通过 `RotationMatrixFromQuaternion` 从四元数获得），最终 BCE 粒子速度精确对应公式 (15)。

---

#### 公式 (16)：BCE 粒子速度线性外推

**代码**（[SphForceWCSPH.cu，第 871–904 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L871)）：

```cuda
// CrmHolmesBC kernel — Holmes method velocity extrapolation
// u_j = (d_j/d_i) * (u_B - u_i) + u_B  [Eq. 16]
Real dFluidBCE = dBCE / dFluid;   // d_j / d_i
// Predication: clamp ratio to [0, 0.5]
int predicateAB = (dFluidBCE > 0.5);
dFluidBCE = predicateAB ? 0.5 : dFluidBCE;
velMasB_new = dFluidBCE * (prescribedVel - sortedVelMasD[j]) + prescribedVel;
// prescribedVel = u_B (prescribed body boundary velocity)
// sortedVelMasD[j] = u_i (fluid particle velocity)
```

**公式**：

\[
\mathbf{u}_j = \frac{d_j}{d_i} (\mathbf{u}_B - \mathbf{u}_i) + \mathbf{u}_B \quad (16)
\]

**解读**：
代码中 `prescribedVel = sortedVelMasD[index]`（即刚体边界速度 [ \mathbf{u}_B ]），`sortedVelMasD[j]` 是最近邻流体粒子速度 [ \mathbf{u}_i ]，`dFluidBCE = dBCE/dFluid = d_j/d_i`（比值上限 0.5）。表达式 `dFluidBCE * (prescribedVel - sortedVelMasD[j]) + prescribedVel` 精确对应公式 (16)：[ (d_j/d_i)(\mathbf{u}_B - \mathbf{u}_i) + \mathbf{u}_B ]。

---

#### 公式 (17)：距离指示函数（基于核函数支撑域）

**代码**（[SphForceWCSPH.cu，第 797–826 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L797)）：

```cuda
// calcKernelSupport kernel — compute chi indicator for each particle
__global__ void calcKernelSupport(/* ... */) {
    Real W0 = W3h(paramsD.kernel_type, 0, paramsD.ooh);  // W(0)
    Real sum_W_all = W0;        // sum over all neighbors (fluid + solid)
    Real sum_W_identical = W0;  // sum over same-type neighbors only
    for (int i = NLStart; i < NLEnd; i++) {
        uint j = neighborList[i];
        Real W3 = W3h(paramsD.kernel_type, d, paramsD.ooh);
        sum_W_all += W3;
        if (abs(index_type - sortedRhoPreMu[j].w) < 0.001) {
            sum_W_identical += W3;  // only same-type neighbors
        }
    }
    sortedKernelSupport[index].x = sum_W_all;       // sum_{k in Omega_f+s} W_{ik}
    sortedKernelSupport[index].y = sum_W_identical; // sum_{k in same type} W_{ik}
}
```

**代码**（[SphForceWCSPH.cu，第 866–868 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L866)）：

```cuda
// CrmHolmesBC — chi and distance calculation (Eq. 17)
Real2 kernelSupport = sortedKernelSupport[index];
Real chi_BCE = kernelSupport.x / kernelSupport.y;   // chi_j = sum_{k in Omega_s} W_jk / sum_{k in Omega_f+s} W_jk
Real dBCE = SuppRadii * (2 * chi_BCE - 1);          // d_j = kappa * h_j * (2*chi_j - 1)
```

**公式**：

\[
d_i = \kappa h_i(2\chi_i - 1),\quad d_j = \kappa h_j(2\chi_j - 1) \quad (17)
\]

\[
\chi_{i} = \frac{\sum_{k\in\Omega_{f}}W_{ik}}{\sum_{k\in\Omega_{f}\cup\Omega_{s}}W_{ik}},\quad \chi_{j} = \frac{\sum_{k\in\Omega_{s}}W_{jk}}{\sum_{k\in\Omega_{f}\cup\Omega_{s}}W_{jk}}
\]

**解读**：
代码中 `calcKernelSupport` 分别计算所有邻居的核函数和（`sum_W_all` = 分母 [ \sum_{k\in\Omega_f\cup\Omega_s}W_{ik} ]）及同类型邻居的和（`sum_W_identical` = 分子）。`chi_BCE = sum_W_all / sum_W_identical` 近似对应 [ \chi_j ]（固体 BCE 粒子）。距离 `dBCE = SuppRadii * (2*chi_BCE - 1)` 即公式 (17) 中的 [ d_j = \kappa h_j(2\chi_j - 1) ]，其中 `SuppRadii = h_multiplier * h = 2h`（ [ \kappa = 2 ]，对于三次样条核）。

---

### 2.4 Enforcing Particle Regularity via PPST（粒子正则化：PPST）

---

#### 公式 (18)：穿透量 PPST 粒子位移向量

**代码**（[SphForceWCSPH.cu，第 1916–1925 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1916)）：

```cuda
// ShiftingAccumulateNeighborContrib — PPST shifting vector calculation (Eq. 18)
if constexpr (SHIFT == ShiftingMethod::PPST || SHIFT == ShiftingMethod::PPST_XSPH) {
    // Fictitious sphere diameter D_s,i = 2 * (3m_i/4pi*rho_i)^{1/3}
    Real dFictitious = paramsD.d0 * Real(1.241);  // D_s,i / 2 ≈ 1.241 * d0
    if (d < 1.25f * dFictitious) {
        Real oodFictitious = 1 / dFictitious;
        Real delta_ij = (dFictitious - d) * oodFictitious;  // (D_s,i - r_ij) / D_s,i
        Real beta = (delta_ij > 0) ? paramsD.shifting_ppst_push : paramsD.shifting_ppst_pull;
        // shifting_ppst_push = beta_1 = 3, shifting_ppst_pull = beta_2 = 1
        inner_sum += beta * fmax(delta_ij, static_cast<Real>(-0.1f)) * (dist3 / d);
        // e_ij = dist3 / d  (unit vector from j to i)
    }
}
```

**代码**（[SphForceWCSPH.cu，第 1983–1994 行](https://github.com/marksweli/chrono/blob/main/src/chrono_fsi/sph/physics/SphForceWCSPH.cu#L1983)）：

```cuda
// Calc_Shifting_D — PPST result post-processing
} else if constexpr (SHIFT == ShiftingMethod::PPST) {
    Real vA = length(velMasA);         // ||u_i||
    Real vAdT = vA * paramsD.dT;       // ||u_i|| * dt

    inner_sum = vAdT * inner_sum;       // delta_r_i = ||u_i|| * dt * sum(...)
    Real upper_limit = 0.05f * vAdT;   // 5% limit on |delta_r_i| / (||u_i|| dt)
    Real cur_len = length(inner_sum);
    if (cur_len > upper_limit) {
        inner_sum *= (upper_limit / (cur_len + 1e-9f));
    }
    result = inner_sum / paramsD.dT;  // Store as velocity for position update
}
```

**公式**：

\[
\delta \mathbf{r}_i = \left\{ \begin{array}{ll}\beta_1\| \mathbf{u}_i\| \Delta t\sum_j\delta r_{ij}\mathbf{e}_{ij} & \delta r_{ij} > 0\\ \beta_2\| \mathbf{u}_i\| \Delta t\sum_j\delta r_{ij}\mathbf{e}_{ij} & \delta r_0< \delta r_{ij}\leq 0\\ \beta_2\| \mathbf{u}_i\| \Delta t\sum_j\delta r_0\mathbf{e}_{ij} & \delta r_{ij}\leq \delta r_0, \end{array} \right. \quad (18)
\]

其中 [ \delta r_{ij} = (D_{s,i} - r_{ij})/D_{s,i} ]，[ \mathbf{e}_{ij} = \mathbf{r}_{ij}/r_{ij} ]，[ \beta_1 = 3 ]，[ \beta_2 = 1 ]，[ \delta r_0 = -0.1 ]。

**解读**：
代码中 `dFictitious = d0 * 1.241` 近似为虚拟球半径 [ D_{s,i}/2 ]（从 [ D_{s,i} = 2(3m_i/4\pi\rho_i)^{1/3} ] 推导，当 [ m_i/\rho_i = d_0^3 ] 时，[ D_{s,i}/2 \approx 0.620 d_0 \cdot 2 = d_0 \cdot 1.241 ]）。
`delta_ij = (dFictitious - d) / dFictitious` 即穿透量 [ \delta r_{ij} ]；`beta` 根据 `delta_ij > 0` 取 `shifting_ppst_push`（[ \beta_1 = 3 ]）或 `shifting_ppst_pull`（[ \beta_2 = 1 ]）；`fmax(delta_ij, -0.1f)` 对应 [ \delta r_0 = -0.1 ] 的截断（公式第三段）；`dist3 / d = e_{ij}`。上限 `0.05 * vAdT` 对应论文中 [ \|\delta\mathbf{r}_i\| / (\|\mathbf{u}_i\|\Delta t) < 5\% ] 的约束。

---

*（续见 paper_code_analysis_part2.md，包含 2.5 时间积分与 2.6 应力后处理）*
