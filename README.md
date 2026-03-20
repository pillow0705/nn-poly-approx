# 用多项式函数一致逼近神经网络：理论与实验

> **摘要**：本文探讨以多项式函数族替代多层感知机（MLP），以降低推理阶段的计算复杂度。我们从 Weierstrass 定理出发，建立多项式一致逼近的理论依据，设计逐层激活替换的符号化组合方法，并在 $\sin(x)$ 逼近任务上完成两组实验：精度对比与推理速度对比。实验表明，所得 18 阶多项式在单点推理速度上比原始 MLP 快约 **16 倍**。

---

## 1. 问题背景

### 1.1 神经网络的万能逼近定理

**定理（Cybenko, 1989；Hornik, 1991）**：设 $\sigma: \mathbb{R} \to \mathbb{R}$ 为非常数、有界且连续的激活函数，则对任意紧集 $K \subset \mathbb{R}^n$、任意连续函数 $f \in C(K)$ 及 $\varepsilon > 0$，存在单隐层网络

$$F(x) = \sum_{j=1}^{N} \alpha_j \,\sigma(w_j^\top x + b_j)$$

使得 $\sup_{x \in K} |F(x) - f(x)| < \varepsilon$。

该定理保证了神经网络的表达能力，但并不给出构造性算法，也不约束参数量。现代深网络的推理代价（浮点运算数、内存访问、超越函数调用）在资源受限场景下仍是瓶颈。

### 1.2 多项式的 Weierstrass 一致逼近定理

**定理（Weierstrass, 1885）**：设 $f \in C[a, b]$，则对任意 $\varepsilon > 0$，存在多项式 $p$ 使得

$$\|f - p\|_\infty = \sup_{x \in [a,b]} |f(x) - p(x)| < \varepsilon.$$

多项式的优势在于：**求值仅需乘法与加法**，不涉及任何超越函数，且 Horner 算法给出了度为 $n$ 的多项式求值的理论最优下界——$n$ 次乘法 $+$ $n$ 次加法，共 $2n$ 次算术运算。

### 1.3 其他函数族的一致逼近条件

| 函数族 | 一致逼近条件 | 局限 |
|--------|-------------|------|
| **多项式** | $f \in C[a,b]$（Weierstrass） | 高维时基数指数增长 |
| **Fourier 级数** | $f \in C_{2\pi}$ 且绝对连续，$f' \in L^2$（Dirichlet 条件） | 吉布斯现象；需周期延拓 |
| **小波** | $f \in L^2(\mathbb{R})$，选取合适母小波（Daubechies 等） | 构造更复杂；非线性逼近 |

多项式在有界闭区间上条件最宽松，且与硬件运算最亲和，是本文的选择。

---

## 2. 实验理论准备

### 2.1 问题形式化

给定已训练的 MLP $\mathcal{F}: [a,b] \to \mathbb{R}$，求多项式 $p \in \mathcal{P}_n$ 使得

$$\|p - \mathcal{F}\|_\infty = \max_{x \in [a,b]} |p(x) - \mathcal{F}(x)|$$

尽可能小，同时 $p$ 的求值代价显著低于 $\mathcal{F}$。

本文取 $[a,b] = [0, 2]$，目标函数为 $f(x) = \sin(x)$，先训练 MLP 逼近 $f$，再用多项式逼近该 MLP。

### 2.2 MLP 模型架构

```
输入 x ∈ ℝ
      │
  ┌───▼────────────────────────────┐
  │  Linear(1 → 8):  h = W₁x + b₁ │  8×1 权重 + 8 偏置
  └───────────────┬────────────────┘
                  │  tanh(·) × 8 neurons
  ┌───────────────▼────────────────┐
  │  Linear(8 → 8):  h = W₂h + b₂ │  8×8 权重 + 8 偏置
  └───────────────┬────────────────┘
                  │  tanh(·) × 8 neurons
  ┌───────────────▼────────────────┐
  │  Linear(8 → 1):  y = W₃h + b₃ │  1×8 权重 + 1 偏置
  └───────────────┬────────────────┘
      输出 y ∈ ℝ
```

**参数量**：$8 + 8 + 64 + 8 + 8 + 1 = 97$ 个参数。

**前向传播计算量**（浮点运算，每次 MAC 算 2 次）：

| 层 | 乘加 (MAC) | 超越函数 |
|----|-----------|---------|
| Linear $1 \to 8$ | 8 | — |
| Tanh $\times 8$ | — | 8 |
| Linear $8 \to 8$ | 64 | — |
| Tanh $\times 8$ | — | 8 |
| Linear $8 \to 1$ | 8 | — |
| **合计** | **80 MAC ≈ 160 FLOP** | **16 次超越函数** |

每次 $\tanh$ 调用约等价于 $\sim 20$ 次算术运算（指数+除法），故总等效运算量约 $\mathbf{480}$ 次。

### 2.3 多项式的构造方法

#### Step 1：激活函数的 Chebyshev 最佳逼近

收集所有隐层的 pre-activation 值域 $[a_0, b_0]$，在此区间上对 $\tanh$ 做 $d$ 阶 Chebyshev 逼近。取第一类 Chebyshev 节点

$$t_k = \cos\!\left(\frac{2k+1}{2(d+1)}\pi\right), \quad k = 0, \ldots, d,$$

将 $t_k$ 从 $[-1,1]$ 映射回 $[a_0, b_0]$：

$$x_k = \frac{b_0 - a_0}{2}\,t_k + \frac{a_0 + b_0}{2},$$

拟合 Chebyshev 级数并转换为关于标准化变量 $t$ 的多项式：

$$\widehat{\sigma}(u) = \sum_{i=0}^{d} c_i \,t^i, \qquad t = \frac{2u - (a_0+b_0)}{b_0 - a_0}.$$

#### Step 2：符号化逐层组合

利用 SymPy 对网络进行符号计算。设输入为符号变量 $x$：

$$\text{第一层：} \quad h_i^{(1)}(x) = \widehat{\sigma}\!\left(W_1^{(i)} x + b_1^{(i)}\right), \quad i = 1,\ldots,8$$

$$\text{第二层：} \quad h_i^{(2)}(x) = \widehat{\sigma}\!\left(\sum_{j=1}^{8} W_2^{(ij)} h_j^{(1)}(x) + b_2^{(i)}\right)$$

$$\text{输出层：} \quad p(x) = \sum_{j=1}^{8} W_3^{(j)}\, h_j^{(2)}(x) + b_3$$

展开后 $p(x)$ 为关于 $x$ 的标准多项式。理论最高阶数为 $d^2$（每次激活替换将阶数提升 $d$ 倍，共 2 层），实际因系数相消通常更低。

#### Step 3：Horner 法求值

最终多项式 $p(x) = \sum_{k=0}^{n} a_k x^k$ 用 Horner 法求值：

$$p(x) = a_0 + x\bigl(a_1 + x\bigl(a_2 + \cdots + x\,a_n\bigr)\bigr)$$

精确需要 $n$ 次乘法 $+$ $n$ 次加法，即 $\mathbf{2n}$ 次算术运算，这是次数为 $n$ 的一元多项式求值的**理论下界**（Motzkin–Belaga 定理）。

---

## 3. 实验方法与结果

### 3.1 实验一：多项式逼近精度

**设置**：网络用 Adam 优化器训练 5000 轮（MSE 损失），激活阶数 $d = 5$，pre-activation 统一范围 $[-1.38,\ 2.56]$（加 20% 余量）。

**结果**：

| 对比 | $\|\cdot\|_\infty$ 误差 |
|------|------------------------|
| MLP $\approx \sin(x)$ | $2.37 \times 10^{-2}$ |
| 多项式 $\approx$ MLP | $1.08 \times 10^{-1}$ |
| 多项式 $\approx \sin(x)$ | $1.31 \times 10^{-1}$ |
| $\tanh$ 激活逼近（5阶） | $2.66 \times 10^{-2}$ |

最终多项式阶数为 **18**（理论上限 $5^2 = 25$，高次项系数相消）。

**所得多项式**（$x \in [0,2]$）：

$$\boxed{p(x) = 0.0253 + 0.7838\,x + 0.3405\,x^2 - 0.3782\,x^3 - 0.0880\,x^4 + 0.1125\,x^5 + 0.0465\,x^6 - 0.0574\,x^7 + \cdots - 6.86 \times 10^{-6}\,x^{18}}$$

（完整系数见 `poly_approx_experiment.py`）

**误差传播分析**：$\tanh$ 逼近误差 $2.66 \times 10^{-2}$ 经两层组合后放大为 $1.08 \times 10^{-1}$（约 4 倍），符合逐层误差累积的理论预期。

### 3.2 实验二：推理速度对比

**设置**：100 个 $[0,2]$ 上的随机输入，每组实验重复 1000 次取均值，分 scalar（逐点）和 batch（批量）两种模式。

**结果**：

| 方法 | 单次耗时 | 相对 Horner |
|------|---------|------------|
| Horner/Python | 4,358 ns | 1× |
| numpy scalar | 5,678 ns | 1.3× |
| MLP scalar（PyTorch） | 67,934 ns | **16×** |
| numpy batch（100点） | 52 μs（522 ns/点） | — |
| MLP batch（100点） | 89 μs（888 ns/点） | 1.7× |

**理论 vs 实测**：

$$\frac{\text{MLP 等效运算量}}{\text{多项式运算量}} = \frac{\sim 480}{2 \times 18} = 13.3\times \quad\longleftrightarrow\quad \text{实测}\ 16\times$$

两者吻合良好。批量模式差距收窄至 1.7×，因为矩阵乘法的 SIMD 向量化使 MLP 在并行度上得到补偿。

---

## 4. 结论

1. **可行性**：逐层 Chebyshev 替换 + 符号化组合是将 MLP 转化为封闭多项式的可行路径，无需重新训练。

2. **速度优势**：在单点推理场景（嵌入式、实时控制、密码学同态加密）下，多项式比 MLP 快约 16 倍，理论分析与实验结果一致。

3. **精度瓶颈**：当前误差的主要来源是逐层误差放大（4×）。提升激活逼近阶数（$d=7$ 或 $d=9$）可改善精度，代价是最终多项式阶数升至 49 或 81，需权衡数值稳定性。

4. **后续方向**：
   - 各层使用不同逼近阶数以平衡误差分配
   - 直接对 $(x_i, \mathcal{F}(x_i))$ 做 $L^\infty$ 回归（线性规划法）
   - 高维输入场景下引入稀疏多项式或张量分解

---

## 参考文献

1. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals, Systems*, 2(4), 303–314.
2. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.
3. Weierstrass, K. (1885). Über die analytische Darstellbarkeit sogenannter willkürlicher Functionen einer reellen Veränderlichen. *Sitzungsberichte der Königlich Preußischen Akademie der Wissenschaften zu Berlin*.
4. Rivlin, T. J. (1990). *Chebyshev Polynomials* (2nd ed.). Wiley.
5. Motzkin, T. S. (1955). Evaluation of polynomials. *Bulletin of the American Mathematical Society*, 61, 163.
