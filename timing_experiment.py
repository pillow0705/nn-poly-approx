"""
对比实验：多项式 vs MLP，100次随机输入的平均计算时间
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ─── 重建MLP（与之前相同的seed/训练）────────────────────────────────────────
DOMAIN = (0.0, 2.0)
HIDDEN = 8
EPOCHS = 5000
torch.manual_seed(42); np.random.seed(42)

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.l3 = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(x)))))

x_train = torch.linspace(*DOMAIN, 500).unsqueeze(1)
y_train = torch.sin(x_train)
model = SmallMLP()
opt = optim.Adam(model.parameters(), lr=1e-3)
for _ in range(EPOCHS):
    opt.zero_grad(); nn.MSELoss()(model(x_train), y_train).backward(); opt.step()
model.eval()

# ─── 多项式系数（之前实验导出）────────────────────────────────────────────────
# numpy.polynomial.polynomial 约定：coeffs[i] = x^i 的系数
poly_coeffs = np.array([
    +0.0252765416,   # x^0
    +0.7838030542,   # x^1
    +0.3405148954,   # x^2
    -0.3782324424,   # x^3
    -0.0880057821,   # x^4
    +0.1125392686,   # x^5
    +0.0464734472,   # x^6
    -0.0574030267,   # x^7
    -0.0040656245,   # x^8
    +0.0173191163,   # x^9
    -0.0009521946,   # x^10
    -0.0032787010,   # x^11
    -0.0003187983,   # x^12
    +0.0011413238,   # x^13
    -0.0002102940,   # x^14
    -0.0001636654,   # x^15
    +0.0000738152,   # x^16
    +0.0000026505,   # x^17
    -0.0000068555,   # x^18
])

def eval_poly_horner(x_val: float) -> float:
    """Horner法，纯Python实现（无numpy overhead）"""
    result = poly_coeffs[-1]
    for c in reversed(poly_coeffs[:-1]):
        result = result * x_val + c
    return result

def eval_poly_numpy(x_val: float) -> float:
    """numpy.polynomial.polyval（内部也用Horner）"""
    from numpy.polynomial.polynomial import polyval
    return polyval(x_val, poly_coeffs)

N = 100
REPEAT = 1000   # 重复多次取均值，降低计时噪声
inputs = np.random.uniform(*DOMAIN, N)

print(f"输入: {N}个随机点，每组实验重复{REPEAT}次取均值\n")

# ─── 方法1：纯Python Horner ──────────────────────────────────────────────────
times = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    for x_val in inputs:
        eval_poly_horner(float(x_val))
    times.append(time.perf_counter() - t0)
t_horner = np.mean(times)
print(f"[多项式-Horner/Python]")
print(f"  总耗时(100次): {t_horner*1e6:.2f} μs")
print(f"  单次平均:      {t_horner/N*1e9:.2f} ns\n")

# ─── 方法2：numpy polyval ────────────────────────────────────────────────────
from numpy.polynomial.polynomial import polyval
times = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    for x_val in inputs:
        polyval(float(x_val), poly_coeffs)
    times.append(time.perf_counter() - t0)
t_numpy_scalar = np.mean(times)
print(f"[多项式-numpy scalar]")
print(f"  总耗时(100次): {t_numpy_scalar*1e6:.2f} μs")
print(f"  单次平均:      {t_numpy_scalar/N*1e9:.2f} ns\n")

# ─── 方法3：MLP 逐个推理 ─────────────────────────────────────────────────────
times = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    for x_val in inputs:
        with torch.no_grad():
            model(torch.tensor([[x_val]], dtype=torch.float32))
    times.append(time.perf_counter() - t0)
t_mlp_scalar = np.mean(times)
print(f"[MLP-逐个推理]")
print(f"  总耗时(100次): {t_mlp_scalar*1e6:.2f} μs")
print(f"  单次平均:      {t_mlp_scalar/N*1e9:.2f} ns\n")

# ─── 方法4：numpy 批量（向量化）──────────────────────────────────────────────
times = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    polyval(inputs, poly_coeffs)
    times.append(time.perf_counter() - t0)
t_numpy_batch = np.mean(times)
print(f"[多项式-numpy 批量(100个)]")
print(f"  总耗时(100次): {t_numpy_batch*1e6:.2f} μs")
print(f"  单次平均:      {t_numpy_batch/N*1e9:.2f} ns\n")

# ─── 方法5：MLP 批量推理 ─────────────────────────────────────────────────────
inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
times = []
for _ in range(REPEAT):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(inputs_tensor)
    times.append(time.perf_counter() - t0)
t_mlp_batch = np.mean(times)
print(f"[MLP-批量推理(100个)]")
print(f"  总耗时(100次): {t_mlp_batch*1e6:.2f} μs")
print(f"  单次平均:      {t_mlp_batch/N*1e9:.2f} ns\n")

# ─── 汇总 ────────────────────────────────────────────────────────────────────
print("=" * 50)
print("  速度比较（单次调用，scalar）")
print("=" * 50)
print(f"  Horner/Python  : {t_horner/N*1e9:8.1f} ns")
print(f"  numpy scalar   : {t_numpy_scalar/N*1e9:8.1f} ns")
print(f"  MLP scalar     : {t_mlp_scalar/N*1e9:8.1f} ns")
print(f"  → MLP比Horner慢  {t_mlp_scalar/t_horner:.0f}x")
print()
print("  速度比较（批量100个）")
print("=" * 50)
print(f"  numpy batch    : {t_numpy_batch*1e6:8.2f} μs")
print(f"  MLP batch      : {t_mlp_batch*1e6:8.2f} μs")
print(f"  → MLP比numpy慢   {t_mlp_batch/t_numpy_batch:.1f}x")
