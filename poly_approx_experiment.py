"""
实验：训练小MLP逼近sin(x)，然后用多项式一致逼近MLP
修复：Chebyshev坐标系变量替换 + 降低激活阶数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sympy as sp
from numpy.polynomial.chebyshev import chebfit, cheb2poly
from numpy.polynomial.polynomial import polyval as polyval_horner

# ─── 0. 全局设置 ───────────────────────────────────────────────────────────────
DOMAIN    = (0.0, 2.0)
N_TRAIN   = 500
HIDDEN    = 8
EPOCHS    = 5000
ACT_DEG   = 5          # tanh逼近阶数: 最终多项式阶数 = ACT_DEG^2 = 25

torch.manual_seed(42)
np.random.seed(42)

# ─── 1. 训练MLP逼近sin(x) ──────────────────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.l3 = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)

x_train = torch.linspace(*DOMAIN, N_TRAIN).unsqueeze(1)
y_train = torch.sin(x_train)

model = SmallMLP()
opt   = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    opt.zero_grad()
    nn.MSELoss()(model(x_train), y_train).backward()
    opt.step()
    if (epoch+1) % 1000 == 0:
        with torch.no_grad():
            loss = nn.MSELoss()(model(x_train), y_train).item()
        print(f"  Epoch {epoch+1}  MSE={loss:.2e}")

model.eval()
with torch.no_grad():
    y_nn = model(x_train).squeeze().numpy()
nn_linf = np.max(np.abs(y_nn - np.sin(x_train.squeeze().numpy())))
print(f"\n[MLP] L∞误差 vs sin(x): {nn_linf:.4e}\n")

# ─── 2. 收集pre-activation范围 ────────────────────────────────────────────────
hooks_data = {}
def make_hook(name):
    def hook(module, inp, out):
        hooks_data[name] = inp[0].detach().numpy()
    return hook

h1 = model.l1.register_forward_hook(make_hook('layer1'))
h2 = model.l2.register_forward_hook(make_hook('layer2'))
with torch.no_grad():
    _ = model(x_train)
h1.remove(); h2.remove()

all_pre = np.concatenate([hooks_data['layer1'].flatten(),
                          hooks_data['layer2'].flatten()])
# 加20%余量
margin = 0.2 * (all_pre.max() - all_pre.min())
a_glob = float(all_pre.min()) - margin
b_glob = float(all_pre.max()) + margin
print(f"[Pre-activation] 统一范围: [{a_glob:.3f}, {b_glob:.3f}]")

# ─── 3. 在[a,b]上对tanh做Chebyshev逼近，正确处理坐标系 ────────────────────────
print(f"\n[Chebyshev] 对tanh做{ACT_DEG}阶逼近:")

# 用Chebyshev节点（[-1,1]），映射到[a,b]
k = np.arange(ACT_DEG + 1)
t_nodes = np.cos(np.pi * (2*k + 1) / (2*(ACT_DEG + 1)))   # t ∈ [-1,1]
x_nodes = 0.5*(b_glob - a_glob)*t_nodes + 0.5*(a_glob + b_glob)  # x ∈ [a,b]
y_nodes = np.tanh(x_nodes)

# 在t空间里拟合Chebyshev级数，再转标准多项式系数（关于t的）
cheb_coeffs  = chebfit(t_nodes, y_nodes, ACT_DEG)
poly_t_coeffs = cheb2poly(cheb_coeffs)  # poly_t_coeffs[i] = t^i 的系数

# 验证（在t∈[-1,1]域上验证）
t_test = np.linspace(-1, 1, 2000)
x_test_domain = 0.5*(b_glob - a_glob)*t_test + 0.5*(a_glob + b_glob)
y_approx_t = polyval_horner(t_test, poly_t_coeffs)
y_true_t   = np.tanh(x_test_domain)
tanh_fit_err = np.max(np.abs(y_approx_t - y_true_t))
print(f"  tanh逼近误差 (L∞): {tanh_fit_err:.4e}")

# ─── 4. 符号化组合 ────────────────────────────────────────────────────────────
print(f"\n[Sympy] 符号化组合 (激活阶数={ACT_DEG}, 预计最终阶数={ACT_DEG**2})...")

W1 = model.l1.weight.detach().numpy()  # (8,1)
b1 = model.l1.bias.detach().numpy()
W2 = model.l2.weight.detach().numpy()  # (8,8)
b2 = model.l2.bias.detach().numpy()
W3 = model.l3.weight.detach().numpy()  # (1,8)
b3 = model.l3.bias.detach().numpy()

x_sym = sp.Symbol('x')

def apply_tanh_poly(expr):
    """
    把多项式激活作用到sympy表达式expr上。
    关键：先做坐标变换 t = (2*expr - (a+b))/(b-a)，再代入poly_t
    """
    t_expr = (2*expr - (a_glob + b_glob)) / (b_glob - a_glob)
    return sum(float(poly_t_coeffs[i]) * t_expr**i
               for i in range(len(poly_t_coeffs)))

# 第一层
print("  第一层: 线性 → tanh...")
h1_pre = [float(W1[i,0])*x_sym + float(b1[i]) for i in range(HIDDEN)]
h1_act = [apply_tanh_poly(h) for h in h1_pre]

# 第二层
print("  第二层: 线性 → tanh...")
h2_pre = [sum(float(W2[i,j])*h1_act[j] for j in range(HIDDEN)) + float(b2[i])
          for i in range(HIDDEN)]
h2_act = []
for i, h in enumerate(h2_pre):
    print(f"    神经元 {i+1}/{HIDDEN}...")
    h2_act.append(apply_tanh_poly(h))

# 输出层
print("  输出层: 线性...")
out_expr = sum(float(W3[0,j])*h2_act[j] for j in range(HIDDEN)) + float(b3[0])

print("  展开多项式...")
out_poly = sp.expand(out_expr)
poly_obj = sp.Poly(out_poly, x_sym)
degree   = poly_obj.degree()
print(f"\n  最终多项式阶数: {degree}")

# ─── 5. 提取系数，用Horner方法评估 ────────────────────────────────────────────
# polyval_horner (numpy.polynomial.polynomial.polyval) 用Horner法，数值稳定
np_coeffs = np.zeros(degree + 1)
for (k_,), v in poly_obj.as_dict().items():
    np_coeffs[k_] = float(v)

x_eval = np.linspace(*DOMAIN, 2000)

y_poly = polyval_horner(x_eval, np_coeffs)   # Horner法
y_nn_eval = model(torch.tensor(x_eval, dtype=torch.float32).unsqueeze(1)
                  ).detach().numpy().squeeze()
y_true_eval = np.sin(x_eval)

err_poly_vs_nn   = np.max(np.abs(y_poly - y_nn_eval))
err_poly_vs_true = np.max(np.abs(y_poly - y_true_eval))
err_nn_vs_true   = np.max(np.abs(y_nn_eval - y_true_eval))

# ─── 6. 结果输出 ───────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  误差汇总")
print("="*55)
print(f"  MLP  vs sin(x)        L∞ = {err_nn_vs_true:.4e}")
print(f"  poly vs MLP           L∞ = {err_poly_vs_nn:.4e}   ← 逼近精度")
print(f"  poly vs sin(x)        L∞ = {err_poly_vs_true:.4e}")
print("="*55)
print(f"  激活函数逼近误差        L∞ = {tanh_fit_err:.4e}")
print(f"  最终多项式阶数              {degree}")
print(f"  多项式系数数量              {degree+1}")

# ─── 7. 打印最终多项式（保留|c|>1e-6的项）────────────────────────────────────
print("\n[最终多项式（|系数| > 1e-6）]")
terms = []
for i in range(degree, -1, -1):
    c = np_coeffs[i]
    if abs(c) > 1e-6:
        if i == 0:
            terms.append(f"{c:+.6f}")
        elif i == 1:
            terms.append(f"{c:+.6f}·x")
        else:
            terms.append(f"{c:+.6f}·x^{i}")
print("p(x) = " + "\n     + ".join(t for t in terms))
