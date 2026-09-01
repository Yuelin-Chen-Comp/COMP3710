import numpy as np
import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存成图片文件
import matplotlib.pyplot as plt

# --- 基本参数 ---
N = 2048     # 采样点数（这段时间里取多少个点）
T = 1.0      # 信号总长度是 1 秒
f0 = 1       # 基础频率 1 Hz（1秒内正好震荡1个周期）

# 时间轴：从 0 到 1 秒，均匀取 2048 个点
t = np.linspace(0.0, T, N, endpoint=False)

# --- 真正的方波（目标）---
square = np.sign(np.sin(2.0 * np.pi * f0 * t))

# --- 只用 n=1 这一项去逼近 ---
def one_harmonic(t, f0, n):
    return (4 / np.pi) * np.sin(2 * np.pi * n * f0 * t) / n

y1 = one_harmonic(t, f0, 1)

plt.figure(figsize=(8, 4))
plt.plot(t, square, 'k--', alpha=0.5, label="target square wave")
plt.plot(t, y1, label="n=1 only (pure sine)")
plt.title("Approximating a square wave with just 1 sine wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("step1_single_sine.png", dpi=120)
print("saved step1_single_sine.png")
