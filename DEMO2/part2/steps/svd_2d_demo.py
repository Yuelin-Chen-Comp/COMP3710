import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Make a 2D "cloud" of points that is stretched along a diagonal direction
# (simulating "height" and "weight" style correlated data)
n = 300
raw = rng.normal(size=(n, 2))
stretch = np.array([[3.0, 0.0], [0.0, 0.8]])   # stretch more along x than y
rotation_angle = np.deg2rad(30)
rot = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                 [np.sin(rotation_angle),  np.cos(rotation_angle)]])
X = raw @ stretch @ rot.T
X += np.array([5.0, 3.0])  # shift away from origin, like real data would be

# --- PCA via SVD, exactly like in the notebook ---
mean = X.mean(axis=0)
Xc = X - mean                      # step 1: center
U, S, V = np.linalg.svd(Xc, full_matrices=False)   # step 2: SVD
# V's rows are the principal directions, S tells us how much each matters

print("Data shape:", X.shape)
print("V (principal directions, one per row):\n", V)
print("S (singular values):", S)
explained_var = (S**2) / (n - 1)
print("Variance explained by each direction:", explained_var)
print("As a fraction of total variance:", explained_var / explained_var.sum())

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[:, 0], X[:, 1], alpha=0.4, s=15, label="data points (each a 2D sample)")
ax.scatter(*mean, color='black', zorder=5, label="mean")

colors = ['red', 'green']
labels = ['1st principal direction (V[0])', '2nd principal direction (V[1])']
for i in range(2):
    direction = V[i]
    length = S[i] / np.sqrt(n - 1) * 2   # scale arrow length by how much variance it explains
    ax.annotate("", xy=mean + direction * length, xytext=mean,
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.5))
    ax.plot([], [], color=colors[i], lw=2.5, label=labels[i])

ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=9)
ax.set_title("SVD / PCA on a 2D point cloud")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("svd_2d_demo.png", dpi=120)
print("saved svd_2d_demo.png")
