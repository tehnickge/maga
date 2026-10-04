import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

# Set random seed for reproducibility
np.random.seed(0)

# ==========================================
# 1. Параметры Варианта № 8
# ==========================================
n_samples = 100
random_state = 3
centers_true = [(0.2, 3), (1.0, 3)]
cluster_std = [(0.07, 0.5), (0.06, 0.5)]

# Генерация синтетических данных
X, _ = make_blobs(
    n_samples=n_samples,
    centers=centers_true,
    cluster_std=cluster_std,
    random_state=random_state,
)

# Начальная инициализация центроидов
z_init = np.array([X[1], X[3]]).copy()
max_iter = 100

# ==========================================
# 2. Эксперимент 1: Классический k-means (d_E)
# ==========================================
z1 = z_init.copy()
labels1 = np.zeros(n_samples)

for iteration in range(max_iter):
    dist0 = np.linalg.norm(X - z1[0], axis=1)
    dist1 = np.linalg.norm(X - z1[1], axis=1)
    labels1 = np.where(dist0 < dist1, 0, 1)

    new_z = np.array([X[labels1 == 0].mean(axis=0), X[labels1 == 1].mean(axis=0)])

    if np.allclose(z1, new_z, atol=1e-5):
        break
    z1 = new_z.copy()

# ==========================================
# 3. Эксперимент 2: Взвешенный k-means (d_wE)
# ==========================================
w = np.array([1.0, 0.1])
z2 = z_init.copy()
labels2 = np.zeros(n_samples)

for iteration in range(max_iter):
    dist0_w = np.sqrt(np.sum(w * ((X - z2[0]) ** 2), axis=1))
    dist1_w = np.sqrt(np.sum(w * ((X - z2[1]) ** 2), axis=1))
    labels2 = np.where(dist0_w < dist1_w, 0, 1)

    new_z = np.array([X[labels2 == 0].mean(axis=0), X[labels2 == 1].mean(axis=0)])

    if np.allclose(z2, new_z, atol=1e-5):
        break
    z2 = new_z.copy()

# ==========================================
# 4. Расчет WCSS (Внутрикластерная сумма квадратов)
# ==========================================
wcss_classic = np.sum([np.sum((X[labels1 == i] - z1[i]) ** 2) for i in range(2)])
wcss_weighted = np.sum([np.sum(w * (X[labels2 == i] - z2[i]) ** 2) for i in range(2)])

print("=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ (Вариант № 8) ===")
print(
    f"Истинные центры: Кластер 1 = {centers_true[0]}, Кластер 2 = {centers_true[1]}\n"
)
print("Эксперимент 1 (Классический k-means, d_E):")
print(f"  Найденный центр 1: x1 = {z1[0][0]:.4f}, x2 = {z1[0][1]:.4f}")
print(f"  Найденный центр 2: x1 = {z1[1][0]:.4f}, x2 = {z1[1][1]:.4f}")
print(f"  WCSS (Евклидова): {wcss_classic:.4f}\n")
print("Эксперимент 2 (Взвешенный k-means, d_wE):")
print(f"  Найденный центр 1: x1 = {z2[0][0]:.4f}, x2 = {z2[0][1]:.4f}")
print(f"  Найденный центр 2: x1 = {z2[1][0]:.4f}, x2 = {z2[1][1]:.4f}")
print(f"  WCSS (Взвешенная): {wcss_weighted:.4f}\n")

# ==========================================
# 5. Визуализация 3 графиков
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# График 1: Классический k-means
axes[0].scatter(
    X[labels1 == 0, 0], X[labels1 == 0, 1], c="purple", alpha=0.6, label="Кластер 1"
)
axes[0].scatter(
    X[labels1 == 1, 0],
    X[labels1 == 1, 1],
    c="yellow",
    edgecolor="k",
    alpha=0.6,
    label="Кластер 2",
)
axes[0].scatter(
    z1[:, 0],
    z1[:, 1],
    c="red",
    marker="X",
    s=200,
    linewidths=2,
    edgecolor="black",
    label="Центроиды $z_j$",
)
axes[0].set_title("Эксперимент 1: Классический $d_E$")
axes[0].set_xlabel("$x_1$")
axes[0].set_ylabel("$x_2$")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend()

# График 2: Взвешенный k-means
axes[1].scatter(
    X[labels2 == 0, 0], X[labels2 == 0, 1], c="purple", alpha=0.6, label="Кластер 1"
)
axes[1].scatter(
    X[labels2 == 1, 0],
    X[labels2 == 1, 1],
    c="yellow",
    edgecolor="k",
    alpha=0.6,
    label="Кластер 2",
)
axes[1].scatter(
    z2[:, 0],
    z2[:, 1],
    c="red",
    marker="X",
    s=200,
    linewidths=2,
    edgecolor="black",
    label="Центроиды $z_j$",
)
axes[1].set_title("Эксперимент 2: Взвешенный $d_{wE}$")
axes[1].set_xlabel("$x_1$")
axes[1].set_ylabel("$x_2$")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend()

# График 3: Сравнение метрик качества WCSS
methods = ["Классический\n($d_E$)", "Взвешенный\n($d_{wE}$)"]
wcss_values = [wcss_classic, wcss_weighted]
colors = ["#d9534f", "#5cb85c"]

bars = axes[2].bar(methods, wcss_values, color=colors, width=0.5, edgecolor="black")
axes[2].set_title("Сравнение ошибки WCSS")
axes[2].set_ylabel("Сумма внутрикластерных квадратов")
axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")

for bar in bars:
    yval = bar.get_height()
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{yval:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig("img/kmeans_comparison.png", dpi=300)
plt.show()
