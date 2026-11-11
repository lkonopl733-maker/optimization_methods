import matplotlib.pyplot as plt
import numpy as np


# Глобальний лічильник звернень до функції F
class Counter:

  def __init__(self):
    self.nf = 0

  def reset(self):
    self.nf = 0


counter = Counter()


#цільова функція
def F_variant(x):
  counter.nf += 1
  return 4 * x[0] ** 2 + 2 * x[1] ** 2 + 2 * x[0] * x[1] + 2 * x[0] - 10 * x[1]


# Негладка функція 
def G_nonsmooth(x):
  counter.nf += 1
  return (
      2 * abs(x[0] - 1)
      + 3 * abs(x[1] + 2)
      + 0.5 * (x[0] - 1) ** 2
      + 0.5 * (x[1] + 2) ** 2
  )


# Функція Розенброка 
def Rosenbrock(x):
  counter.nf += 1
  return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


#Дослідний пошук 
def explore(F, y, Fy, d):
  z = np.copy(y)
  Fz = Fy
  for i in range(len(z)):
    # Проба "плюс"
    t = np.copy(z)
    t[i] += d
    Ft = F(t)
    if Ft < Fz:
      z, Fz = t, Ft
      continue
    # Проба "мінус"
    t = np.copy(z)
    t[i] -= d
    Ft = F(t)
    if Ft < Fz:
      z, Fz = t, Ft
  return z, Fz


# 3. Метод Хука-Дживса
def hooke_jeeves(F, x0, d0=1.0, eps=0.01, max_iter=500):
  base = np.array(x0, dtype=float)
  Fb = F(base)
  y = np.copy(base)
  Fy = Fb
  d = d0
  from_pattern = False

  history = []
  history.append(
      {
          "k": 0,
          "action": "Старт",
          "from": list(base),
          "F": Fb,
          "delta": d,
          "to": list(base),
          "NF": counter.nf,
      }
  )

  k = 0
  while d >= eps and k < max_iter:
    z, Fz = explore(F, y, Fy, d)

    if Fz < Fb:
      # Успіх: нова база і зразок
      k += 1
      next_y = z + (z - base)
      history.append(
          {
              "k": k,
              "action": "Успіх / Зразок",
              "from": list(y),
              "F": Fz,
              "delta": d,
              "to": list(next_y),
              "NF": counter.nf,
          }
      )
      base, Fb = np.copy(z), Fz
      y = np.copy(next_y)
      Fy = F(y)
      from_pattern = True
    elif from_pattern:
      # Зразок не спрацював -> повертаємося в базу
      history.append(
          {
              "k": k,
              "action": "Повернення в базу",
              "from": list(y),
              "F": Fy,
              "delta": d,
              "to": list(base),
              "NF": counter.nf,
          }
      )
      y = np.copy(base)
      Fy = Fb
      from_pattern = False
    else:
      # Невдача з бази -> дроблення кроку
      d = d / 2.0
      history.append(
          {
              "k": k,
              "action": f"Дроблення дільника до {d:.4f}",
              "from": list(base),
              "F": Fb,
              "delta": d,
              "to": list(base),
              "NF": counter.nf,
          }
      )
      y = np.copy(base)
      Fy = Fb

  return base, Fb, history


#ВИКОНАННЯ ОБЧИСЛЕНЬ
print("=== ЛАБОРАТОРНА РОБОТА № 10: МЕТОД ХУКА-ДЖИВСА ===\n")

# 1. Основний прогін: d0 = 1.0, eps = 0.01
counter.reset()
x0 = [-5.0, 0.0]
res1, f1, hist1 = hooke_jeeves(F_variant, x0, d0=1.0, eps=0.01)

print(f"1. Основний прогін (Δ0 = 1.0, ε = 0.01):")
print(
    f"   Знайдений мінімум x* = ({res1[0]:.4f}, {res1[1]:.4f}), F(x*) ="
    f" {f1:.6f}"
)
print(f"   Загальна кількість звернень NF = {counter.nf}\n")

# 2. Другий прогін: d0 = 0.7, eps = 0.01
counter.reset()
res2, f2, hist2 = hooke_jeeves(F_variant, x0, d0=0.7, eps=0.01)

print(f"2. Другий прогін (Δ0 = 0.7, ε = 0.01):")
print(
    f"   Знайдений мінімум x* = ({res2[0]:.4f}, {res2[1]:.4f}), F(x*) ="
    f" {f2:.6f}"
)
print(f"   Загальна кількість звернень NF = {counter.nf}\n")

# 3. Негладка функція G
counter.reset()
res_g, fg, hist_g = hooke_jeeves(G_nonsmooth, [5.0, -3.0], d0=1.0, eps=0.01)
print(f"3. Негладка функція G:")
print(
    f"   Знайдений мінімум x* = ({res_g[0]:.4f}, {res_g[1]:.4f}), G(x*) ="
    f" {fg:.6f}, NF = {counter.nf}\n"
)

# 4. Розенброк (ε = 0.01 та ε = 10^-4)
counter.reset()
res_r1, fr1, _ = hooke_jeeves(Rosenbrock, [-1.2, 1.0], d0=1.0, eps=0.01)
nf_r1 = counter.nf

counter.reset()
res_r2, fr2, _ = hooke_jeeves(Rosenbrock, [-1.2, 1.0], d0=1.0, eps=1e-4)
nf_r2 = counter.nf

print(f"4. Функція Розенброка (пастка критерію зупинки):")
print(
    f"   При ε = 0.01: x* = ({res_r1[0]:.4f}, {res_r1[1]:.4f}), f = {fr1:.4f}, NF"
    f" = {nf_r1} (Пастка!)"
)
print(
    f"   При ε = 10^-4: x* = ({res_r2[0]:.4f}, {res_r2[1]:.4f}), f = {fr2:.6f},"
    f" NF = {nf_r2}\n"
)

#ЗВЕДЕНА ТАБЛИЦЯ
print("=" * 85)
print(
    f"{'Метод / Задача':<35} | {'Старт':<10} | {'Δ0':<5} | {'ε':<8} |"
    f" {'Знайдено x*':<15} | {'F(x*)':<10} | {'NF':<5}"
)
print("-" * 85)
print(
    f"{'Хук-Дживс (Варіант, Δ0=1)':<35} | {str(x0):<10} | {1.0:<5} |"
    f" {0.01:<8} | {f'({res1[0]:.2f}, {res1[1]:.2f})':<15} | {f'{f1:.4f}':<10} |"
    f" {hist1[-1]['NF']:<5}"
)
print(
    f"{'Хук-Дживс (Варіант, Δ0=0.7)':<35} | {str(x0):<10} | {0.7:<5} |"
    f" {0.01:<8} | {f'({res2[0]:.2f}, {res2[1]:.2f})':<15} | {f'{f2:.4f}':<10} |"
    f" {hist2[-1]['NF']:<5}"
)
print(
    f"{'Хук-Дживс (Негладка G)':<35} | {'[5, -3]':<10} | {1.0:<5} |"
    f" {0.01:<8} | {f'({res_g[0]:.2f}, {res_g[1]:.2f})':<15} |"
    f" {f'{fg:.4f}':<10} | {hist_g[-1]['NF']:<5}"
)
print(
    f"{'Розенброк (ε=0.01)':<35} | {'[-1.2, 1]':<10} | {1.0:<5} |"
    f" {0.01:<8} | {f'({res_r1[0]:.2f}, {res_r1[1]:.2f})':<15} |"
    f" {f'{fr1:.4f}':<10} | {nf_r1:<5}"
)
print(
    f"{'Розенброк (ε=10^-4)':<35} | {'[-1.2, 1]':<10} | {1.0:<5} |"
    f" {1e-4:<8} | {f'({res_r2[0]:.2f}, {res_r2[1]:.2f})':<15} |"
    f" {f'{fr2:.4e}':<10} | {nf_r2:<5}"
)
print("=" * 85 + "\n")


#ПУБДОВА ГРАФІКІВ ЛІНІЙ РІВНЯ ТА ТРАЄКТОРІЙ 
def plot_contour_and_trajectories():
  # Створення сітки координат для ліній рівня
  x1_vals = np.linspace(-6, 1, 400)
  x2_vals = np.linspace(-1, 4, 400)
  X1, X2 = np.meshgrid(x1_vals, x2_vals)

  # Обчислення значень функції на сітці
  Z = 4 * X1**2 + 2 * X2**2 + 2 * X1 * X2 + 2 * X1 - 10 * X2

  plt.figure(figsize=(10, 8))

  # Малювання ліній рівня
  contours = plt.contour(X1, X2, Z, levels=30, cmap="viridis", alpha=0.6)
  plt.clabel(contours, inline=True, fontsize=8)

  # Витягування координат базових точок із історії прогонів
  traj1 = np.array(
      [h["from"] for h in hist1 if h["action"] in ["Старт", "Успіх / Зразок"]]
      + [res1]
  )
  traj2 = np.array(
      [h["from"] for h in hist2 if h["action"] in ["Старт", "Успіх / Зразок"]]
      + [res2]
  )

  # Траєкторія 1: Δ0 = 1.0
  plt.plot(
      traj1[:, 0],
      traj1[:, 1],
      "ro-",
      linewidth=2,
      markersize=7,
      label="Хук-Дживс (Δ₀ = 1.0)",
  )

  # Траєкторія 2: Δ0 = 0.7
  plt.plot(
      traj2[:, 0],
      traj2[:, 1],
      "bs--",
      linewidth=1.5,
      markersize=5,
      label="Хук-Дживс (Δ₀ = 0.7)",
  )

  # Позначення точок старту та аналітичного мінімуму (-1, 3)
  plt.plot(x0[0], x0[1], "ko", markersize=9, label="Старт (-5, 0)")
  plt.plot(
      -1.0, 3.0, "g*", markersize=14, label="Аналітичний мінімум (-1, 3)"
  )

  plt.title("Траєкторія руху методу Хука-Дживса на лініях рівня F(x₁, x₂)")
  plt.xlabel("x₁")
  plt.ylabel("x₂")
  plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
  plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(loc="upper right")
  plt.tight_layout()
  plt.show()


# Запуск побудови графіка
plot_contour_and_trajectories()