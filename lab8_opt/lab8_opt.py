# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


#Функція та градієнт
def F(x):
    x1, x2 = x
    return 4*x1**2 + 2*x2**2 + 2*x1*x2 + 2*x1 - 10*x2

def grad_F(x):
    x1, x2 = x
    return np.array([8*x1 + 2*x2 + 2,
                     2*x1 + 4*x2 - 10], dtype=float)

X_STAR = np.array([-1.0, 3.0])   # точний мінімум
F_STAR = F(X_STAR)               

# Вибір напрямку за метрикою
def get_direction(g, metric):
    if metric == "sphere":
        ng = np.linalg.norm(g, 2)
        return np.zeros_like(g) if ng == 0 else -g / ng
    elif metric == "cube":
        return -np.sign(g)
    elif metric == "octahedron":
        j = np.argmax(np.abs(g))
        d = np.zeros_like(g)
        d[j] = -np.sign(g[j])
        return d
    else:
        raise ValueError("Невідома метрика")


#Градієнтний метод з дробленням кроку
def gradient_descent(metric, h0=1.0, eps=0.01, max_iter=300, h_min=1e-12):
    x = np.array([-5.0, 0.0], dtype=float)
    h = float(h0)

    history = []
    nF = 1          # початкове значення F(x0)
    nG = 0
    total_halves = 0

    fx = F(x)
    history.append({
        'k': 0, 'x1': x[0], 'x2': x[1], 'F': fx,
        'g1': None, 'g2': None, 'norm_g': None,
        'd1': None, 'd2': None, 'h': h, 'halves': 0
    })

    for k in range(1, max_iter + 1):
        g = grad_F(x)
        nG += 1
        norm_g = np.linalg.norm(g, 2)

        # Критерій зупинки 
        if norm_g <= eps:
            break

        d = get_direction(g, metric)

        # Напрямок має бути спусковим
        if np.dot(g, d) >= 0:
            break

        # Дроблення кроку (h НЕ повертається до h0)
        halves = 0
        h_try = h
        accepted = False

        while h_try >= h_min:
            x_bar = x + h_try * d
            f_bar = F(x_bar)
            nF += 1

            if f_bar < fx:
                x = x_bar
                fx = f_bar
                h = h_try          # залишаємо поточний (можливо зменшений) крок
                total_halves += halves
                accepted = True
                break
            else:
                h_try /= 2.0
                halves += 1

        if not accepted:
            break   # крок став занадто малим

        history.append({
            'k': k,
            'x1': x[0], 'x2': x[1], 'F': fx,
            'g1': g[0], 'g2': g[1], 'norm_g': norm_g,
            'd1': d[0], 'd2': d[1],
            'h': h, 'halves': halves
        })

    # Фінальна перевірка градієнта 
    g_final = grad_F(x)
    nG += 1

    return x, fx, len(history)-1, nF, nG, history, total_halves

#Запуск для двох початкових кроків
metrics = ['sphere', 'cube', 'octahedron']
metric_ua = {
    'sphere': 'Сферична (L₂)',
    'cube':   'Кубічна (L_∞)',
    'octahedron': 'Октаедрична (L₁)'
}
h_values = [1.0, 0.7]
results = {}

for h0 in h_values:
    print(f"\n{'='*70}")
    print(f"  ПОЧАТКОВИЙ КРОК h₀ = {h0}")
    print(f"{'='*70}")
    results[h0] = {}

    for m in metrics:
        x_res, f_res, n, nF, nG, hist, halves = gradient_descent(m, h0=h0)
        results[h0][m] = {
            'x': x_res, 'F': f_res, 'n': n,
            'N_F': nF, 'N_grad': nG,
            'history': hist, 'halves': halves
        }

        print(f"\n▶ {metric_ua[m]}")
        print(f"  n (ітерацій)      = {n}")
        print(f"  N_F               = {nF}")
        print(f"  N_∇F              = {nG}")
        print(f"  x*                = ({x_res[0]:.6f}; {x_res[1]:.6f})")
        print(f"  F(x*)             = {f_res:.6f}")
        print(f"  ‖x* − x_точн‖₂   = {np.linalg.norm(x_res - X_STAR):.6e}")
        print(f"  кількість дроблень = {halves}")


#Таблиці ітерацій (для h₀ = 1.0)
def print_iteration_table(hist, metric_name):
    print(f"\n{'─'*90}")
    print(f"Таблиця ітерацій — {metric_name}")
    print(f"{'─'*90}")
    print(f"{'k':>3} | {'x1':>10} | {'x2':>10} | {'F':>12} | "
          f"{'g1':>9} | {'g2':>9} | {'‖g‖₂':>8} | "
          f"{'d1':>8} | {'d2':>8} | {'h':>8} | {'др.':>3}")
    print(f"{'-'*90}")

    for rec in hist:
        if rec['k'] == 0:
            print(f"{0:3d} | {rec['x1']:10.6f} | {rec['x2']:10.6f} | {rec['F']:12.6f} | "
                  f"{'—':>9} | {'—':>9} | {'—':>8} | "
                  f"{'—':>8} | {'—':>8} | {rec['h']:8.4g} | {0:3d}")
        else:
            print(f"{rec['k']:3d} | {rec['x1']:10.6f} | {rec['x2']:10.6f} | {rec['F']:12.6f} | "
                  f"{rec['g1']:9.4f} | {rec['g2']:9.4f} | {rec['norm_g']:8.4f} | "
                  f"{rec['d1']:8.4f} | {rec['d2']:8.4f} | {rec['h']:8.4g} | {rec['halves']:3d}")

print("\n" + "="*70)
print("ТАБЛИЦІ ІТЕРАЦІЙ (h₀ = 1.0)")
print("="*70)
for m in metrics:
    print_iteration_table(results[1.0][m]['history'], metric_ua[m])


#Зведена таблиця
print("\n" + "="*70)
print("ЗВЕДЕНА ТАБЛИЦЯ  (n, N_F, N_∇F)")
print("="*70)

for h0 in h_values:
    print(f"\nh₀ = {h0}")
    print(f"{'Метрика':<22} | {'n':>4} | {'N_F':>5} | {'N_∇F':>5} | {'Разом':>6}")
    print("-"*55)
    for m in metrics:
        r = results[h0][m]
        total = r['N_F'] + r['N_grad']
        print(f"{metric_ua[m]:<22} | {r['n']:4d} | {r['N_F']:5d} | {r['N_grad']:5d} | {total:6d}")


#Графіки ліній рівня + траєкторії
def plot_trajectories(h0, results_dict):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Лінії рівня
    x1g = np.linspace(-6.5, 1.5, 400)
    x2g = np.linspace(-1.5, 5.5, 400)
    X1, X2 = np.meshgrid(x1g, x2g)
    Z = 4*X1**2 + 2*X2**2 + 2*X1*X2 + 2*X1 - 10*X2

    levels = [-16, -10, -5, 0, 10, 30, 60, 100, 160, 250]
    cs = ax.contour(X1, X2, Z, levels=levels, cmap='viridis', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%g')

    colors  = {'sphere': '#1f77b4', 'cube': '#ff7f0e', 'octahedron': '#2ca02c'}
    markers = {'sphere': 'o',       'cube': 's',       'octahedron': '^'}

    for m in metrics:
        hist = results_dict[m]['history']
        xs = [r['x1'] for r in hist]
        ys = [r['x2'] for r in hist]
        ax.plot(xs, ys, marker=markers[m], markersize=5,
                color=colors[m], linewidth=1.8, label=metric_ua[m])
        ax.plot(xs[0], ys[0], 'ko', markersize=9)                     # старт
        ax.plot(xs[-1], ys[-1], color=colors[m],
                marker='*', markersize=14, markeredgecolor='k')       # фініш

    ax.plot(X_STAR[0], X_STAR[1], 'r*', markersize=18,
            label=f'Точний мін. (−1; 3), F = {F_STAR}')
    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title(f'Лінії рівня та траєкторії градієнтних методів  (h₀ = {h0})',
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

plot_trajectories(1.0, results[1.0])
plot_trajectories(0.7, results[0.7])