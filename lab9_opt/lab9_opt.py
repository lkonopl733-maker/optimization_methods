# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# 1. Постановка задачі, функції, градієнт та гесіан
def F(x):
    x1, x2 = x
    return 4*x1**2 + 2*x2**2 + 2*x1*x2 + 2*x1 - 10*x2

def grad_F(x):
    x1, x2 = x
    return np.array([8*x1 + 2*x2 + 2,
                     2*x1 + 4*x2 - 10], dtype=float)

# Гесіан H для квадратичної функції є сталим (матриця других похідних):
# d2F/dx1^2 = 8, d2F/dx1dx2 = 2, d2F/dx2dx1 = 2, d2F/dx2^2 = 4
H = np.array([[8.0, 2.0], 
              [2.0, 4.0]], dtype=float)

X_STAR = np.array([-1.0, 3.0])  # Аналітичний мінімум
F_STAR = F(X_STAR)

print("=" * 70)
print("1. АНАЛІТИЧНИЙ АНАЛІЗ ФУНКЦІЇ")
print("=" * 70)
print(f"Функція: F(x1, x2) = 4x1² + 2x2² + 2x1x2 + 2x1 − 10x2")
print(f"Градиент ∇F(x) = [8x1 + 2x2 + 2, 2x1 + 4x2 - 10]")
print(f"Гесіан H =\n{H}")
print(f"Аналітичний мінімум x* = ({X_STAR[0]}; {X_STAR[1]}), F* = {F_STAR:.4f}")

# Функція та градієнт для Розенброка (Пункт 7)
def F_rosen(x):
    x1, x2 = x
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2

def grad_Rosen(x):
    x1, x2 = x
    df_dx1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2], dtype=float)

# 2. Одновимірний пошук: Свенн + золотий переріз
def swann_bracket(phi, h0=0.01, max_exp=40):
    """Локалізація мінімуму методом Свенна з розширенням інтервалу."""
    n_calls = 0
    phi0 = phi(0.0)
    n_calls += 1

    h = h0
    phi1 = phi(h)
    n_calls += 1
    if phi1 < phi0:
        prev_h, prev_phi = 0.0, phi0
        curr_h, curr_phi = h, phi1
        for _ in range(max_exp):
            next_h = curr_h + 2*(curr_h - prev_h)
            next_phi = phi(next_h)
            n_calls += 1
            if next_phi >= curr_phi:
                return prev_h, next_h, n_calls
            prev_h, prev_phi = curr_h, curr_phi
            curr_h, curr_phi = next_h, next_phi
        return prev_h, curr_h, n_calls
    else:
        h = -h0
        phi1 = phi(h)
        n_calls += 1
        if phi1 < phi0:
            prev_h, prev_phi = 0.0, phi0
            curr_h, curr_phi = h, phi1
            for _ in range(max_exp):
                next_h = curr_h + 2*(curr_h - prev_h)
                next_phi = phi(next_h)
                n_calls += 1
                if next_phi >= curr_phi:
                    return min(prev_h, next_h), max(prev_h, next_h), n_calls
                prev_h, prev_phi = curr_h, curr_phi
                curr_h, curr_phi = next_h, next_phi
            return min(prev_h, curr_h), max(prev_h, curr_h), n_calls
        else:
            return -2*h0, 2*h0, n_calls


def golden_section(phi, a, b, tol=1e-6, max_iter=100):
    """Уточнення мінімуму золотим перерізом."""
    phi_ratio = (np.sqrt(5) - 1) / 2  
    n_calls = 0

    c = b - phi_ratio * (b - a)
    d = a + phi_ratio * (b - a)
    fc = phi(c)
    fd = phi(d)
    n_calls += 2

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi_ratio * (b - a)
            fc = phi(c)
            n_calls += 1
        else:
            a = c
            c = d
            fc = fd
            d = a + phi_ratio * (b - a)
            fd = phi(d)
            n_calls += 1

    h_star = (a + b) / 2
    f_star = phi(h_star)
    n_calls += 1
    return h_star, f_star, n_calls


def exact_line_search(func, x, d, delta0=0.01, tol=1e-6):
    """Точний лінійний пошук вздовж напрямку d."""
    def phi(h):
        return func(x + h * d)

    a, b, n_swann = swann_bracket(phi, h0=delta0)
    h_star, f_star, n_gold = golden_section(phi, a, b, tol=tol)
    return h_star, f_star, n_swann + n_gold

# Перевірка першого кроку за аналітичною формулою (Пункт 2)
print("\n" + "=" * 70)
print("ПЕРЕВІРКА ПЕРШОГО КРОКУ (Свенн + золотий переріз vs Формула)")
print("=" * 70)
x_test = np.array([-5.0, 0.0])
g_test = grad_F(x_test)
d_test = -g_test

def phi_test(h):
    return F(x_test + h * d_test)

h_num, _, _ = exact_line_search(F, x_test, d_test)
# Для квадратичної функції точний крок: h* = -<g, d> / (d^T * H * d)
h_analyt = -np.dot(g_test, d_test) / np.dot(d_test, np.dot(H, d_test))

print(f"Числовий крок (Свенн + Золотий переріз): h* = {h_num:.6f}")
print(f"Аналітичний крок для параболи:          h* = {h_analyt:.6f}")


# 3, 4, 5. Методи оптимізації (Флетчер–Рівс із перезапуском, Антиградієнт)
def conjugate_gradients(func, grad_func, x0, eps=0.01, max_iter=100, 
                        delta0=0.01, tol_1d=1e-6, restart_every=None, use_fletcher_reeves=True):
    x = np.array(x0, dtype=float)
    g = grad_func(x)
    d = -g.copy()

    history = []
    nF = 1  
    nG = 1  

    fx = func(x)
    history.append({
        'k': 0,
        'x1': x[0], 'x2': x[1], 'F': fx,
        'g1': g[0], 'g2': g[1], 'norm_g': np.linalg.norm(g),
        'd1': d[0], 'd2': d[1],
        'beta': 0.0, 'h': 0.0,
        'nF_step': 0
    })

    g_prev = g.copy()
    d_prev = d.copy()

    for k in range(1, max_iter + 1):
        norm_g = np.linalg.norm(g)
        if norm_g <= eps:
            break

        h, fx_new, nF_ls = exact_line_search(func, x, d, delta0=delta0, tol=tol_1d)
        nF += nF_ls

        x = x + h * d
        fx = fx_new

        g = grad_func(x)
        nG += 1
        norm_g = np.linalg.norm(g)

        if use_fletcher_reeves:
            denominator = np.dot(g_prev, g_prev)
            if denominator < 1e-12:
                beta = 0.0
            else:
                beta = (np.dot(g, g)) / denominator
        else:
            beta = 0.0  # Чистий антиградієнт із точним пошуком

        if restart_every is not None and k % restart_every == 0:
            beta = 0.0

        d = -g + beta * d_prev

        history.append({
            'k': k,
            'x1': x[0], 'x2': x[1], 'F': fx,
            'g1': g[0], 'g2': g[1], 'norm_g': norm_g,
            'd1': d[0], 'd2': d[1],
            'beta': beta, 'h': h,
            'nF_step': nF_ls
        })

        g_prev = g.copy()
        d_prev = d.copy()

        if norm_g <= eps:
            break

    return {
        'x': x,
        'F': fx,
        'n': len(history) - 1,
        'N_F': nF,
        'N_grad': nG,
        'history': history
    }


# Запуск методів для квадратичної функції (Пункт 3 та 5)
x0 = np.array([-5.0, 0.0])
eps = 0.01

# Флетчер-Рівс із перезапуском кожні n=2 кроки (пункт 3)
res_fr = conjugate_gradients(F, grad_F, x0, eps=eps, restart_every=2, use_fletcher_reeves=True)
# Антиградієнт + точний пошук (β = 0 завжди,пункт 5)
res_ag = conjugate_gradients(F, grad_F, x0, eps=eps, use_fletcher_reeves=False)


# 4. Виконання трьох обов'язкових математичних перевірок 
print("\n" + "=" * 70)
print("4. ТРИ МАТЕМАТИЧНІ ПЕРЕВІРКИ (після першої ітерації)")
print("=" * 70)
hist_fr = res_fr['history']
if len(hist_fr) > 1:
    #точки та вектори першої ітерації
    # x(0) -> x(1)
    g0 = grad_F(np.array([hist_fr[0]['x1'], hist_fr[0]['x2']]))
    d0 = np.array([hist_fr[0]['d1'], hist_fr[0]['d2']])
    
    g1 = grad_F(np.array([hist_fr[1]['x1'], hist_fr[1]['x2']]))
    d1 = np.array([hist_fr[1]['d1'], hist_fr[1]['d2']])

    check1 = np.dot(g1, d0)
    check2 = np.dot(d1, np.dot(H, d0))
    check3 = np.dot(g1, d1)

    print(1, f"1. Перпендикулярність ⟨∇F(x(1)), d(0)⟩ = {check1:.2e} (має бути ≈ 0)")
    print(2, f"2. Спряженість d(1)^T · H · d(0) = {check2:.2e} (має бути ≈ 0)")
    print(3, f"3. Напрямок спуску ⟨∇F(x(1)), d(1)⟩ = {check3:.4f} (має бути < 0)")


# Таблиця ітерацій для Флетчера-Рівса (Пункт 3)
print("\n" + "=" * 90)
print("ТАБЛИЦЯ ІТЕРАЦІЙ — МЕТОД ФЛЕТЧЕРА–РІВСА (перезапуск кожні n=2)")
print("=" * 90)
print(f"{'k':>3} | {'x1':>11} | {'x2':>11} | {'F':>14} | {'‖g‖₂':>10} | {'β':>10} | {'h':>11} | {'N_F кроку':>9}")
print("-" * 90)
for rec in hist_fr:
    if rec['k'] == 0:
        print(f"{0:3d} | {rec['x1']:11.6f} | {rec['x2']:11.6f} | {rec['F']:14.6f} | "
              f"{rec['norm_g']:10.6f} | {'—':>10} | {'—':>11} | {0:9d}")
    else:
        print(f"{rec['k']:3d} | {rec['x1']:11.6f} | {rec['x2']:11.6f} | {rec['F']:14.6f} | "
              f"{rec['norm_g']:10.6f} | {rec['beta']:10.6f} | {rec['h']:11.6f} | {rec['nF_step']:9d}")


# Метод сферичної метрики з ЛР 8 
def gradient_descent_sphere_lr8(h0=1.0, eps=0.01, max_iter=300):
    x = np.array([-5.0, 0.0], dtype=float)
    h = float(h0)
    history = [{'k': 0, 'x1': x[0], 'x2': x[1], 'F': F(x)}]
    nF, nG = 1, 0
    for k in range(1, max_iter + 1):
        g = grad_F(x)
        nG += 1
        if np.linalg.norm(g) <= eps:
            break
        d = -g / np.linalg.norm(g)
        
        h_try = h
        accepted = False
        fx = F(x)
        for _ in range(30):
            x_bar = x + h_try * d
            f_bar = F(x_bar)
            nF += 1
            if f_bar < fx:
                x = x_bar
                h = h_try
                accepted = True
                break
            else:
                h_try /= 2.0
        if not accepted:
            break
        history.append({'k': k, 'x1': x[0], 'x2': x[1], 'F': F(x)})
    return history, nF, nG, len(history)-1

hist_lr8, nF_lr8, nG_lr8, n_lr8 = gradient_descent_sphere_lr8()

# 7. Дослідження функції Розенброка
print("\n" + "=" * 70)
print("7. ДОСЛІДЖЕННЯ ФУНКЦІЇ РОЗЕНБРОКА")
print("=" * 70)
x0_rosen = np.array([-1.2, 1.0])

res_rosen_restarts   = conjugate_gradients(F_rosen, grad_Rosen, x0_rosen, eps=0.01, max_iter=500, restart_every=2, use_fletcher_reeves=True)
res_rosen_no_restart = conjugate_gradients(F_rosen, grad_Rosen, x0_rosen, eps=0.01, max_iter=500, restart_every=None, use_fletcher_reeves=True)
res_rosen_ag         = conjugate_gradients(F_rosen, grad_Rosen, x0_rosen, eps=0.01, max_iter=500, use_fletcher_reeves=False)

print(f"Флетчер–Рівс (з перезапуском кожні n=2): ітерацій = {res_rosen_restarts['n']}, N_F = {res_rosen_restarts['N_F']}, N_∇F = {res_rosen_restarts['N_grad']}")
print(f"Флетчер–Рівс (без перезапуску):            ітерацій = {res_rosen_no_restart['n']}, N_F = {res_rosen_no_restart['N_F']}, N_∇F = {res_rosen_no_restart['N_grad']}")
print(f"Чистий антиградієнт (β=0):                 ітерацій = {res_rosen_ag['n']}, N_F = {res_rosen_ag['N_F']}, N_∇F = {res_rosen_ag['N_grad']}")


# 8. Зведена таблиця результатів (Пункт 8)
print("\n" + "=" * 80)
print("8. ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
print("=" * 80)
print(f"{'Метод':<50} | {'n':>4} | {'N_F':>5} | {'N_∇F':>5} | {'Разом':>6}")
print("-" * 80)

print(f"{'Основна функція (квадратична)':<50}")
print(f"{'  Сферична метрика (ЛР 8)':<50} | {n_lr8:4d} | {nF_lr8:5d} | {nG_lr8:5d} | {nF_lr8+nG_lr8:6d}")
print(f"{'  Антиградієнт + точний пошук':<50} | {res_ag['n']:4d} | {res_ag['N_F']:5d} | {res_ag['N_grad']:5d} | {res_ag['N_F']+res_ag['N_grad']:6d}")
print(f"{'  Флетчер–Рівс (перезапуск n=2)':<50} | {res_fr['n']:4d} | {res_fr['N_F']:5d} | {res_fr['N_grad']:5d} | {res_fr['N_F']+res_fr['N_grad']:6d}")

print()
print(f"{'Функція Розенброка':<50}")
print(f"{'  Флетчер–Рівс (з перезапуском n=2)':<50} | {res_rosen_restarts['n']:4d} | {res_rosen_restarts['N_F']:5d} | {res_rosen_restarts['N_grad']:5d} | {res_rosen_restarts['N_F']+res_rosen_restarts['N_grad']:6d}")
print(f"{'  Флетчер–Рівс (без перезапуску)':<50} | {res_rosen_no_restart['n']:4d} | {res_rosen_no_restart['N_F']:5d} | {res_rosen_no_restart['N_grad']:5d} | {res_rosen_no_restart['N_F']+res_rosen_no_restart['N_grad']:6d}")
print(f"{'  Чистий антиградієнт (β=0)':<50} | {res_rosen_ag['n']:4d} | {res_rosen_ag['N_F']:5d} | {res_rosen_ag['N_grad']:5d} | {res_rosen_ag['N_F']+res_rosen_ag['N_grad']:6d}")
print("-" * 80)


# 6. Графік із ТРЬОМА траєкторіями (Пункт 6)
def plot_three_trajectories():
    fig, ax = plt.subplots(figsize=(10, 8))

    x1g = np.linspace(-6.5, 1.5, 400)
    x2g = np.linspace(-1.5, 5.5, 400)
    X1, X2 = np.meshgrid(x1g, x2g)
    Z = 4*X1**2 + 2*X2**2 + 2*X1*X2 + 2*X1 - 10*X2

    levels = [-18, -15, -10, -5, 0, 10, 30, 60, 100, 160, 250]
    cs = ax.contour(X1, X2, Z, levels=levels, cmap='viridis', linewidths=1.2)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%g')

    # 1. Сферична метрика з ЛР 8
    xs_lr8 = [r['x1'] for r in hist_lr8]
    ys_lr8 = [r['x2'] for r in hist_lr8]
    ax.plot(xs_lr8, ys_lr8, 'o-', color='#1f77b4', markersize=5, linewidth=1.5, label='Сферична метрика (ЛР 8)')

    # 2. Антиградієнт + точний пошук
    hist_ag = res_ag['history']
    xs_ag = [r['x1'] for r in hist_ag]
    ys_ag = [r['x2'] for r in hist_ag]
    ax.plot(xs_ag, ys_ag, 's--', color='#2ca02c', markersize=6, linewidth=1.8, label='Антиградієнт + точний пошук')

    # 3. Флетчер–Рівс
    xs_fr = [r['x1'] for r in hist_fr]
    ys_fr = [r['x2'] for r in hist_fr]
    ax.plot(xs_fr, ys_fr, '^-', color='#d62728', markersize=8, linewidth=2.2, label='Флетчер–Рівс (ЛР 9)')

    ax.plot(x0[0], x0[1], 'ko', markersize=10, label='Старт (−5; 0)')
    ax.plot(X_STAR[0], X_STAR[1], 'r*', markersize=20, label=f'Точний мін. (−1; 3)')

    ax.set_xlabel(r'$x_1$', fontsize=12)
    ax.set_ylabel(r'$x_2$', fontsize=12)
    ax.set_title('Порівняння трьох траєкторій градієнтних методів', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

plot_three_trajectories()

