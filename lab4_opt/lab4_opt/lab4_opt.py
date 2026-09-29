import math
import matplotlib.pyplot as plt

# Цільова функція
def f(x):
    return 0.5 * (x**2 + 1) * math.atan(x) - 1.5 * x

# Точний мінімум (для похибки)
X_EXACT = 1.1623398491423087

# свенн
def swann(x0, delta0=0.1):
    print("*" * 90)
    print("Метод Свенна")
    print("*" * 90)
    print(f"x0 = {x0}")
    print(f"дельта0 = {delta0}\n")

    f0 = f(x0)
    f_right = f(x0 + delta0)
    points = [x0]
    values = [f0]

    if f_right < f0:
        delta = delta0
        points.append(x0 + delta)
        values.append(f_right)
        print("Напрямок пошуку: вправо (дельта > 0)")
    else:
        f_left = f(x0 - delta0)
        if f_left < f0:
            delta = -delta0
            points.append(x0 + delta)
            values.append(f_left)
            print("Напрямок пошуку: вліво (дельта < 0)")
        else:
            a0 = x0 - abs(delta0)
            b0 = x0 + abs(delta0)
            print("Мінімум знаходиться в початковому околі x0.")
            print(f"[a0, b0] = [{a0:.6f}; {b0:.6f}]")
            return a0, b0, [a0, x0, b0], 3

    print("\nТаблиця ітерацій:")
    print("*" * 75)
    print(f"{'k':>3} | {'Крок (2^k * дельта)':>20} | {'xk':>12} | {'f(xk)':>14} | {'f(xk) < f(xk-1)':>16}")
    print("*" * 75)
    print(f"{0:>3} | {'—':>20} | {points[0]:>12.6f} | {values[0]:>14.8f} | {'—':>16}")
    print(f"{1:>3} | {delta:>20.6f} | {points[1]:>12.6f} | {values[1]:>14.8f} | {'+':>16}")

    k = 1
    while True:
        step = (2 ** k) * delta
        x_next = points[-1] + step
        f_next = f(x_next)
        points.append(x_next)
        values.append(f_next)
        k += 1
        is_decreasing = f_next < values[-2]
        comparison = "+" if is_decreasing else "−"
        print(f"{k:>3} | {step:>20.6f} | {x_next:>12.6f} | {f_next:>14.8f} | {comparison:>16}")
        if not is_decreasing:
            break

    a0 = min(points[-3], points[-1])
    b0 = max(points[-3], points[-1])
    calc_count = len(values)
    print("*" * 75)
    print(f"\nІнтервал локалізації [a0, b0] = [{a0:.6f}; {b0:.6f}]")
    print(f"Довжина інтервалу L0 = {b0 - a0:.6f}")
    print(f"Всього обчислень f(x): {calc_count}\n")
    return a0, b0, points, calc_count

# Метод Фібоначчі
def fibonacci_search(a0, b0, eps=0.01):
    print("*" * 135)
    print("Метод чисел Фібоначчі (ЛР 4)")
    print("*" * 135)

    L0 = b0 - a0
    delta = eps / 10.0
    R = L0 / (eps - delta)

    F = [0, 1, 1]
    while F[-1] < R:
        F.append(F[-1] + F[-2])
    N = len(F) - 1

    print(f"Початковий інтервал: [{a0:.6f}; {b0:.6f}], L0 = {L0:.6f}")
    print(f"ε = {eps}, R = {R:.4f}")
    print(f"N = {N} (F_N = {F[N]} ≥ {R:.4f})")
    print(f"Планована кількість обчислень: N-1 = {N-1}")
    print(f"Теоретична межа Ln ≤ L0/F_N + δ = {L0/F[N] + delta:.8f}\n")

    print("Таблиця пошуку N:")
    print(f"{'N':>4} | {'FN':>8} | {'FN ≥ R?':>10} | {'L0/FN':>12}")
    print("-" * 45)
    for i in range(1, N + 1):
        flag = "ТАК ✓" if F[i] >= R else "ні"
        print(f"{i:>4} | {F[i]:>8} | {flag:>10} | {L0/F[i]:>12.6f}")
    print()

    a, b = float(a0), float(b0)
    points = []

    # Перші дві точки
    m = N
    x1 = a + F[m - 2] / F[m] * (b - a)
    x2 = a + F[m - 1] / F[m] * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    calc_count = 2
    points.extend([x1, x2])

    print("Таблиця ітерацій:")
    print("*" * 140)
    header = (
        f"{'k':>3} | {'m':>3} | {'Fm-2/Fm':>10} | {'Fm-1/Fm':>10} | "
        f"{'ak':>10} | {'bk':>10} | {'x1':>10} | {'x2':>10} | "
        f"{'f(x1)':>12} | {'f(x2)':>12} | {'Lk':>9} | {'Рішення':>22}"
    )
    print(header)
    print("*" * 140)

    for k in range(N - 2):
        m = N - k
        Lk = b - a
        frac1 = F[m - 2] / F[m]
        frac2 = F[m - 1] / F[m]

        if f1 <= f2:
            decision = "f1 ≤ f2 → b = x2"
            print(
                f"{k:>3} | {m:>3} | {frac1:>10.6f} | {frac2:>10.6f} | "
                f"{a:>10.6f} | {b:>10.6f} | {x1:>10.6f} | {x2:>10.6f} | "
                f"{f1:>12.8f} | {f2:>12.8f} | {Lk:>9.6f} | {decision:>22}"
            )
            b = x2
            x2, f2 = x1, f1 # успадкування

            if k < N - 3:
                m_next = N - (k + 1)
                if m_next == 3:
                    # залишається успадкована x2, нову x1 зі зсувом
                    x1 = x2 - delta
                    f1 = f(x1)
                    calc_count += 1
                    points.append(x1)
                else:
                    x1 = a + F[m_next - 2] / F[m_next] * (b - a)
                    f1 = f(x1)
                    calc_count += 1
                    points.append(x1)
        else:
            decision = "f1 > f2 → a = x1"
            print(
                f"{k:>3} | {m:>3} | {frac1:>10.6f} | {frac2:>10.6f} | "
                f"{a:>10.6f} | {b:>10.6f} | {x1:>10.6f} | {x2:>10.6f} | "
                f"{f1:>12.8f} | {f2:>12.8f} | {Lk:>9.6f} | {decision:>22}"
            )
            a = x1
            x1, f1 = x2, f2 # успадкування

            if k < N - 3:
                m_next = N - (k + 1)
                if m_next == 3:
                    # залишається успадкована x1, нову x2  зі зсувом
                    x2 = x1 + delta
                    f2 = f(x2)
                    calc_count += 1
                    points.append(x2)
                else:
                    x2 = a + F[m_next - 1] / F[m_next] * (b - a)
                    f2 = f(x2)
                    calc_count += 1
                    points.append(x2)

    x_star = (a + b) / 2
    f_star = f(x_star)
    Ln = b - a
    points.append(x_star)
    error = abs(x_star - X_EXACT)

    print("*" * 140)
    print(f"\nx* = {x_star:.8f}")
    print(f"f(x*) = {f_star:.8f}")
    print(f"Похибка |x* − x*_точн| = {error:.8f}  (x*_точн ≈ {X_EXACT:.8f})")
    print(f"Кількість ітерацій: {N-2}")
    print(f"Кінцева довжина Ln = {Ln:.8f}")
    print(f"Фактично обчислень f(x): {calc_count} (Заплановано N-1 = {N-1})")
    bound = L0 / F[N] + delta
    print(f"Гарантія: Ln ≤ L0/F_N + δ → {Ln:.8f} ≤ {bound:.8f} → {Ln <= bound + 1e-12}")
    print(f"Точність: Ln ≤ ε → {Ln:.8f} ≤ {eps} → {Ln <= eps}")

    return x_star, f_star, calc_count, points, N - 2, Ln, error
#Метод Золотого Перерізу 
def golden_section(a0, b0, eps=0.01):
    tau = (math.sqrt(5) - 1) / 2
    a, b = a0, b0
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1 = f(x1)
    f2 = f(x2)
    calc_count = 2
    k = 1
    points = [x1, x2]

    while (b - a) > eps:
        if f1 <= f2:
            b = x2
            x2, f2 = x1, f1
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)
            points.append(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
            points.append(x2)
        calc_count += 1
        k += 1

    x_star = (a + b) / 2
    f_star = f(x_star)
    L_n = b - a
    points.append(x_star)
    error = abs(x_star - X_EXACT)
    return x_star, f_star, calc_count, points, k - 1, L_n, error

#Графічна візуалізація
def plot_results(a0, b0, fib_pts, x_f, x_g):
    left = a0 - 0.15
    right = b0 + 0.15
    x_vals = [left + i * (right - left) / 500 for i in range(501)]
    y_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(9, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.scatter(fib_pts[:-1], [f(x) for x in fib_pts[:-1]],
                color="orange", edgecolor="black", marker="o", s=50,
                label="Пробні точки (Фібоначчі)")
    plt.scatter([x_f], [f(x_f)], color="red", s=120, marker="*", zorder=6,
                label=f"Фібоначчі x* = {x_f:.5f}")
    plt.axvline(a0, color="gray", linestyle="--")
    plt.axvline(b0, color="gray", linestyle="--")
    plt.title("Метод чисел Фібоначчі")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Головний блок
if __name__ == "__main__":
    x0 = 1.15
    delta0 = 0.1
    eps = 0.01

    print(f"Цільова функція: f(x) = 0.5 * (x^2 + 1) * arctan(x) - 1.5 * x")
    print(f"Початкова точка x0 = {x0}\n")

    #Метод Свенна
    a0, b0, swann_pts, n_swann = swann(x0, delta0)

    # Метод Фібоначчі
    x_f, f_f, n_fib, fib_pts, iter_fib, L_fib, err_fib = fibonacci_search(a0, b0, eps)

    #Золотий переріз (для порівняння)
    x_g, f_g, n_gold, gold_pts, iter_gold, L_gold, err_gold = golden_section(a0, b0, eps)

    # Підсумкове порівняння
    L0 = b0 - a0
    eta_fib = (L_fib / L0) ** (1 / n_fib) if n_fib > 0 else 0
    eta_gold = (L_gold / L0) ** (1 / n_gold) if n_gold > 0 else 0

    print("=" * 130)
    print("Порівняння результатів (ε = 0.01)")
    print("=" * 130)
    print(f"{'Метод':<22} | {'x*':<12} | {'f(x*)':<12} | {'Ітерацій':<10} | "
          f"{'Обчислень f(x)':<15} | {'Ln':<12} | {'Похибка |x*-x*точн|':<20}")
    print("=" * 130)
    print(f"{'Фібоначчі (ЛР 4)':<22} | {x_f:<12.8f} | {f_f:<12.8f} | {iter_fib:<10} | "
          f"{n_fib:<15} | {L_fib:<12.8f} | {err_fib:<20.8f}")
    print(f"{'Золотий переріз (ЛР 3)':<22} | {x_g:<12.8f} | {f_g:<12.8f} | {iter_gold:<10} | "
          f"{n_gold:<15} | {L_gold:<12.8f} | {err_gold:<20.8f}")
    print("=" * 130)
    print(f"Точний мінімум: x*_точн ≈ {X_EXACT:.10f}")

    plot_results(a0, b0, fib_pts, x_f, x_g)