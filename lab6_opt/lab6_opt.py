# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
#цільова функція і похідні
def f(x):
    return 0.5 * (x**2 + 1) * math.atan(x) - 1.5 * x

def df(x):
    return x * math.atan(x) - 1.0

def d2f(x):
    return math.atan(x) + x / (x**2 + 1.0)

def verify_derivatives(x, h=1e-5):
    df_num = (f(x + h) - f(x - h)) / (2 * h)
    d2f_num = (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    print("*" * 75)
    print("Чисельна перевірка похідних у точці x = {:.4f} (h = {}):".format(x, h))
    print("  f'(x) аналітична: {:.8f} | чисельна: {:.8f} | різниця: {:.2e}".format(
        df(x), df_num, abs(df(x) - df_num)))
    print("  f''(x) аналітична: {:.8f} | чисельна: {:.8f} | різниця: {:.2e}".format(
        d2f(x), d2f_num, abs(d2f(x) - d2f_num)))
    print("*" * 75 + "\n")

X_EXACT = 1.1623398491423087
F_EXACT = f(X_EXACT)

# свенн
def swann(x0, delta0=0.1):
    f0 = f(x0)
    f_right = f(x0 + delta0)
    points = [x0]
    values = [f0]
    if f_right < f0:
        delta = delta0
        points.append(x0 + delta)
        values.append(f_right)
    else:
        f_left = f(x0 - delta0)
        if f_left < f0:
            delta = -delta0
            points.append(x0 + delta)
            values.append(f_left)
        else:
            return x0 - abs(delta0), x0 + abs(delta0), 3
    k = 1
    while True:
        step = (2 ** k) * delta
        x_next = points[-1] + step
        f_next = f(x_next)
        points.append(x_next)
        values.append(f_next)
        k += 1
        if not (f_next < values[-2]):
            break
    a0 = min(points[-3], points[-1])
    b0 = max(points[-3], points[-1])
    return a0, b0, len(values)

# з лр2-5
def dichotomy(a0, b0, sigma=0.01, eps=0.001):
    a, b = a0, b0
    k = calc_count = 0
    for _ in range(100):
        if (b - a) <= sigma:
            break
        x1 = (a + b) / 2 - eps / 2
        x2 = x1 + eps
        f1, f2 = f(x1), f(x2)
        calc_count += 2
        if f1 < f2:
            b = x2
        else:
            a = x1
        k += 1
    return (a + b) / 2, f((a + b) / 2), calc_count, k

def half_division(a0, b0, sigma=0.01):
    a, b = a0, b0
    xm = (a + b) / 2
    fm = f(xm)
    calc_count = 1
    k = 0
    for _ in range(100):
        if (b - a) <= sigma:
            break
        L = b - a
        x1 = a + L / 4
        x2 = b - L / 4
        f1, f2 = f(x1), f(x2)
        calc_count += 2
        if f1 < fm:
            b, xm, fm = xm, x1, f1
        elif f2 < fm:
            a, xm, fm = xm, x2, f2
        else:
            a, b = x1, x2
        k += 1
    return xm, fm, calc_count, k

def golden_section(a0, b0, eps=0.01):
    tau = (math.sqrt(5) - 1) / 2
    a, b = a0, b0
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1, f2 = f(x1), f(x2)
    calc_count = 2
    k = 1
    for _ in range(100):
        if (b - a) <= eps:
            break
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
        calc_count += 1
        k += 1
    return (a + b) / 2, f((a + b) / 2), calc_count, k - 1

def fibonacci_search(a0, b0, eps=0.01):
    L0 = b0 - a0
    delta = eps / 10.0
    R = L0 / max(eps - delta, 1e-12)
    F = [0, 1, 1]
    while F[-1] < R and len(F) < 50:
        F.append(F[-1] + F[-2])
    N = len(F) - 1
    a, b = float(a0), float(b0)
    x1 = a + F[N - 2] / F[N] * (b - a)
    x2 = a + F[N - 1] / F[N] * (b - a)
    f1, f2 = f(x1), f(x2)
    calc_count = 2
    for k in range(N - 2):
        if f1 <= f2:
            b = x2
            x2, f2 = x1, f1
            if k < N - 3:
                m_next = N - (k + 1)
                x1 = x2 - delta if m_next == 3 else a + F[m_next - 2] / F[m_next] * (b - a)
                f1 = f(x1)
                calc_count += 1
        else:
            a = x1
            x1, f1 = x2, f2
            if k < N - 3:
                m_next = N - (k + 1)
                x2 = x1 + delta if m_next == 3 else a + F[m_next - 1] / F[m_next] * (b - a)
                f2 = f(x2)
                calc_count += 1
    return (a + b) / 2, f((a + b) / 2), calc_count, N - 2

def powell(x1, dx=0.3, sigma_f=0.01, sigma_x=0.001):
    x2 = x1 + dx
    f1, f2 = f(x1), f(x2)
    x3 = x1 + 2 * dx if f1 > f2 else x1 - dx
    f3 = f(x3)
    calc_count = 3
    history_len = 0
    for k in range(1, 30):
        xs = [x1, x2, x3]
        fs = [f1, f2, f3]
        i_min = fs.index(min(fs))
        i_max = fs.index(max(fs))
        a1 = (f2 - f1) / (x2 - x1)
        a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2)
        if abs(a2) < 1e-14:
            break
        if a2 > 0:
            x_tilde = -a1 / (2 * a2) + (x1 + x2) / 2
        else:
            x_tilde = 2 * xs[i_min] - xs[i_max]
        f_tilde = f(x_tilde)
        calc_count += 1
        history_len = k
        if abs(fs[i_min] - f_tilde) <= sigma_f and abs(xs[i_min] - x_tilde) <= sigma_x:
            return x_tilde, f_tilde, calc_count, history_len
        if i_max == 0:
            x1, f1 = x_tilde, f_tilde
        elif i_max == 1:
            x2, f2 = x_tilde, f_tilde
        else:
            x3, f3 = x_tilde, f_tilde
    return xs[i_min], fs[i_min], calc_count, history_len


# таблиця класичний Ньютон
def newton_classical(x0, eps=0.001, max_iter=30, verbose=False):
    if verbose:
        print("=" * 110)
        print("ТАБЛИЦЯ 4.8: Класичний метод Ньютона (Старт x0 = {:.4f}, ε = {})".format(x0, eps))
        print("=" * 110)
        print("{:>3} | {:>12} | {:>14} | {:>14} | {:>12} | {:>12}".format(
            "k", "xk", "f(xk)", "f'(xk)", "|f'(xk)|", "f''(xk)"))
        print("-" * 110)

    x = x0
    n_df = n_d2f = 0

    for k in range(max_iter):
        fx = f(x)
        d1 = df(x)
        n_df += 1

        # Перевірка умови зупинки до обчислення f''(x)
        if abs(d1) <= eps:
            if verbose:
                print("{:>3} | {:>12.6f} | {:>14.8f} | {:>14.6f} | {:>12.6f} | {:>12}".format(
                    k, x, fx, d1, abs(d1), "—"))
                print("-" * 110)
                print("Умову зупинки виконано: x* = {:.10f}, ітерацій: {}, N(f') = {}, N(f'') = {}\n".format(
                    x, k, n_df, n_d2f))
            return x, fx, k, 0, n_df, n_d2f, True

        d2 = d2f(x)
        n_d2f += 1

        # Захист від ділення на нуль або неправильного напрямку
        if d2 <= 1e-12:
            if verbose:
                print("Зупинка: f''(x) <= 1e-12. Метод не може продовжуватися.")
            return x, fx, k, 0, n_df, n_d2f, False

        if verbose:
            print("{:>3} | {:>12.6f} | {:>14.8f} | {:>14.6f} | {:>12.6f} | {:>12.6f}".format(
                k, x, fx, d1, abs(d1), d2))

        # Формула класичного методу Ньютона
        x = x - d1 / d2

    # Перевіряємо кінцеву точку після останньої ітерації
    fx = f(x)
    d1 = df(x)
    n_df += 1

    if abs(d1) <= eps:
        if verbose:
            print("{:>3} | {:>12.6f} | {:>14.8f} | {:>14.6f} | {:>12.6f} | {:>12}".format(
                max_iter, x, fx, d1, abs(d1), "—"))
            print("-" * 110)
            print("Умову зупинки виконано: x* = {:.10f}, ітерацій: {}, N(f') = {}, N(f'') = {}\n".format(
                x, max_iter, n_df, n_d2f))
        return x, fx, max_iter, 0, n_df, n_d2f, True

    if verbose:
        print("Досягнуто максимальної кількості ітерацій. Збіжність не підтверджено.\n")

    return x, fx, max_iter, 0, n_df, n_d2f, False

# таблиця Ньютона-Рафсона з регулюванням
def newton_raphson_regulated(x0, eps=0.001, alpha=0.5, max_iter=30, verbose=False):
    if verbose:
        print("=" * 135)
        print("ТАБЛИЦЯ 4.9: Метод Ньютона-Рафсона з регулюванням кроку (Старт x0 = {:.4f}, α = {})".format(x0, alpha))
        print("=" * 135)
        print("{:>3} | {:>10} | {:>11} | {:>11} | {:>11} | {:>5} | {:>10} | {:>11} | {:>12} | {:<15}".format(
            "k", "xk", "f(xk)", "f'(xk)", "f''(xk)", "t", "x_bar", "f(x_bar)", "f'(x_bar)", "Умова (3.10)"))
        print("-" * 135)

    x = x0
    n_f = n_df = n_d2f = 0
    fx = f(x)
    d1 = df(x)
    n_f += 1
    n_df += 1

    if abs(d1) <= eps:
        return x, fx, 0, n_f, n_df, n_d2f, True

    for k in range(1, max_iter + 1):
        d2 = d2f(x)
        n_d2f += 1
        if d2 <= 0:
            break

        t = 1.0
        success = False
        while t >= 1e-8:
            x_tilde = x - t * d1 / d2
            fx_tilde = f(x_tilde)
            d1_tilde = df(x_tilde)
            n_f += 1
            n_df += 1

            if abs(d1_tilde) <= eps:
                if verbose:
                    print("{:>3} | {:>10.4f} | {:>11.4f} | {:>11.4f} | {:>11.4f} | {:>5.1f} | {:>10.4f} | {:>11.4f} | {:>12.4f} | {:<15}".format(
                        k, x, fx, d1, d2, t, x_tilde, fx_tilde, d1_tilde, "Виконано (OK)"))
                    print("-" * 135)
                    print("Умову зупинки виконано: x* = {:.10f}, ітерацій: {}, N(f) = {}, N(f') = {}, N(f'') = {}\n".format(
                        x_tilde, k, n_f, n_df, n_d2f))
                return x_tilde, fx_tilde, k, n_f, n_df, n_d2f, True

            condition_met = (fx_tilde - fx) <= -alpha * t * (d1 ** 2) / d2
            cond_str = "Виконано (OK)" if condition_met else "Відхилено (X)"

            if verbose:
                print("{:>3} | {:>10.4f} | {:>11.4f} | {:>11.4f} | {:>11.4f} | {:>5.1f} | {:>10.4f} | {:>11.4f} | {:>12.4f} | {:<15}".format(
                    k, x, fx, d1, d2, t, x_tilde, fx_tilde, d1_tilde, cond_str))

            if condition_met:
                success = True
                break
            t /= 2.0

        if not success:
            break

        x = x_tilde
        fx = fx_tilde
        d1 = d1_tilde

    return x, fx, k, n_f, n_df, n_d2f, False

# графік
def plot_newton_geometry(a0, b0, x0):
    left = a0 - 0.5
    right = b0 + 0.5
    xs = [left + i * (right - left) / 500 for i in range(501)]
    phi_vals = [df(x) for x in xs]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, phi_vals, label="phi(x) = f'(x)", color="blue", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", alpha=0.7, label="phi(x) = 0")
    plt.axvline(a0, color="gray", linestyle=":", label="a0 = {:.2f}".format(a0))
    plt.axvline(b0, color="gray", linestyle=":", label="b0 = {:.2f}".format(b0))

    x_curr = x0
    for i in range(2):
        d1 = df(x_curr)
        d2 = d2f(x_curr)
        if abs(d2) < 1e-14:
            break
        x_next = x_curr - d1 / d2
        tangent_y = [d1 + d2 * (x - x_curr) for x in xs]
        plt.plot(xs, tangent_y, linestyle="--", label="Дотична в x{} = {:.3f}".format(i, x_curr))
        plt.scatter([x_curr], [d1], color="red", zorder=5)
        x_curr = x_next

    plt.title("Метод Ньютона: пошук кореня f'(x) = 0 через дотичні")
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# головна програма
if __name__ == "__main__":
    x0_initial = 1.15
    delta0 = 0.1
    epsilon = 0.001

    print("=" * 90)
    print("ЛАБОРАТОРНА РОБОТА № 6: Методи з використанням похідних")
    print("Цільова функція: f(x) = 1/2(x^2 + 1) arctan(x) - 3/2 x")
    print("Точний мінімум: x* = {:.10f}, f* = {:.10f}".format(X_EXACT, F_EXACT))
    print("=" * 90 + "\n")

    a0, b0, _ = swann(x0_initial, delta0)
    print("Інтервал локалізації за Свенном: [a0, b0] = [{:.6f}; {:.6f}]\n".format(a0, b0))
    verify_derivatives(a0)

    # Таблиці 
    x_n, f_n, it_n, _, n_df_n, n_d2f_n, _ = newton_classical(a0, eps=epsilon, verbose=True)
    x_nr, f_nr, it_nr, n_f_nr, n_df_nr, n_d2f_nr, _ = newton_raphson_regulated(a0, eps=epsilon, alpha=0.5, verbose=True)

    # Експеримент зі стартами
    print("=" * 115)
    print("ЕКСПЕРИМЕНТ ЗІ СТАРТАМИ (Пункт 7)")
    print("=" * 115)
    test_starts = [
        ("Ліва межа (a0)", a0),
        ("Права межа (b0)", b0),
        ("Далеко за межами (x = 3.5)", 3.5)
    ]
    print("{:<30} | {:<35} | {:<38}".format("Стартова точка", "Класичний Ньютон", "Ньютон із регулюванням"))
    print("-" * 115)

    for name, st in test_starts:
        rx_n, rf_n, rit_n, _, _, _, r_n = newton_classical(st, eps=epsilon, max_iter=20, verbose=False)
        rx_nr, rf_nr, rit_nr, _, _, _, r_nr = newton_raphson_regulated(st, eps=epsilon, alpha=0.5, max_iter=20, verbose=False)
        res_n_str = "x*={:.4f}, ітер={}, {}".format(rx_n, rit_n, "Успіх" if r_n else "Збій")
        res_nr_str = "x*={:.4f}, ітер={}, {}".format(rx_nr, rit_nr, "Успіх" if r_nr else "Збій")
        print("{:<30} | {:<35} | {:<38}".format(name, res_n_str, res_nr_str))
    print("=" * 115 + "\n")

    # Методи попередніх ЛР
    x_d, f_d, n_dich, it_d = dichotomy(a0, b0, sigma=epsilon)
    x_h, f_h, n_half, it_h = half_division(a0, b0, sigma=epsilon)
    x_g, f_g, n_gold, it_gold = golden_section(a0, b0, eps=epsilon)
    x_f, f_f, n_fib, it_fib = fibonacci_search(a0, b0, eps=epsilon)
    x_p, f_p, n_powell, it_powell = powell(a0, dx=0.3)

    # Зведена таблиця
    print("=" * 130)
    print("ЗВЕДЕНА ТАБЛИЦЯ ПОРІВНЯННЯ ВСІХ МЕТОДІВ ОПТИМІЗАЦІЇ (ЛР 1 — ЛР 6)")
    print("=" * 130)
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12} | {:>12} | {:>14}".format(
        "Метод", "ітер (n)", "N(f)", "N(f')", "N(f'')", "x*", "f(x*)", "Абс. похибка"))
    print("-" * 130)
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Дихотомія (ЛР 2)", it_d, n_dich, "—", "—", x_d, f_d, abs(x_d - X_EXACT)))
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Половинний поділ (ЛР 2)", it_h, n_half, "—", "—", x_h, f_h, abs(x_h - X_EXACT)))
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Золотий переріз (ЛР 3)", it_gold, n_gold, "—", "—", x_g, f_g, abs(x_g - X_EXACT)))
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Фібоначчі (ЛР 4)", it_fib, n_fib, "—", "—", x_f, f_f, abs(x_f - X_EXACT)))
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Пауелл (ЛР 5)", it_powell, n_powell, "—", "—", x_p, f_p, abs(x_p - X_EXACT)))
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Класичний Ньютон (ЛР 6)", it_n, "—", n_df_n, n_d2f_n, x_n, f_n, abs(x_n - X_EXACT)))
    print("{:<35} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Ньютон-Рафсон з регулюванням (ЛР 6)", it_nr, n_f_nr, n_df_nr, n_d2f_nr, x_nr, f_nr, abs(x_nr - X_EXACT)))
    print("=" * 130 + "\n")

    # Графік
    plot_newton_geometry(a0, b0, a0)
    print("Роботу завершено успішно.")