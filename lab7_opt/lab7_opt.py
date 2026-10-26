# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt

# цільова функція і похідні (та сама, що в ЛР 6)
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

# свен (з ЛР 1)

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

# попередні методи для таблиці
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


# Методи Ньютона (з ЛР 6) для порівняння
def newton_classical(x0, eps=0.001, max_iter=30, verbose=False):
    x = x0
    n_df = n_d2f = 0
    for k in range(max_iter):
        fx = f(x)
        d1 = df(x)
        n_df += 1
        if abs(d1) <= eps:
            return x, fx, k, 0, n_df, n_d2f, True
        d2 = d2f(x)
        n_d2f += 1
        if d2 <= 1e-12:
            return x, fx, k, 0, n_df, n_d2f, False
        x = x - d1 / d2
    fx = f(x)
    d1 = df(x)
    n_df += 1
    return x, fx, max_iter, 0, n_df, n_d2f, abs(d1) <= eps

def newton_raphson_regulated(x0, eps=0.001, alpha=0.5, max_iter=30, verbose=False):
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
                return x_tilde, fx_tilde, k, n_f, n_df, n_d2f, True
            condition_met = (fx_tilde - fx) <= -alpha * t * (d1 ** 2) / d2
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

# метод січних
def secant_method(x0, x1, eps=0.001, max_iter=50, verbose=True):
    """
    Метод січних (друга модифікація Ньютона) за схемою посібника.
    Кроки 1–6, правило знаку (regula falsi).
    Повертає: x*, f(x*), n (ітерацій), Nf', історію.
    """
    xm = float(x0)          # x_{k-1} (ліва)
    xk = float(x1)          # x_k     (права)
    pm = df(xm)             # f'(x_{k-1})
    pk = df(xk)             # f'(x_k)
    n_df = 2                # два обчислення на старті

    if verbose:
        print("=" * 130)
        print("ТАБЛИЦЯ 4.11: Метод січних (схема посібника, правило знаку)")
        print("Старт: x0 = {:.6f}, x1 = {:.6f}, ε = {}".format(x0, x1, eps))
        print("=" * 130)
        print("{:>3} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:<18}".format(
            "k", "x_{k-1}", "f'(x_{k-1})", "x_k", "f'(x_k)", "x_{k+1}", "f'(x_{k+1})", "f(x_{k+1})", "|x_{k+1}-x*|", "Дія (крок 5)"))
        print("-" * 130)

    history = []

    for k in range(1, max_iter + 1):
        # Захист від ділення на нуль
        if abs(pk - pm) < 1e-14:
            if verbose:
                print("Зупинка: f'(xk) ≈ f'(x_{k-1}) — січна горизонтальна.")
            return xm if abs(pm) < abs(pk) else xk, f(xm if abs(pm) < abs(pk) else xk), k-1, n_df, history, False

        # формула (3.16)
        xn = (pk * xm - pm * xk) / (pk - pm)

        #обчислити f'(x_{k+1})
        pn = df(xn)
        n_df += 1
        fn = f(xn)

        action = ""
        stop = abs(pn) <= eps

        if stop:
            action = "|f'| ≤ ε → СТОП"
        else:
            #правило знаку
            if pn > 0:
                xk = xn
                pk = pn
                action = "xk ← xk+1 (права)"
            else:
                xm = xn
                pm = pn
                action = "x_{k-1} ← xk+1 (ліва)"

        err = abs(xn - X_EXACT)
        history.append({
            "k": k, "xm": xm, "pm": pm, "xk": xk, "pk": pk,
            "xn": xn, "pn": pn, "fn": fn, "err": err, "action": action
        })

        if verbose:
            print("{:>3} | {:>12.6f} | {:>12.6f} | {:>12.6f} | {:>12.6f} | {:>12.6f} | {:>12.6f} | {:>12.6f} | {:>12.2e} | {:<18}".format(
                k, history[-1]["xm"] if k > 1 else x0, 
                history[-1]["pm"] if k > 1 else df(x0),
                history[-1]["xk"] if not stop else (xk if pn > 0 else xn),
                history[-1]["pk"] if not stop else (pk if pn > 0 else pn),
                xn, pn, fn, err, action))

        if stop:
            if verbose:
                print("-" * 130)
                print("Умову зупинки виконано: x* = {:.10f}, f(x*) = {:.10f}".format(xn, fn))
                print("Ітерацій n = {}, N(f') = {} (n+2), N(f'') = 0\n".format(k, n_df))
            return xn, fn, k, n_df, history, True

    if verbose:
        print("Досягнуто максимальної кількості ітерацій.\n")
    return xn, fn, max_iter, n_df, history, False



# «чисті січні (завжди дві останні точки) експеримент
def pure_secant_method(x0, x1, eps=0.001, max_iter=50, verbose=True):
    """
    Класичний метод січних (без правила знаку).
    Завжди: x_{k-1} ← x_k, x_k ← x_{k+1}
    """
    xm = float(x0)
    xk = float(x1)
    pm = df(xm)
    pk = df(xk)
    n_df = 2

    if verbose:
        print("=" * 110)
        print("ЕКСПЕРИМЕНТ: «Чисті» січні (дві останні точки, без правила знаку)")
        print("Старт: x0 = {:.6f}, x1 = {:.6f}, ε = {}".format(x0, x1, eps))
        print("=" * 110)
        print("{:>3} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}".format(
            "k", "x_{k-1}", "x_k", "оцінка f''", "x_{k+1}", "f'(x_{k+1})", "|x_{k+1}-x*|", "f(x_{k+1})"))
        print("-" * 110)

    for k in range(1, max_iter + 1):
        if abs(pk - pm) < 1e-14:
            if verbose:
                print("Зупинка: f' однакові.")
            return xk, f(xk), k-1, n_df, False

        # оцінка f''
        f2_est = (pk - pm) / (xk - xm)

        xn = (pk * xm - pm * xk) / (pk - pm)
        pn = df(xn)
        n_df += 1
        fn = f(xn)
        err = abs(xn - X_EXACT)

        if verbose:
            print("{:>3} | {:>12.6f} | {:>12.6f} | {:>12.4f} | {:>12.6f} | {:>12.6f} | {:>12.2e} | {:>12.6f}".format(
                k, xm, xk, f2_est, xn, pn, err, fn))

        if abs(pn) <= eps:
            if verbose:
                print("-" * 110)
                print("Умову зупинки виконано: x* = {:.10f}, f(x*) = {:.10f}".format(xn, fn))
                print("Ітерацій n = {}, N(f') = {}\n".format(k, n_df))
            return xn, fn, k, n_df, True

        # завжди зсуваємо пару
        xm, pm = xk, pk
        xk, pk = xn, pn

    return xn, fn, max_iter, n_df, False

# ГРАФІК φ(x) = f'(x) з першими січними
def plot_secant_geometry(a0, b0, history):
    left = a0 - 0.3
    right = b0 + 0.3
    xs = [left + i * (right - left) / 500 for i in range(501)]
    phi_vals = [df(x) for x in xs]

    plt.figure(figsize=(11, 6.5))
    plt.plot(xs, phi_vals, label=r"$\varphi(x) = f'(x)$", color="blue", linewidth=2.2)
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)

    # точний корінь
    plt.axvline(X_EXACT, color="red", linestyle=":", alpha=0.7, label=r"$x^* \approx {:.4f}$".format(X_EXACT))

    # перші дві січні
    if len(history) >= 1:
        # січна 1: через (a0, f'(a0)) і (b0, f'(b0))
        x_a, p_a = a0, df(a0)
        x_b, p_b = b0, df(b0)
        x2 = history[0]["xn"]
        plt.plot([x_a, x_b], [p_a, p_b], color="orange", linestyle="--", linewidth=1.8,
                 label="січна через $x_0, x_1$ → $x_2={:.4f}$".format(x2))
        plt.scatter([x_a, x_b, x2], [p_a, p_b, df(x2)], color="orange", zorder=5, s=60)

    if len(history) >= 2:
        # січна 2: через (a0, f'(a0)) і (x2, f'(x2))
        x2 = history[0]["xn"]
        p2 = history[0]["pn"]
        x3 = history[1]["xn"]
        plt.plot([a0, x2], [df(a0), p2], color="green", linestyle="--", linewidth=1.8,
                 label="січна через $x_0, x_2$ → $x_3={:.4f}$".format(x3))
        plt.scatter([x3], [df(x3)], color="green", zorder=5, s=60)

    plt.title("Метод січних: $\\varphi(x)=f'(x)$, хорди замість дотичних")
    plt.xlabel("x")
    plt.ylabel(r"$f'(x)$")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()


# головна програма
if __name__ == "__main__":
    x0_initial = 1.15
    delta0 = 0.1
    epsilon = 0.001

    print("=" * 100)
    print("ЛАБОРАТОРНА РОБОТА № 7: Метод січних (друга модифікація методу Ньютона)")
    print("Цільова функція: f(x) = 1/2 (x² + 1) arctan(x) − 3/2 x")
    print("Точний мінімум: x* = {:.10f}, f* = {:.10f}".format(X_EXACT, F_EXACT))
    print("=" * 100 + "\n")

    #Інтервал за Свенном
    a0, b0, _ = swann(x0_initial, delta0)
    print("Інтервал локалізації за Свенном: [a₀, b₀] = [{:.6f}; {:.6f}]".format(a0, b0))
    print("f'(a₀) = {:.6f},  f'(b₀) = {:.6f}  (знаки різні — умова виконана)\n".format(df(a0), df(b0)))

    verify_derivatives(a0)

    # Метод січних 
    x_sec, f_sec, it_sec, n_df_sec, hist_sec, ok_sec = secant_method(a0, b0, eps=epsilon, verbose=True)

    #Експеримент: «чисті» січні
    x_pure, f_pure, it_pure, n_df_pure, ok_pure = pure_secant_method(a0, b0, eps=epsilon, verbose=True)

    # Методи попередніх робіт (для зведеної таблиці)
    x_d, f_d, n_dich, it_d = dichotomy(a0, b0, sigma=epsilon)
    x_h, f_h, n_half, it_h = half_division(a0, b0, sigma=epsilon)
    x_g, f_g, n_gold, it_gold = golden_section(a0, b0, eps=epsilon)
    x_f, f_f, n_fib, it_fib = fibonacci_search(a0, b0, eps=epsilon)
    x_p, f_p, n_powell, it_powell = powell(a0, dx=0.3)
    x_n, f_n, it_n, _, n_df_n, n_d2f_n, _ = newton_classical(a0, eps=epsilon)
    x_nr, f_nr, it_nr, n_f_nr, n_df_nr, n_d2f_nr, _ = newton_raphson_regulated(a0, eps=epsilon, alpha=0.5)

    #Зведена таблиця
    print("=" * 145)
    print("ЗВЕДЕНА ТАБЛИЦЯ ПОРІВНЯННЯ ВСІХ МЕТОДІВ ОПТИМІЗАЦІЇ (ЛР 1 — ЛР 7)")
    print("=" * 145)
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12} | {:>12} | {:>14}".format(
        "Метод", "ітер (n)", "N(f)", "N(f')", "N(f'')", "x*", "f(x*)", "Абс. похибка"))
    print("-" * 145)
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Дихотомія (ЛР 2)", it_d, n_dich, "—", "—", x_d, f_d, abs(x_d - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Половинний поділ (ЛР 2)", it_h, n_half, "—", "—", x_h, f_h, abs(x_h - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Золотий переріз (ЛР 3)", it_gold, n_gold, "—", "—", x_g, f_g, abs(x_g - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Фібоначчі (ЛР 4)", it_fib, n_fib, "—", "—", x_f, f_f, abs(x_f - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Пауелл (ЛР 5)", it_powell, n_powell, "—", "—", x_p, f_p, abs(x_p - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Класичний Ньютон (ЛР 6)", it_n, "—", n_df_n, n_d2f_n, x_n, f_n, abs(x_n - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Ньютон-Рафсон з регулюванням (ЛР 6)", it_nr, n_f_nr, n_df_nr, n_d2f_nr, x_nr, f_nr, abs(x_nr - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "Метод січних (ЛР 7, схема посібника)", it_sec, "—", n_df_sec, 0, x_sec, f_sec, abs(x_sec - X_EXACT)))
    print("{:<38} | {:>8} | {:>6} | {:>6} | {:>6} | {:>12.8f} | {:>12.8f} | {:>14.2e}".format(
        "«Чисті» січні (експеримент)", it_pure, "—", n_df_pure, 0, x_pure, f_pure, abs(x_pure - X_EXACT)))
    print("=" * 145 + "\n")

    #Короткий підсумок порівняння
    print("КОРОТКИЙ ПІДСУМОК:")
    print("  • Метод січних (посібник):   n = {}, N(f') = {}, похибка = {:.2e}".format(
        it_sec, n_df_sec, abs(x_sec - X_EXACT)))
    print("  • «Чисті» січні:             n = {}, N(f') = {}, похибка = {:.2e}".format(
        it_pure, n_df_pure, abs(x_pure - X_EXACT)))
    print("  • Класичний Ньютон:          n = {}, N(f')+N(f'') = {}+{}, похибка = {:.2e}".format(
        it_n, n_df_n, n_d2f_n, abs(x_n - X_EXACT)))
    print("  Висновок: відмова від f'' коштує лише кількох додаткових ітерацій,")
    print("            але повністю знімає потребу у формулі другої похідної.\n")

    #Графік
    plot_secant_geometry(a0, b0, hist_sec)

    print("Роботу завершено успішно.")
