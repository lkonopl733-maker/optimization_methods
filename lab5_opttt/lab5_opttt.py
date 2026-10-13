# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt
#цільова функція
def f(x):
    return 0.5 * (x**2 + 1) * math.atan(x) - 1.5 * x

# Точний мінімум для оцінки похибки
X_EXACT = 1.1623398491423087
F_EXACT = f(X_EXACT)
#свенн
def swann(x0, delta0=0.1):
    print("*" * 90)
    print("Метод Свенна (знаходження початкового інтервалу)")
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
            calc_count = 3
            print("Мінімум знаходиться в початковому околі x0.")
            print(
                f"Інтервал локалізації [a0, b0] = "
                f"[{a0:.6f}; {b0:.6f}]"
            )
            print(f"Всього обчислень f(x): {calc_count}\n")
            return a0, b0, [a0, x0, b0], calc_count

    print("\nТаблиця ітерацій Свенна:")
    print("-" * 75)
    print(
        f"{'k':>3} | {'Крок':>18} | {'xk':>12} | "
        f"{'f(xk)':>14} | {'f↓?':>6}"
    )
    print("-" * 75)
    print(
        f"{0:>3} | {'—':>18} | {points[0]:>12.6f} | "
        f"{values[0]:>14.8f} | {'—':>6}"
    )
    print(
        f"{1:>3} | {delta:>18.6f} | {points[1]:>12.6f} | "
        f"{values[1]:>14.8f} | {'+':>6}"
    )

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

        print(
            f"{k:>3} | {step:>18.6f} | {x_next:>12.6f} | "
            f"{f_next:>14.8f} | {comparison:>6}"
        )

        if not is_decreasing:
            break

    a0 = min(points[-3], points[-1])
    b0 = max(points[-3], points[-1])
    calc_count = len(values)

    print("-" * 75)
    print(
        f"\nІнтервал локалізації [a0, b0] = "
        f"[{a0:.6f}; {b0:.6f}]"
    )
    print(f"Довжина інтервалу L0 = {b0 - a0:.6f}")
    print(f"Всього обчислень f(x) у Свенні: {calc_count}\n")

    return a0, b0, points, calc_count

#метод Пауелла. Квадратина апроксимація
def powell(x1, dx=0.3, sigma_f=0.01, sigma_x=0.001):
    print("*" * 145)
    print("Метод Пауелла (квадратична апроксимація)")
    print("*" * 145)
    print(f"Початкова точка x1 = {x1:.6f}, крок Δx = {dx:.6f}")
    print(f"Точності: sig_f = {sigma_f}, sig_f = {sigma_x}\n")

    # Початкова трійка точок
    x2 = x1 + dx

    f1 = f(x1)
    f2 = f(x2)

    if f1 > f2:
        x3 = x1 + 2 * dx
    else:
        x3 = x1 - dx

    f3 = f(x3)
    calc_count = 3
    history = []

    header = (
        f"{'k':>3} | {'x1':>10} | {'x2':>10} | {'x3':>10} | "
        f"{'f(x1)':>11} | {'f(x2)':>11} | {'f(x3)':>11} | "
        f"{'a1':>10} | {'a2':>10} | {'x~':>10} | "
        f"{'f(x~)':>11} | {'|Δf|':>10} | {'|Δx|':>10} | "
        f"{'Дія':>10}"
    )

    print(header)
    print("-" * 155)

    for k in range(1, 25):
        xs = [x1, x2, x3]
        fs = [f1, f2, f3]

        # Перевірка, чи всі точки різні
        if (
            abs(x2 - x1) < 1e-12
            or abs(x3 - x1) < 1e-12
            or abs(x3 - x2) < 1e-12
        ):
            i_best = fs.index(min(fs))
            print("Помилка: точки збігаються.")
            print("Зупинка методу Пауелла.")

            return xs[i_best], fs[i_best], calc_count, history

        i_min = fs.index(min(fs))
        i_max = fs.index(max(fs))

        # Коефіцієнти квадратичної апроксимації
        a1 = (f2 - f1) / (x2 - x1)

        a2 = (
            ((f3 - f1) / (x3 - x1) - a1)
            / (x3 - x2)
        )

        if abs(a2) < 1e-14:
            print("Увага: a2 = 0. Змініть початковий крок Δx.")
            return xs[i_min], fs[i_min], calc_count, history

        # Вершина параболи або відбиття найгіршої точки
        if a2 > 0:
            x_tilde = -a1 / (2 * a2) + (x1 + x2) / 2
        else:
            x_tilde = 2 * xs[i_min] - xs[i_max]

        # Перевірка збігу нової точки з наявними
        if any(abs(x_tilde - x) < 1e-12 for x in xs):
            print("Нова точка збігається з наявною.")
            print("Зупинка методу. Спробуйте змінити Δx.")

            return xs[i_min], fs[i_min], calc_count, history

        f_tilde = f(x_tilde)
        calc_count += 1

        diff_f = abs(fs[i_min] - f_tilde)
        diff_x = abs(xs[i_min] - x_tilde)

        if i_max == 0:
            action = "x1 - x~"
        elif i_max == 1:
            action = "x2 - x~"
        else:
            action = "x3 - x~"

        history.append({
            "k": k,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "a1": a1,
            "a2": a2,
            "xt": x_tilde,
            "ft": f_tilde,
            "diff_f": diff_f,
            "diff_x": diff_x,
            "action": action
        })

        print(
            f"{k:>3} | {x1:>10.6f} | {x2:>10.6f} | "
            f"{x3:>10.6f} | {f1:>11.6f} | "
            f"{f2:>11.6f} | {f3:>11.6f} | "
            f"{a1:>10.6f} | {a2:>10.6f} | "
            f"{x_tilde:>10.6f} | {f_tilde:>11.6f} | "
            f"{diff_f:>10.6f} | {diff_x:>10.6f} | "
            f"{action:>10}"
        )

        # Обидві умови зупинки мають виконуватися одночасно
        if diff_f <= sigma_f and diff_x <= sigma_x:
            x_star = x_tilde
            f_star = f_tilde

            print("-" * 155)
            print(f"\nУмови зупинки виконані на ітерації {k}!")
            print(f"x* = {x_star:.10f}")
            print(f"f(x*) = {f_star:.10f}")
            print(
                f"Похибка |x* − x*_точн| = "
                f"{abs(x_star - X_EXACT):.2e}"
            )
            print(
                f"Всього обчислень f(x): {calc_count} "
                f"(Nf = 3 + n, n = {k})\n"
            )

            return x_star, f_star, calc_count, history

        # Заміна найгіршої точки
        if i_max == 0:
            x1, f1 = x_tilde, f_tilde
        elif i_max == 1:
            x2, f2 = x_tilde, f_tilde
        else:
            x3, f3 = x_tilde, f_tilde

    # Якщо досягнуто максимальної кількості ітерацій
    xs = [x1, x2, x3]
    fs = [f1, f2, f3]
    i_best = fs.index(min(fs))

    print("Досягнуто максимальної кількості ітерацій.")
    return xs[i_best], fs[i_best], calc_count, history

#золотий переріз
def golden_section(a0, b0, eps=0.01):
    tau = (math.sqrt(5) - 1) / 2

    a, b = a0, b0

    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)

    f1 = f(x1)
    f2 = f(x2)

    calc_count = 2
    k = 0

    while (b - a) > eps:
        if f1 <= f2:
            b = x2
            x2, f2 = x1, f1

            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)

        else:
            a = x1
            x1, f1 = x2, f2

            x2 = a + tau * (b - a)
            f2 = f(x2)

        calc_count += 1
        k += 1

    x_star = (a + b) / 2
    f_star = f(x_star)
    calc_count += 1

    return x_star, f_star, calc_count, k, b - a

#фібоначчі
def fibonacci_search(a0, b0, eps=0.01):
    L0 = b0 - a0

    delta = eps / 10.0
    R = L0 / (eps - delta)

    F = [0, 1, 1]

    while F[-1] < R:
        F.append(F[-1] + F[-2])

    N = len(F) - 1

    a, b = float(a0), float(b0)
    m = N

    x1 = a + F[m - 2] / F[m] * (b - a)
    x2 = a + F[m - 1] / F[m] * (b - a)

    f1 = f(x1)
    f2 = f(x2)

    calc_count = 2

    for k in range(N - 2):
        if f1 <= f2:
            b = x2
            x2, f2 = x1, f1

            if k < N - 3:
                m_next = N - (k + 1)

                if m_next == 3:
                    x1 = x2 - delta
                else:
                    x1 = (
                        a
                        + F[m_next - 2] / F[m_next] * (b - a)
                    )

                f1 = f(x1)
                calc_count += 1

        else:
            a = x1
            x1, f1 = x2, f2

            if k < N - 3:
                m_next = N - (k + 1)

                if m_next == 3:
                    x2 = x1 + delta
                else:
                    x2 = (
                        a
                        + F[m_next - 1] / F[m_next] * (b - a)
                    )

                f2 = f(x2)
                calc_count += 1

    x_star = (a + b) / 2
    f_star = f(x_star)
    calc_count += 1

    return x_star, f_star, calc_count, N - 2, b - a
#графік методу Пауела
def plot_powell(history, x_star, f_star, a0, b0):
    if not history:
        print("Графік Пауелла не побудовано: немає даних ітерацій.")
        return

    first = history[0]

    x1 = first["x1"]
    x2 = first["x2"]
    x3 = first["x3"]

    a1 = first["a1"]
    a2 = first["a2"]

    left = min(a0, x1, x2, x3, x_star) - 0.3
    right = max(b0, x1, x2, x3, x_star) + 0.3

    xs = [
        left + i * (right - left) / 600
        for i in range(601)
    ]

    ys = [f(x) for x in xs]

    y_parab = [
        first["f1"]
        + a1 * (x - x1)
        + a2 * (x - x1) * (x - x2)
        for x in xs
    ]

    plt.figure(figsize=(11, 6))

    plt.plot(
        xs, ys,
        "b-",
        linewidth=2,
        label="f(x)"
    )

    plt.plot(
        xs, y_parab,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Парабола (1-а ітерація)"
    )

    plt.scatter(
        [x1, x2, x3],
        [first["f1"], first["f2"], first["f3"]],
        c="purple",
        s=70,
        zorder=5,
        label="Стартова трійка (x1, x2, x3)"
    )

    plt.scatter(
        [x_star],
        [f_star],
        c="red",
        s=160,
        marker="*",
        zorder=6,
        label=f"x* = {x_star:.6f}"
    )

    plt.axvline(a0, color="gray", linestyle=":", alpha=0.7)
    plt.axvline(b0, color="gray", linestyle=":", alpha=0.7)

    plt.title("Метод Пауелла · Квадратична апроксимація")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
#головна програма
if __name__ == "__main__":

    x0 = 1.15
    delta0 = 0.1

    sigma_f = 0.01
    sigma_x = 0.001

    eps = 0.01

    print("=" * 80)
    print("Метод Пауела (квадратична апроксимація)")
    print("=" * 80)

    print(
        "Цільова функція: "
        "f(x) = 1/2(x² + 1) arctan(x) − 3/2 x"
    )
    print(f"Початкова точка x0 = {x0}")
    print(
        f"Точний мінімум: x* = {X_EXACT:.10f}, "
        f"f* = {F_EXACT:.10f}\n"
    )

    # Метод Свенна
    a0, b0, swann_pts, n_swann = swann(x0, delta0)

    #Метод Пауелла
    interval_len = b0 - a0

    dx = (
        interval_len / 3.0
        if interval_len > 1e-12
        else 0.3
    )

    x1_start = a0

    x_p, f_p, n_powell, history = powell(
        x1_start,
        dx=dx,
        sigma_f=sigma_f,
        sigma_x=sigma_x
    )

    it_powell = len(history)

    #Методи виключення інтервалів
    print("*" * 90)
    print("Методи виключення інтервалів для порівняння")
    print(f"ε = {eps}")
    print("*" * 90)

    x_g, f_g, n_gold, it_gold, L_gold = golden_section(
        a0, b0, eps
    )

    print(
        f"Золотий переріз: x* = {x_g:.8f}, "
        f"f* = {f_g:.8f}, Nf = {n_gold}, "
        f"n = {it_gold}, Ln = {L_gold:.6f}"
    )

    x_f, f_f, n_fib, it_fib, L_fib = fibonacci_search(
        a0, b0, eps
    )

    print(
        f"Фібоначчі:       x* = {x_f:.8f}, "
        f"f* = {f_f:.8f}, Nf = {n_fib}, "
        f"n = {it_fib}, Ln = {L_fib:.6f}"
    )

    #Зведена таблиця порівняння
    print("\n")
    print("=" * 125)
    print("ПОорівняння методів")
    print("=" * 125)

    print(
        f"{'Метод':<27} | {'Nf':>5} | {'n':>4} | "
        f"{'x*':>12} | {'f(x*)':>12} | "
        f"{'Абс. похибка':>14} | {'Примітка':<15}"
    )

    print("-" * 125)

    print(
        f"{'Пауелл':<27} | {n_powell:>5} | "
        f"{it_powell:>4} | {x_p:>12.8f} | "
        f"{f_p:>12.8f} | "
        f"{abs(x_p - X_EXACT):>14.2e} | "
        f"{'квадратична':<15}"
    )

    print(
        f"{'Фібоначчі (ε=0.01)':<27} | {n_fib:>5} | "
        f"{it_fib:>4} | {x_f:>12.8f} | "
        f"{f_f:>12.8f} | "
        f"{abs(x_f - X_EXACT):>14.2e} | "
        f"{'інтервальний':<15}"
    )

    print(
        f"{'Золотий переріз (ε=0.01)':<27} | {n_gold:>5} | "
        f"{it_gold:>4} | {x_g:>12.8f} | "
        f"{f_g:>12.8f} | "
        f"{abs(x_g - X_EXACT):>14.2e} | "
        f"{'інтервальний':<15}"
    )

    print("=" * 125)

    print(f"Точний мінімум: x*_точн = {X_EXACT:.10f}")
    print(f"Метод Свенна: інтервал [{a0:.6f}; {b0:.6f}]")
    print(f"Обчислень f(x) у методі Свенна: {n_swann}")
    print()

    #Графік
    plot_powell(history, x_p, f_p, a0, b0)

    print("Роботу завершено.")