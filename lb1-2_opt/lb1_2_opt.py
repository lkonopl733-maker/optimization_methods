import math
import matplotlib.pyplot as plt


#цільова функція

def f(x):
    return 0.5 * (x**2 + 1) * math.atan(x) - 1.5 * x

#Свенн

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

    # Перевірка напрямку
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
            # Випадок, коли f(x0-дельта) >= f(x0) <= f(x0+дельта)
            a0 = x0 - abs(delta0)
            b0 = x0 + abs(delta0)
            print("Мінімум знаходиться в початковому окролі x0.")
            print(f"[a0, b0] = [{a0:.6f}; {b0:.6f}]")
            print("Обчислень f(x): 3")
            return a0, b0, [a0, x0, b0], 3

    print("\nТаблиця ітерацій:")
    print("*" * 75)
    print(f"{'k':>3} | {'Крок (2^k * дельт)':>16} | {'xk':>12} | {'f(xk)':>14} | {'f(xk) < f(xk-1)':>16}")
    print("*" * 75)
    print(f"{0:>3} | {'—':>16} | {points[0]:>12.6f} | {values[0]:>14.8f} | {'—':>16}")
    print(f"{1:>3} | {delta:>16.6f} | {points[1]:>12.6f} | {values[1]:>14.8f} | {'+':>16}")

    k = 1
    # Основний цикл розгону з подвоєнням кроку
    while True:
        step = (2 ** k) * delta
        x_next = points[-1] + step
        f_next = f(x_next)

        points.append(x_next)
        values.append(f_next)
        k += 1

        is_decreasing = f_next < values[-2]
        comparison = "+" if is_decreasing else "−"
        print(f"{k:>3} | {step:>16.6f} | {x_next:>12.6f} | {f_next:>14.8f} | {comparison:>16}")

        if not is_decreasing:
            break

    # Визначення кінцевого інтервалу [a0, b0] за [x_{k-2}, x_k]
    a0 = min(points[-3], points[-1])
    b0 = max(points[-3], points[-1])
    calc_count = len(values)  # Точна кількість обчислень f(x)

    print("*" * 75)
    print(f"\nІнтервал локалізації [a0, b0] = [{a0:.6f}; {b0:.6f}]")
    print(f"Довжина інтервалу L0 = {b0 - a0:.6f}")
    print(f"Всього обчислень f(x): {calc_count}\n")

    return a0, b0, points, calc_count


#Дихотомія

def dichotomy(a0, b0, sigma=0.01, eps=0.001):
    print("*" * 120)
    print("Метод дихотомії")
    print("*" * 120)
    print(f"[a0, b0] = [{a0:.6f}; {b0:.6f}]")
    print(f"σ = {sigma}")
    print(f"ε = {eps}\n")

    a = a0
    b = b0
    k = 0
    calc_count = 0
    points = []

    print("Таблиця ітерацій:")
    print("*" * 120)
    print(f"{'k':>3} | {'x1':>11} | {'x2':>11} | {'f(x1)':>14} | {'f(x2)':>14} | {'[ak, bk]':>27} | {'Lk':>10}")
    print("*" * 120)
    print(f"{0:>3} | {'—':>11} | {'—':>11} | {'—':>14} | {'—':>14} | [{a:>10.6f}; {b:>10.6f}] | {b-a:>10.6f}")

    while (b - a) > sigma:
        x1 = (a + b) / 2 - eps / 2
        x2 = x1 + eps

        f1 = f(x1)
        f2 = f(x2)
        calc_count += 2
        points.extend([x1, x2])

        if f1 < f2:
            b = x2
        else:
            a = x1

        k += 1
        print(f"{k:>3} | {x1:>11.6f} | {x2:>11.6f} | {f1:>14.8f} | {f2:>14.8f} | [{a:>10.6f}; {b:>10.6f}] | {b-a:>10.6f}")

    x_star = (a + b) / 2
    f_star = f(x_star)
    calc_count += 1
    points.append(x_star)

    print("*" * 120)
    print(f"\nx* = {x_star:.8f}")
    print(f"f(x*) = {f_star:.8f}")
    print(f"Ітерацій: {k}")
    print(f"Обчислень f(x): {calc_count}\n")

    return x_star, f_star, calc_count, points


#половиний поділ

def half_division(a0, b0, sigma=0.01):
    print("*" * 120)
    print("Метод половинного поділу")
    print("*" * 120)
    print(f"[a0, b0] = [{a0:.6f}; {b0:.6f}]")
    print(f"σ = {sigma}\n")

    a = a0
    b = b0

    xm = (a + b) / 2
    fm = f(xm)
    calc_count = 1
    k = 0

    points = [xm]

    print("Таблиця ітерацій:")
    print("*" * 120)
    print(f"{'k':>3} | {'x1':>11} | {'xm':>11} | {'x2':>11} | {'f(x1)':>13} | {'f(xm)':>13} | {'f(x2)':>13} | {'Lk':>10}")
    print("*" * 120)
    print(f"{0:>3} | {'—':>11} | {xm:>11.6f} | {'—':>11} | {'—':>13} | {fm:>13.8f} | {'—':>13} | {b-a:>10.6f}")

    while (b - a) > sigma:
        L = b - a
        x1 = a + L / 4
        x2 = b - L / 4

        f1 = f(x1)
        f2 = f(x2)

        xm_old = xm
        fm_old = fm
        calc_count += 2
        points.extend([x1, x2])

        # Правило виключення інтервалу (триточковий алгоритм)
        if f1 < fm_old:
            b = xm_old
            xm = x1
            fm = f1
        elif f2 < fm_old:
            a = xm_old
            xm = x2
            fm = f2
        else:
            a = x1
            b = x2

        k += 1
        print(f"{k:>3} | {x1:>11.6f} | {xm_old:>11.6f} | {x2:>11.6f} | {f1:>13.8f} | {fm_old:>13.8f} | {f2:>13.8f} | {b-a:>10.6f}")

    x_star = (a + b) / 2
    f_star = f(x_star)
    calc_count += 1
    points.append(x_star)

    print("*" * 125)
    print(f"\nx* = {x_star:.8f}")
    print(f"f(x*) = {f_star:.8f}")
    print(f"Ітерацій: {k}")
    print(f"Обчислень f(x): {calc_count}\n")

    return x_star, f_star, calc_count, points


#графіки

def plot_function(a0, b0, swann_points, dichotomy_points, half_points, x_d, x_h):
    left = a0 - 0.15
    right = b0 + 0.15
    x_vals = [left + i * (right - left) / 500 for i in range(501)]
    y_vals = [f(x) for x in x_vals]

    # свенн
    plt.figure(figsize=(9, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.scatter(swann_points, [f(x) for x in swann_points], color="orange", s=50, zorder=5, label="Точки Свенна")
    plt.axvline(a0, color="gray", linestyle="--", label=f"a0 = {a0:.2f}")
    plt.axvline(b0, color="gray", linestyle="--", label=f"b0 = {b0:.2f}")
    plt.title("Метод Свенна: Знаходження інтервалу локалізації")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # дихотометрія
    plt.figure(figsize=(9, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.scatter(dichotomy_points[:-1], [f(x) for x in dichotomy_points[:-1]], color="green", marker="x", s=40, label="Пробні точки")
    plt.scatter([x_d], [f(x_d)], color="red", s=120, marker="*", zorder=6, label=f"x* = {x_d:.5f}")
    plt.axvline(a0, color="gray", linestyle="--")
    plt.axvline(b0, color="gray", linestyle="--")
    plt.title("Метод дихотомії")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # половиний поділ
    plt.figure(figsize=(9, 5))
    plt.plot(x_vals, y_vals, label="f(x)", color="blue")
    plt.scatter(half_points[:-1], [f(x) for x in half_points[:-1]], color="purple", marker="^", s=40, label="Пробні точки")
    plt.scatter([x_h], [f(x_h)], color="black", s=100, marker="P", zorder=6, label=f"x* = {x_h:.5f}")
    plt.axvline(a0, color="gray", linestyle="--")
    plt.axvline(b0, color="gray", linestyle="--")
    plt.title("Метод половинного поділу")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # порівняльний з наближенням біля min
    plt.figure(figsize=(9, 5))
    x_zoom = [a0 + i * (b0 - a0) / 300 for i in range(301)]
    plt.plot(x_zoom, [f(x) for x in x_zoom], label="f(x)", color="blue")
    plt.scatter([x_d], [f(x_d)], color="red", s=120, marker="*", label=f"Дихотомія x* = {x_d:.5f}")
    plt.scatter([x_h], [f(x_h)], color="black", s=80, marker="P", label=f"Пол. поділ x* = {x_h:.5f}")
    plt.title("Порівняння знайдених мінімумів (Масштаб [a0, b0])")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


#осовна функція

if __name__ == "__main__":
    x0 = 1.15
    delta0 = 0.1
    sigma = 0.01
    eps = 0.001
    print("f(x) = 1/2 (x² + 1) arctan(x) - 3/2 x")
    print(f"x0 = {x0}")
    

    # свенн
    a0, b0, swann_pts, n_swann = swann(x0, delta0)

    # дихотомія
    x_d, f_d, n_dichotomy, dich_pts = dichotomy(a0, b0, sigma, eps)

    # половиний поділ
    x_h, f_h, n_half, half_pts = half_division(a0, b0, sigma)

    # Підсумок
    print("*" * 50)
    print("порівняння методів")
    print("*" * 50)
    print(f"Метод Свенна:             [a0, b0] = [{a0:.6f}; {b0:.6f}],  обчислень f(x) = {n_swann}")
    print(f"Метод дихотомії:          x* = {x_d:.8f}, f(x*) = {f_d:.8f}, обчислень f(x) = {n_dichotomy}")
    print(f"Метод половинного поділу: x* = {x_h:.8f}, f(x*) = {f_h:.8f}, обчислень f(x) = {n_half}")
    

    # Графічна візуалізація
    plot_function(a0, b0, swann_pts, dich_pts, half_pts, x_d, x_h)
