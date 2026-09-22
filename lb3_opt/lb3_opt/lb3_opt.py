import math
import matplotlib.pyplot as plt

# Цільова функція 
def f(x): 
    return 0.5 * (x**2 + 1) * math.atan(x) - 1.5 * x 

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
            print("Обчислень f(x): 3") 
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

# Метод Золотого Перерізу 
def golden_section(a0, b0, eps=0.01): 
    print("*" * 120) 
    print("Метод Золотого Перерізу") 
    print("*" * 120) 
    print(f"[a0, b0] = [{a0:.6f}; {b0:.6f}]") 
    print(f"ε = {eps}\n") 

    tau = (math.sqrt(5) - 1) / 2 
    a, b = a0, b0 

    x1 = a + (1 - tau) * (b - a) 
    x2 = a + tau * (b - a) 

    f1 = f(x1) 
    f2 = f(x2) 
    calc_count = 2 

    k = 1 
    points = [x1, x2] 

    print("Таблиця ітерацій:") 
    print("*" * 135) 
    print(f"{'k':>3} | {'ak':>11} | {'bk':>11} | {'x1':>11} | {'x2':>11} | {'f(x1)':>13} | {'f(x2)':>13} | {'Lk':>10} | {'Рішення / Успадкування':>30}") 
    print("*" * 135) 

    while (b - a) > eps: 
        L_k = b - a 

        if f1 <= f2: 
            decision = "f1<=f2 -> b=x2 (успадковано x2)" 
            print(f"{k:>3} | {a:>11.6f} | {b:>11.6f} | {x1:>11.6f} | {x2:>11.6f} | {f1:>13.8f} | {f2:>13.8f} | {L_k:>10.6f} | {decision:>30}") 
            b = x2 
            x2, f2 = x1, f1 
            x1 = a + (1 - tau) * (b - a) 
            f1 = f(x1) 
            points.append(x1) 
        else: 
            decision = "f1>f2  -> a=x1 (успадковано x1)" 
            print(f"{k:>3} | {a:>11.6f} | {b:>11.6f} | {x1:>11.6f} | {x2:>11.6f} | {f1:>13.8f} | {f2:>13.8f} | {L_k:>10.6f} | {decision:>30}") 
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

    print("*" * 135) 
    print(f"\nx* = {x_star:.8f}") 
    print(f"f(x*) = {f_star:.8f}") 
    print(f"Кількість ітерацій: {k - 1}") 
    print(f"Кінцева довжина інтервалу Ln = {L_n:.8f}") 
    print(f"Обчислень f(x): {calc_count}\n") 

    return x_star, f_star, calc_count, points, k - 1, L_n 

# дихотомія
def dichotomy(a0, b0, eps=0.01, delta=0.001): 
    a, b = a0, b0 
    k = 0 
    calc_count = 0 
    points = [] 

    while (b - a) > eps: 
        x1 = (a + b) / 2 - delta / 2 
        x2 = (a + b) / 2 + delta / 2 

        f1 = f(x1) 
        f2 = f(x2) 
        calc_count += 2 
        points.extend([x1, x2]) 

        if f1 < f2: 
            b = x2 
        else: 
            a = x1 
        k += 1 

    x_star = (a + b) / 2 
    f_star = f(x_star) 
    points.append(x_star) 
    L_n = b - a 

    return x_star, f_star, calc_count, points, k, L_n 

# Метод Половинного поділу 
def half_division(a0, b0, eps=0.01): 
    a, b = a0, b0 
    xm = (a + b) / 2 
    fm = f(xm) 
    calc_count = 1 
    k = 0 
    points = [xm] 

    while (b - a) > eps: 
        L = b - a 
        x1 = a + L / 4 
        x2 = b - L / 4 

        f1 = f(x1) 
        f2 = f(x2) 

        calc_count += 2 
        points.extend([x1, x2]) 

        if f1 < fm: 
            b = xm 
            xm, fm = x1, f1 
        elif f2 < fm: 
            a = xm 
            xm, fm = x2, f2 
        else: 
            a, b = x1, x2 
            

        k += 1 

    x_star = xm
    f_star = fm
    points.append(x_star) 
    L_n = b - a 

    return x_star, f_star, calc_count, points, k, L_n 

# Візуалізація
def plot_function(a0, b0, gold_pts, x_d, x_h, x_g): 
    left = a0 - 0.15 
    right = b0 + 0.15 
    x_vals = [left + i * (right - left) / 500 for i in range(501)] 
    y_vals = [f(x) for x in x_vals] 

    # Графік Золотого перерізу 
    plt.figure(figsize=(9, 5)) 
    plt.plot(x_vals, y_vals, label="f(x)", color="blue") 
    plt.scatter(gold_pts[:-1], [f(x) for x in gold_pts[:-1]], color="gold", edgecolor="black", marker="o", s=50, label="Пробні точки (Золотий переріз)") 
    plt.scatter([x_g], [f(x_g)], color="red", s=120, marker="*", zorder=6, label=f"x* = {x_g:.5f}") 
    plt.axvline(a0, color="gray", linestyle="--") 
    plt.axvline(b0, color="gray", linestyle="--") 
    plt.title("Метод золотого перерізу") 
    plt.xlabel("x") 
    plt.ylabel("f(x)") 
    plt.grid(True) 
    plt.legend() 
    plt.tight_layout() 

    # Порівняльний графік
    plt.figure(figsize=(9, 5)) 
    x_zoom = [a0 + i * (b0 - a0) / 300 for i in range(301)] 
    plt.plot(x_zoom, [f(x) for x in x_zoom], label="f(x)", color="blue") 
    plt.scatter([x_d], [f(x_d)], color="red", s=120, marker="*", label=f"Дихотомія x* = {x_d:.5f}") 
    plt.scatter([x_h], [f(x_h)], color="black", s=80, marker="P", label=f"Пол. поділ x* = {x_h:.5f}") 
    plt.scatter([x_g], [f(x_g)], color="green", s=90, marker="D", label=f"Золотий переріз x* = {x_g:.5f}") 
    plt.title("Порівняння знайдених мінімумів (Масштаб [a0, b0])") 
    plt.xlabel("x") 
    plt.ylabel("f(x)") 
    plt.grid(True) 
    plt.legend() 
    plt.tight_layout() 

    plt.show() 

# Головна програма
if __name__ == "__main__": 
    x0 = 1.15 
    delta0 = 0.1 
    eps = 0.01 
    delta_dich = 0.001 

    print("f(x) = 0.5 * (x^2 + 1) * arctan(x) - 1.5 * x") 
    print(f"x0 = {x0}\n") 

    # Свенн 
    a0, b0, swann_pts, n_swann = swann(x0, delta0) 

    # Золотий переріз  
    x_g, f_g, n_gold, gold_pts, iter_gold, L_gold = golden_section(a0, b0, eps) 

    # Дихотомія 
    x_d, f_d, n_dich, dich_pts, iter_dich, L_dich = dichotomy(a0, b0, eps, delta_dich) 

    # Половинний поділ  
    x_h, f_h, n_half, half_pts, iter_half, L_half = half_division(a0, b0, eps) 

    # Обчислення коефіцієнтів стиснення 
    L0 = b0 - a0 
    eta_gold = (L_gold / L0) ** (1 / n_gold) 
    eta_dich = (L_dich / L0) ** (1 / n_dich) 
    eta_half = (L_half / L0) ** (1 / n_half) 

    # Підсумкова таблиця 
    print("=" * 125) 
    print(f"{'Метод':<25} | {'x*':<12} | {'f(x*)':<12} | {'Ітерацій':<10} | {'Обчислень f(x)':<15} | {'Ln':<12} | {'Коеф. стиснення (eta)':<20}") 
    print("=" * 125) 

    print(f"{'Золотий переріз':<25} | {x_g:<12.8f} | {f_g:<12.8f} | {iter_gold:<10} | {n_gold:<15} | {L_gold:<12.8f} | {eta_gold:<20.4f}") 
    print(f"{'Дихотомія':<25} | {x_d:<12.8f} | {f_d:<12.8f} | {iter_dich:<10} | {n_dich:<15} | {L_dich:<12.8f} | {eta_dich:<20.4f}") 
    print(f"{'Половинний поділ':<25} | {x_h:<12.8f} | {f_h:<12.8f} | {iter_half:<10} | {n_half:<15} | {L_half:<12.8f} | {eta_half:<20.4f}") 

    print("=" * 125) 

    plot_function(a0, b0, gold_pts, x_d, x_h, x_g)