import numpy as np
import matplotlib.pyplot as plt

# частина 1
def F(x):
    x1, x2 = float(x[0]), float(x[1])
    return 4*x1**2 + 2*x2**2 + 2*x1*x2 + 2*x1 - 10*x2

def gradF(x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        8*x1 + 2*x2 + 2,
        2*x1 + 4*x2 - 10
    ], dtype=float)

# Лінійне обмеження: 2*x1 - x2 >= 6  =>  g1(x) = 6 - 2*x1 + x2 <= 0
def g1(x):
    return 6.0 - 2.0*float(x[0]) + float(x[1])

def grad_g1(x):
    return np.array([-2.0, 1.0], dtype=float)

cons = [(g1, grad_g1)]
x0 = np.array([-5.0, 0.0], dtype=float)


# ОДНОВИМІРНИЙ ПОШУК (СВЕНН + ЗОЛОТИЙ ПЕРЕРІЗ) І МЕТОД ФЛЕТЧЕРА-РІВСА
def svenn_bracket(phi, h0=0.0, delta=0.01, max_steps=100):
    f0 = phi(h0)
    f_plus = phi(h0 + delta)
    f_minus = phi(h0 - delta)

    if f_plus <= f0:
        step = delta
        prev_h, curr_h = h0, h0 + step
        prev_f, curr_f = f0, f_plus
        k = 1
        while curr_f <= prev_f and k < max_steps:
            step *= 2.0
            next_h = curr_h + step
            next_f = phi(next_h)
            prev_h, curr_h = curr_h, next_h
            prev_f, curr_f = curr_f, next_f
            k += 1
        a = min(h0, curr_h - step) if step > delta else min(prev_h, curr_h)
        b = max(h0, curr_h)
        return a, b
    elif f_minus <= f0:
        step = -delta
        prev_h, curr_h = h0, h0 + step
        prev_f, curr_f = f0, f_minus
        k = 1
        while curr_f <= prev_f and k < max_steps:
            step *= 2.0
            next_h = curr_h + step
            next_f = phi(next_h)
            prev_h, curr_h = curr_h, next_h
            prev_f, curr_f = curr_f, next_f
            k += 1
        return min(prev_h, curr_h), max(prev_h, curr_h)
    else:
        return h0 - delta, h0 + delta


def golden_section(phi, a, b, tol=1e-6, max_iter=200):
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = phi(c), phi(d)
    it = 0

    while abs(b - a) > tol and it < max_iter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = phi(d)
        it += 1

    h_star = 0.5 * (a + b)
    return h_star


def fletcher_reeves(func, grad_func, x_start, eps_grad=0.01, max_iter=300, log_inner=False):
    x = np.array(x_start, dtype=float)
    n_dim = len(x)
    n_f = 0
    n_g = 0

    def eval_f(pt):
        nonlocal n_f
        n_f += 1
        return float(func(pt))

    def eval_g(pt):
        nonlocal n_g
        n_g += 1
        return np.array(grad_func(pt), dtype=float)

    g = eval_g(x)
    d = -g.copy()
    history = []

    for k in range(max_iter):
        if np.linalg.norm(g) <= eps_grad:
            break

        if np.dot(g, d) >= 0:
            d = -g.copy()

        def phi_1d(h):
            return eval_f(x + h * d)

        a, b = svenn_bracket(phi_1d, h0=0.0, delta=0.01)
        h_opt = golden_section(phi_1d, a, b, tol=1e-6)

        x_next = x + h_opt * d
        g_next = eval_g(x_next)

        if log_inner:
            history.append({
                "iter": k + 1,
                "x_prev": x.copy(),
                "d": d.copy(),
                "h": h_opt,
                "x_next": x_next.copy(),
                "norm_g_next": np.linalg.norm(g_next)
            })

        if (k + 1) % n_dim == 0:
            d = -g_next.copy()
        else:
            denom = np.dot(g, g)
            beta = np.dot(g_next, g_next) / denom if denom > 1e-30 else 0.0
            d = -g_next + beta * d

        x, g = x_next, g_next

    return {
        "x": x,
        "iters": len(history),
        "n_f": n_f,
        "n_g": n_g,
        "history": history
    }



#ШТРАФНА ФУНКЦІЯ ТА ЗОВНІШНІЙ ЦИКЛ SUMT
def penalty_P(x, cons):
    return sum(max(0.0, float(g(x))) ** 2 for g, _ in cons)


def make_penalty_objective(func, grad_func, cons, r):
    def Phi(x):
        return float(func(x)) + r * penalty_P(x, cons)

    def gradPhi(x):
        v = np.array(grad_func(x), dtype=float)
        for g, gg in cons:
            s = max(0.0, float(g(x)))
            if s > 0.0:
                v += 2.0 * r * s * np.array(gg(x), dtype=float)
        return v

    return Phi, gradPhi


def run_sumt(func, grad_func, cons, x0, r0=1.0, c=10.0, eps=0.01, eps_inner=0.01):
    x_curr = np.array(x0, dtype=float)
    r = float(r0)
    total_nf, total_ng = 0, 0
    protocol = []
    inner_r1_log = None

    for k in range(20):
        start_pt = x_curr.copy()
        Phi, gradPhi = make_penalty_objective(func, grad_func, cons, r)
        res_fr = fletcher_reeves(
            Phi, gradPhi, start_pt,
            eps_grad=eps_inner,
            log_inner=(abs(r - 1.0) < 1e-12)
        )
        if abs(r - 1.0) < 1e-12:
            inner_r1_log = res_fr["history"]

        x_curr = res_fr["x"]
        total_nf += res_fr["n_f"]
        total_ng += res_fr["n_g"]

        f_val = float(func(x_curr))
        p_val = penalty_P(x_curr, cons)
        rP_val = r * p_val
        viol = max(0.0, float(g1(x_curr)))
        lam_est = 2.0 * r * viol

        protocol.append({
            "k": k + 1,
            "r": r,
            "start": start_pt,
            "fr_iters": res_fr["iters"],
            "n_Phi": res_fr["n_f"],
            "n_gradPhi": res_fr["n_g"],
            "x": x_curr.copy(),
            "F": f_val,
            "g": viol,
            "rP": rP_val,
            "lam": lam_est,
            "total_n_Phi": total_nf
        })

        if rP_val <= eps:
            break
        r *= c

    return x_curr, protocol, inner_r1_log


#АНАЛІТИЧНА ФОРМУЛА x(r) ДЛЯ ПЕРЕВІРКИ КОЖНОГО КРОКУ 
def analytical_xr(r):
    # Розв'язок системи nabla_Phi_r(x) = 0 при g1(x) > 0:
    # (8 + 8r)*x1 + (2 - 4r)*x2 = 24r - 2
    # (2 - 4r)*x1 + (4 + 2r)*x2 = -12r + 10
    A = np.array([
        [8.0 + 8.0*r, 2.0 - 4.0*r],
        [2.0 - 4.0*r, 4.0 + 2.0*r]
    ], dtype=float)
    b = np.array([24.0*r - 2.0, -12.0*r + 10.0], dtype=float)
    return np.linalg.solve(A, b)


def project_to_boundary(x):
    # x_hat = x - g1(x) * grad_g1 / ||grad_g1||^2, де grad_g1 = (-2, 1), ||grad_g1||^2 = 5
    gv = g1(x)
    if gv > 0:
        return x - (gv / 5.0) * grad_g1(x)
    return x.copy()



# ЗАПУСК І ВИВЕДЕННЯ ПРОТОКОЛІВ ТА ГРАФІКІВ
if __name__ == "__main__":
    x_unc = np.array([-1.0, 3.0])
    x_star = np.array([2.4375, -1.125])
    f_star = 36.9375
    lam_star = 9.625

    print("=== 1. АНАЛІТИЧНИЙ КОНТРОЛЬ ===")
    print(f"Безумовний мінімум: x_unc = ({x_unc[0]:.4f}; {x_unc[1]:.4f}), F = {F(x_unc):.4f}, g1 = {g1(x_unc):.4f} > 0")
    print(f"Умовний мінімум:   x*    = ({x_star[0]:.4f}; {x_star[1]:.4f}), F* = {f_star:.4f}, lambda* = {lam_star:.4f}\n")

    x_final, protocol, inner_r1 = run_sumt(F, gradF, cons, x0, r0=1.0, c=10.0, eps=0.01, eps_inner=0.01)

    print("=== 2. ПРОТОКОЛ ЗОВНІШНІХ ІТЕРАЦІЙ (r0=1, c=10, eps=0.01) ===")
    header = (
        f"{'k':>2} | {'r':>7} | {'старт':>17} | {'іт.ФР':>5} | {'N_Phi':>5} | {'N_gPhi':>6} | "
        f"{'x(r) чисельно':>19} | {'x(r) аналіт.':>19} | {'F(x(r))':>9} | {'g(x(r))':>9} | {'r*P':>8} | {'2*r*g':>7} | {'N_Phi сум':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in protocol:
        xr_an = analytical_xr(row["r"])
        st_s = f"({row['start'][0]:.2f};{row['start'][1]:.2f})"
        xr_s = f"({row['x'][0]:.4f};{row['x'][1]:.4f})"
        an_s = f"({xr_an[0]:.4f};{xr_an[1]:.4f})"
        print(
            f"{row['k']:>2} | {row['r']:>7.4g} | {st_s:>17} | {row['fr_iters']:>5} | {row['n_Phi']:>5} | {row['n_gradPhi']:>6} | "
            f"{xr_s:>19} | {an_s:>19} | {row['F']:>9.4f} | {row['g']:>9.6f} | {row['rP']:>8.5f} | {row['lam']:>7.3f} | {row['total_n_Phi']:>9}"
        )

    print("\n=== 3. ПОВНИЙ ПРОТОКОЛ ВНУТРІШНЬОЇ ЗАДАЧІ ДЛЯ r = 1 ===")
    for step in inner_r1:
        print(
            f"Іт. {step['iter']}: x=({step['x_prev'][0]:.4f}; {step['x_prev'][1]:.4f}), "
            f"d=({step['d'][0]:.4f}; {step['d'][1]:.4f}), h={step['h']:.6f} -> "
            f"x_next=({step['x_next'][0]:.4f}; {step['x_next'][1]:.4f}), ||gradPhi||={step['norm_g_next']:.2e}"
        )

    # Дошліфовування проєкцією та порівняння з r = 10^6 одразу
    x_proj = project_to_boundary(x_final)
    Phi_1e6, gradPhi_1e6 = make_penalty_objective(F, gradF, cons, 1e6)
    res_1e6 = fletcher_reeves(Phi_1e6, gradPhi_1e6, x0, eps_grad=0.01, max_iter=1500)
    x_1e6 = res_1e6["x"]

    print("\n=== 4. ДОШЛІФОВУВАННЯ ТА ПОРІВНЯННЯ З r = 10^6 ОДРАЗУ ===")
    print(f"Зупинка SUMT (r={protocol[-1]['r']:.4g}): x=({x_final[0]:.6f}; {x_final[1]:.6f}), "
          f"g={g1(x_final):.6f}, F={F(x_final):.6f}, N_Phi={protocol[-1]['total_n_Phi']}")
    print(f"+ Проєкція на межу:               x_hat=({x_proj[0]:.6f}; {x_proj[1]:.6f}), "
          f"g={g1(x_proj):.6e}, F={F(x_proj):.6f}, +0 обчислень")
    print(f"Одразу r = 10^6 зі старту x0:     x=({x_1e6[0]:.6f}; {x_1e6[1]:.6f}), "
          f"g={g1(x_1e6):.6e}, F={F(x_1e6):.6f}, іт. ФР={res_1e6['iters']}, N_Phi={res_1e6['n_f']}")

    # Графіки
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    xs = np.linspace(-6, 5, 300)
    ys = np.linspace(-5, 5, 300)
    X1, X2 = np.meshgrid(xs, ys)
    Z = 4*X1**2 + 2*X2**2 + 2*X1*X2 + 2*X1 - 10*X2

    cs = ax1.contour(X1, X2, Z, levels=22, colors="steelblue", alpha=0.65, linewidths=1.0)
    ax1.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    # Межа 2*x1 - x2 = 6  =>  x2 = 2*x1 - 6
    line_x1 = np.linspace(-6, 5, 200)
    line_x2 = 2.0 * line_x1 - 6.0
    ax1.plot(line_x1, line_x2, "g-", linewidth=2.2, label="Межа 2x1 - x2 = 6")

    # Допустима область: 2*x1 - x2 >= 6
    feas = (2.0*X1 - X2) >= 6.0
    ax1.contourf(X1, X2, feas.astype(int), levels=[0.5, 1.5], colors=["#a8e6a3"], alpha=0.35)

    traj = np.vstack([x0] + [row["x"] for row in protocol])
    ax1.plot(traj[:, 0], traj[:, 1], "o--", color="navy", linewidth=1.8, markersize=5, label="Траєкторія x(r)")
    ax1.scatter([x0[0]], [x0[1]], color="black", s=60, zorder=5, label="Старт x_(0) = (-5; 0)")
    ax1.scatter([x_unc[0]], [x_unc[1]], color="red", s=60, zorder=5, label="Безумовний min (-1; 3)")
    ax1.scatter([x_star[0]], [x_star[1]], color="darkgreen", s=70, zorder=6, label="Умовний min x*")

    ax1.set_xlim(-6, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Лінії рівня F(x), допустима множина Ω і траєкторія x(r)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)

    r_vals = [row["r"] for row in protocol]
    g_vals = [max(row["g"], 1e-16) for row in protocol]
    est_vals = [lam_star / (2.0 * rv) for rv in r_vals]

    ax2.loglog(r_vals, g_vals, "s-", color="crimson", linewidth=2, markersize=6, label="Порушення g(x(r))")
    ax2.loglog(r_vals, est_vals, "k--", linewidth=1.4, label="Теоретична оцінка λ* / (2r)")
    ax2.set_xlabel("Коефіцієнт штрафу r (log)")
    ax2.set_ylabel("Порушення g(x(r)) (log)")
    ax2.set_title("Спадання порушення обмеження зі зростанням r")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.show()
