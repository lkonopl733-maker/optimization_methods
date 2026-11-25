import numpy as np
import matplotlib.pyplot as plt


# базові функції 
def F(x):
    x1, x2 = float(x[0]), float(x[1])
    return 4.0*x1**2 + 2.0*x2**2 + 2.0*x1*x2 + 2.0*x1 - 10.0*x2

def gradF(x):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([
        8.0*x1 + 2.0*x2 + 2.0,
        2.0*x1 + 4.0*x2 - 10.0
    ], dtype=float)

# Обмеження 1: 2*x1 - x2 >= 6  <=>  g1(x) = 6 - 2*x1 + x2 <= 0
def g1(x):
    return 6.0 - 2.0*float(x[0]) + float(x[1])

def grad_g1(x):
    return np.array([-2.0, 1.0], dtype=float)

# Обмеження 2 з таблиці варіантів: x2 <= -4  <=>  g2(x) = x2 + 4 <= 0
def g2(x):
    return float(x[1]) + 4.0

def grad_g2(x):
    return np.array([0.0, 1.0], dtype=float)

# Обмеження 3 з таблиці варіантів ("поза колом" R^2 = 16, R = 4):
# x1^2 + x2^2 >= 16  <=>  g3(x) = 16 - x1^2 - x2^2 <= 0
R_CIRCLE = 4.0
def g3_circle(x):
    return R_CIRCLE**2 - float(x[0])**2 - float(x[1])**2

def grad_g3_circle(x):
    return np.array([-2.0*float(x[0]), -2.0*float(x[1])], dtype=float)

cons_base = [(g1, grad_g1)]
x0_base = np.array([-5.0, 0.0], dtype=float)


# ОДНОВИМІРНИЙ ПОШУК (СВЕНН + ЗОЛОТИЙ ПЕРЕРІЗ) ТА ОПТИМІЗАТОРИ
def svenn_bracket(phi, h0=0.0, delta=0.01, max_steps=100):
    f0 = phi(h0)
    if not np.isfinite(f0):
        return h0, h0

    f_plus = phi(h0 + delta)
    f_minus = phi(h0 - delta)

    if np.isfinite(f_plus) and f_plus <= f0:
        step = delta
        prev_h, curr_h = h0, h0 + step
        prev_f, curr_f = f0, f_plus
        k = 1
        while k < max_steps:
            next_h = curr_h + step * 2.0
            next_f = phi(next_h)
            if not np.isfinite(next_f) or next_f > curr_f:
                break
            step *= 2.0
            prev_h, curr_h = curr_h, next_h
            prev_f, curr_f = curr_f, next_f
            k += 1
        a = min(h0, curr_h - step) if step > delta else min(prev_h, curr_h)
        b = max(h0, curr_h)
        return a, b
    elif np.isfinite(f_minus) and f_minus <= f0:
        step = -delta
        prev_h, curr_h = h0, h0 + step
        prev_f, curr_f = f0, f_minus
        k = 1
        while k < max_steps:
            next_h = curr_h + step * 2.0
            next_f = phi(next_h)
            if not np.isfinite(next_f) or next_f > curr_f:
                break
            step *= 2.0
            prev_h, curr_h = curr_h, next_h
            prev_f, curr_f = curr_f, next_f
            k += 1
        return min(prev_h, curr_h), max(prev_h, curr_h)
    else:
        return h0 - delta, h0 + delta


def golden_section(phi, a, b, tol=1e-6, max_iter=200):
    if a == b:
        return a
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
    if not np.isfinite(phi(h_star)):
        # Фолбек на кращу скінченну точку серед c, d, a, b
        candidates = [pt for pt in (c, d, a, b, 0.0) if np.isfinite(phi(pt))]
        if candidates:
            h_star = min(candidates, key=lambda pt: phi(pt))
    return h_star


def fletcher_reeves(func, grad_func, x_start, eps_grad=0.01, max_iter=400, log_inner=False):
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
    converged = False

    for k in range(max_iter):
        if np.linalg.norm(g) <= eps_grad:
            converged = True
            break

        if np.dot(g, d) >= 0:
            d = -g.copy()

        def phi_1d(h):
            return eval_f(x + h * d)

        a, b = svenn_bracket(phi_1d, h0=0.0, delta=0.01)
        h_opt = golden_section(phi_1d, a, b, tol=1e-6)

        x_next = x + h_opt * d
        if not np.isfinite(eval_f(x_next)):
            # Обережне зменшення кроку, якщо точка вийшла за межу допустимої області
            h_opt *= 0.5
            for _ in range(30):
                x_next = x + h_opt * d
                if np.isfinite(eval_f(x_next)):
                    break
                h_opt *= 0.5

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

    if np.linalg.norm(g) <= eps_grad:
        converged = True

    return {
        "x": x,
        "iters": k + 1 if not converged else k,
        "n_f": n_f,
        "n_g": n_g,
        "converged": converged,
        "history": history
    }


def steepest_descent_line_search(func, grad_func, x_start, eps_grad=0.01, max_iter=1500):
    """Антиградієнт із дробленням кроку (сферична метрика, ЛР 8) для порівняння на r=100."""
    x = np.array(x_start, dtype=float)
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

    f_curr = eval_f(x)
    g = eval_g(x)
    iters = 0
    converged = False

    for k in range(max_iter):
        if np.linalg.norm(g) <= eps_grad:
            converged = True
            break
        d = -g
        alpha = 1.0
        rho = 0.5
        c1 = 1e-4
        g_dot_d = np.dot(g, d)

        while alpha > 1e-15:
            x_try = x + alpha * d
            f_try = eval_f(x_try)
            if np.isfinite(f_try) and f_try <= f_curr + c1 * alpha * g_dot_d:
                break
            alpha *= rho

        x = x + alpha * d
        f_curr = eval_f(x)
        g = eval_g(x)
        iters = k + 1

    if np.linalg.norm(g) <= eps_grad:
        converged = True

    return {
        "x": x,
        "iters": iters,
        "n_f": n_f,
        "n_g": n_g,
        "converged": converged
    }


# ЗОВНІШНІЙ ШТРАФ (НЕРІВНОСТІ ТА РІВНОСТІ)
def penalty_P(x, ineq_cons=(), eq_cons=()):
    p_ineq = sum(max(0.0, float(g(x))) ** 2 for g, _ in ineq_cons)
    p_eq = sum(float(h(x)) ** 2 for h, _ in eq_cons)
    return p_ineq + p_eq


def make_external_penalty(func, grad_func, ineq_cons, eq_cons, r):
    def Phi(x):
        return float(func(x)) + r * penalty_P(x, ineq_cons, eq_cons)

    def gradPhi(x):
        v = np.array(grad_func(x), dtype=float)
        for g, gg in ineq_cons:
            s = max(0.0, float(g(x)))
            if s > 0.0:
                v += 2.0 * r * s * np.array(gg(x), dtype=float)
        for h, gh in eq_cons:
            hv = float(h(x))
            v += 2.0 * r * hv * np.array(gh(x), dtype=float)
        return v

    return Phi, gradPhi


def run_external_penalty(
    func, grad_func, x0,
    ineq_cons=(), eq_cons=(),
    r0=1.0, c=10.0, eps=0.01, eps_inner=0.01,
    adaptive_inner=False, max_outer=25, inner_solver=fletcher_reeves
):
    x_curr = np.array(x0, dtype=float)
    r = float(r0)
    total_nf = 0
    total_ng = 0
    rows = []

    for k in range(max_outer):
        start_pt = x_curr.copy()
        eps_in = eps / np.sqrt(r) if adaptive_inner else eps_inner
        Phi, gradPhi = make_external_penalty(func, grad_func, ineq_cons, eq_cons, r)
        res = inner_solver(Phi, gradPhi, start_pt, eps_grad=eps_in)

        x_curr = res["x"]
        total_nf += res["n_f"]
        total_ng += res["n_g"]

        f_val = float(func(x_curr))
        p_val = penalty_P(x_curr, ineq_cons, eq_cons)
        rP_val = r * p_val

        ineq_viols = [max(0.0, float(g(x_curr))) for g, _ in ineq_cons]
        eq_viols = [abs(float(h(x_curr))) for h, _ in eq_cons]
        max_viol = max(ineq_viols + eq_viols) if (ineq_viols or eq_viols) else 0.0

        lam_ineq = [2.0 * r * s for s in ineq_viols]
        lam_eq = [2.0 * r * float(h(x_curr)) for h, _ in eq_cons]

        rows.append({
            "k": k + 1,
            "r": r,
            "start": start_pt,
            "iters": res["iters"],
            "n_Phi": res["n_f"],
            "n_gradPhi": res["n_g"],
            "x": x_curr.copy(),
            "F": f_val,
            "viol": max_viol,
            "ineq_viols": ineq_viols,
            "eq_viols": eq_viols,
            "rP": rP_val,
            "lam_ineq": lam_ineq,
            "lam_eq": lam_eq,
            "total_n_Phi": total_nf,
            "total_n_gradPhi": total_ng,
            "converged": res["converged"]
        })

        if rP_val <= eps:
            break
        r *= c

    return x_curr, rows

#ЛОГАРИФМІЧНИЙ БАР'ЄР (ВНУТРІШНІЙ ШТРАФ)
def make_log_barrier(func, grad_func, ineq_cons, mu):
    def Phi(x):
        gs = [float(g(x)) for g, _ in ineq_cons]
        if max(gs) >= 0.0:
            return float("inf")
        val = float(func(x)) - mu * sum(np.log(-v) for v in gs)
        return float(val) if np.isfinite(val) else float("inf")

    def gradPhi(x):
        gs = [float(g(x)) for g, _ in ineq_cons]
        if max(gs) >= 0.0:
            return np.zeros_like(x, dtype=float)
        v = np.array(grad_func(x), dtype=float)
        for g, gg in ineq_cons:
            gv = float(g(x))
            v += (-mu / gv) * np.array(gg(x), dtype=float)
        return v

    return Phi, gradPhi


def run_log_barrier(func, grad_func, ineq_cons, x_int0, mu0=10.0, c=10.0, eps=0.01, eps_inner=0.01):
    x_curr = np.array(x_int0, dtype=float)
    assert max(float(g(x_curr)) for g, _ in ineq_cons) < 0.0, "Початкова точка має бути строго всередині Ω (g < 0)!"

    mu = float(mu0)
    m_cons = len(ineq_cons)
    total_nf = 0
    total_ng = 0
    rows = []

    for k in range(25):
        Phi, gradPhi = make_log_barrier(func, grad_func, ineq_cons, mu)
        res = fletcher_reeves(Phi, gradPhi, x_curr, eps_grad=eps_inner)
        x_curr = res["x"]
        total_nf += res["n_f"]
        total_ng += res["n_g"]

        f_val = float(func(x_curr))
        gs = [float(g(x_curr)) for g, _ in ineq_cons]
        lam_est = [-mu / gv if gv < 0.0 else np.nan for gv in gs]

        rows.append({
            "k": k + 1,
            "mu": mu,
            "x": x_curr.copy(),
            "g": gs[0] if len(gs) == 1 else gs,
            "F": f_val,
            "lam": lam_est[0] if len(lam_est) == 1 else lam_est,
            "iters": res["iters"],
            "n_Phi": res["n_f"],
            "n_gradPhi": res["n_g"],
            "total_n_Phi": total_nf,
            "total_n_gradPhi": total_ng
        })

        if mu * m_cons <= eps:
            break
        mu /= c

    return x_curr, rows


if __name__ == "__main__":
    x_star_1 = np.array([2.4375, -1.125])
    f_star_1 = F(x_star_1)
    lam_star_1 = 9.625

    
    #МНОЖНИК c = 2, 4, 10, 100 НА БАЗОВІЙ ЗАДАЧІ
    print("=== 1. ВПЛИВ МНОЖНИКА c (r0=1, eps=0.01, eps_inner=0.01) ===")
    print(f"{'c':>4} | {'зовн.іт':>8} | {'кінц.r':>8} | {'N_Phi':>6} | {'N_gPhi':>6} | {'порушення':>10} | {'x(r)':>19} | {'F(x(r))':>9}")
    print("-" * 92)
    c_runs = {}
    for c_val in [2.0, 4.0, 10.0, 100.0]:
        xf, rws = run_external_penalty(F, gradF, x0_base, ineq_cons=cons_base, r0=1.0, c=c_val, eps=0.01, eps_inner=0.01)
        c_runs[c_val] = (xf, rws)
        last = rws[-1]
        xs = f"({xf[0]:.4f};{xf[1]:.4f})"
        print(f"{c_val:>4.0f} | {len(rws):>8} | {last['r']:>8.4g} | {last['total_n_Phi']:>6} | {last['total_n_gradPhi']:>6} | {last['viol']:>10.6f} | {xs:>19} | {last['F']:>9.4f}")

    #ВНУТРІШНЯ ТОЧНІСТЬ eps_inner = 0.1, 0.01, 0.001
    print("\n=== 2. ВПЛИВ ВНУТРІШНЬОЇ ТОЧНОСТІ eps_inner (r0=1, c=10, eps=0.01) ===")
    print(f"{'eps_вн':>7} | {'зовн.іт':>8} | {'N_Phi':>6} | {'N_gPhi':>6} | {'x при кінц.r':>19} | {'||x-x*||':>9} | {'порушення':>10}")
    print("-" * 88)
    for eps_in in [0.1, 0.01, 0.001]:
        xf, rws = run_external_penalty(F, gradF, x0_base, ineq_cons=cons_base, r0=1.0, c=10.0, eps=0.01, eps_inner=eps_in)
        last = rws[-1]
        err_x = np.linalg.norm(xf - x_star_1)
        xs = f"({xf[0]:.4f};{xf[1]:.4f})"
        print(f"{eps_in:>7.3f} | {len(rws):>8} | {last['total_n_Phi']:>6} | {last['total_n_gradPhi']:>6} | {xs:>19} | {err_x:>9.6f} | {last['viol']:>10.6f}")

    #ОБМЕЖЕННЯ-РІВНІСТЬ h(x) = 6 - 2*x1 + x2 = 0 (штраф r*h^2 без зрізки)
    print("\n=== 3. ОБМЕЖЕННЯ-РІВНІСТЬ h(x) = 6 - 2*x1 + x2 = 0 ===")
    eq_cons_1 = [(g1, grad_g1)]
    xf_eq, rws_eq = run_external_penalty(F, gradF, x0_base, eq_cons=eq_cons_1, r0=1.0, c=10.0, eps=0.01, eps_inner=0.01)
    _, rws_ineq = c_runs[10.0]
    print(f"{'r':>7} | {'x(r) рівність':>19} | {'|h(x(r))|':>10} | {'F рівн.':>9} | {'x(r) нерівність':>19} | {'F нерівн.':>9}")
    print("-" * 92)
    for re_row, ri_row in zip(rws_eq, rws_ineq):
        xe_s = f"({re_row['x'][0]:.4f};{re_row['x'][1]:.4f})"
        xi_s = f"({ri_row['x'][0]:.4f};{ri_row['x'][1]:.4f})"
        print(f"{re_row['r']:>7.4g} | {xe_s:>19} | {re_row['viol']:>10.6f} | {re_row['F']:>9.4f} | {xi_s:>19} | {ri_row['F']:>9.4f}")
    print(f"Рівність разом: зовн.іт={len(rws_eq)}, N_Phi={rws_eq[-1]['total_n_Phi']}, N_gPhi={rws_eq[-1]['total_n_gradPhi']}")

    #ЛОГАРИФМІЧНИЙ БАР'ЄР І ДВОСТОРОННЯ ОЦІНКА F(x(r)) <= F* <= F(x(mu))
    print("\n=== 4. ЛОГАРИФМІЧНИЙ БАР'ЄР (СТАРТ ВСЕРЕДИНІ Ω: x_int0 = (5; 0), g1 = -4 < 0) ===")
    x_int0 = np.array([5.0, 0.0], dtype=float)
    xf_bar, rws_bar = run_log_barrier(F, gradF, cons_base, x_int0, mu0=10.0, c=10.0, eps=0.01, eps_inner=0.01)
    print(f"{'k':>2} | {'mu':>7} | {'x(mu)':>19} | {'g1(x(mu))':>11} | {'F(x(mu))':>9} | {'mu/(-g1)->lam':>13} | {'іт./N_Phi':>10} | {'N_Phi сум':>9}")
    print("-" * 102)
    for b in rws_bar:
        xb_s = f"({b['x'][0]:.4f};{b['x'][1]:.4f})"
        it_nf = f"{b['iters']}/{b['n_Phi']}"
        print(f"{b['k']:>2} | {b['mu']:>7.4g} | {xb_s:>19} | {b['g']:>11.6f} | {b['F']:>9.4f} | {b['lam']:>13.4f} | {it_nf:>10} | {b['total_n_Phi']:>9}")

    f_ext_last = rws_ineq[-1]["F"]
    f_bar_last = rws_bar[-1]["F"]
    print(f"Двостороння оцінка F*: F_зовн(x(r)) = {f_ext_last:.4f} <= F* = {f_star_1:.4f} <= F_бар(x(mu)) = {f_bar_last:.4f}")

    #ДВА ОБМЕЖЕННЯ: g1 = 6 - 2x1 + x2 <= 0 та g2 = x2 + 4 <= 0
    print("\n=== 5. ДВА ОБМЕЖЕННЯ (g1: 2x1-x2>=6; g2: x2<=-4) ТА ПЕРЕВІРКА УМОВ ККТ ===")
    cons_two = [(g1, grad_g1), (g2, grad_g2)]
    xf_two, rws_two = run_external_penalty(F, gradF, x0_base, ineq_cons=cons_two, r0=1.0, c=10.0, eps=0.01, eps_inner=0.01)
    print(f"{'r':>7} | {'x(r)':>19} | {'s1':>9} | {'s2':>9} | {'F':>9} | {'r*P':>8} | {'2r*s1':>8} | {'2r*s2':>8}")
    print("-" * 96)
    for row in rws_two:
        xs = f"({row['x'][0]:.4f};{row['x'][1]:.4f})"
        s1, s2 = row["ineq_viols"]
        l1, l2 = row["lam_ineq"]
        print(f"{row['r']:>7.4g} | {xs:>19} | {s1:>9.6f} | {s2:>9.6f} | {row['F']:>9.4f} | {row['rP']:>8.5f} | {l1:>8.4f} | {l2:>8.4f}")

    # Аналітична перевірка ККТ для двох обмежень:
    # Якщо обидва активні: x2* = -4, і з 2*x1* - x2* = 6 => 2*x1* + 4 = 6 => x1* = 1.
    # Перевірка gradF(1, -4) + lam1 * grad_g1 + lam2 * grad_g2 = 0:
    # gradF(1, -4) = (8(1)+2(-4)+2, 2(1)+4(-4)-10) = (2, -24)
    # lam1*(-2, 1) + lam2*(0, 1) = (-2*lam1, lam1 + lam2) = (-2, 24) => lam1 = 1.0, lam2 = 23.0.
    x_kkt_two = np.array([1.0, -4.0])
    lam1_kkt, lam2_kkt = 1.0, 23.0
    print(f"Точний ККТ-розв'язок для двох обмежень: x* = ({x_kkt_two[0]:.4f}; {x_kkt_two[1]:.4f}), "
          f"F* = {F(x_kkt_two):.4f}, lambda = ({lam1_kkt:.3f}; {lam2_kkt:.3f}) [обидва обмеження активні]")

    #КРИВА МЕЖА ("ПОЗА КОЛОМ" x1^2 + x2^2 >= 16) З ДВОХ СТАРТІВ
    print("\n=== 6. КРИВА МЕЖА x1^2 + x2^2 >= 16 (ПОЗА КОЛОМ R=4) З ДВОХ СТАРТІВ ===")
    cons_circle = [(g3_circle, grad_g3_circle)]
    starts_circle = [np.array([-5.0, 0.0]), np.array([5.0, 0.0])]
    circle_runs = []
    for st in starts_circle:
        xf_c, rws_c = run_external_penalty(F, gradF, st, ineq_cons=cons_circle, r0=1.0, c=10.0, eps=0.01, eps_inner=0.01)
        circle_runs.append((st, xf_c, rws_c))
        print(f"Старт ({st[0]:5.1f}; {st[1]:5.1f}) -> зовн.іт={len(rws_c)}, x=({xf_c[0]:.4f}; {xf_c[1]:.4f}), "
              f"||x||={np.linalg.norm(xf_c):.4f}, g3={g3_circle(xf_c):.6f}, F={F(xf_c):.4f}, N_Phi={rws_c[-1]['total_n_Phi']}")

   
    #ВНУТРІШНІЙ МЕТОД ПРИ r = 100: ФЛЕТЧЕР-РІВС ПРОТИ АНИГРАДІЄНТА (ЛР 8)
 
    print("\n=== 7. ВНУТРІШНІЙ МЕТОД ПРИ r = 100 (СТАРТ x0 = (-5; 0)) ===")
    Phi_100, gradPhi_100 = make_external_penalty(F, gradF, cons_base, (), r=100.0)
    res_fr_100 = fletcher_reeves(Phi_100, gradPhi_100, x0_base, eps_grad=0.01)
    res_sd_100 = steepest_descent_line_search(Phi_100, gradPhi_100, x0_base, eps_grad=0.01)
    print(f"Флетчер-Рівс:   ітерацій={res_fr_100['iters']:4d}, N_Phi={res_fr_100['n_f']:5d}, N_gPhi={res_fr_100['n_g']:5d}, "
          f"x=({res_fr_100['x'][0]:.4f}; {res_fr_100['x'][1]:.4f}), g1={g1(res_fr_100['x']):.6f}")
    print(f"Антиградієнт:   ітерацій={res_sd_100['iters']:4d}, N_Phi={res_sd_100['n_f']:5d}, N_gPhi={res_sd_100['n_g']:5d}, "
          f"x=({res_sd_100['x'][0]:.4f}; {res_sd_100['x'][1]:.4f}), g1={g1(res_sd_100['x']):.6f}")

    
    #ЗВЕДЕНА ТАБЛИЦЯ ВСІХ ПРОГОНІВ
    print("\n=== 8. ЗВЕДЕНА ТАБЛИЦЯ ВСІХ ПРОГОНІВ ===")
    summary_items = [
        ("Зовнішній штраф (c=2)", c_runs[2.0][1]),
        ("Зовнішній штраф (c=4)", c_runs[4.0][1]),
        ("Зовнішній штраф (c=10, база)", c_runs[10.0][1]),
        ("Зовнішній штраф (c=100)", c_runs[100.0][1]),
        ("Рівність h(x)=0 (c=10)", rws_eq),
        ("Лог-бар'єр (старт (5;0))", rws_bar),
        ("Два обмеження (g1, g2)", rws_two),
        ("Коло R=4 (старт (-5;0))", circle_runs[0][2]),
        ("Коло R=4 (старт (5;0))", circle_runs[1][2]),
    ]

    print(f"{'Прогін':<28} | {'зовн.іт':>8} | {'N_Phi':>6} | {'N_gPhi':>6} | {'відповідь':>19} | {'поруш.':>10} | {'F':>9} | {'оцінка lam':>18}")
    print("-" * 124)
    for name, rws in summary_items:
        last = rws[-1]
        ans_s = f"({last['x'][0]:.4f};{last['x'][1]:.4f})"
        if "lam" in last:
            lam_s = f"{last['lam']:.3f}"
        elif len(last["lam_ineq"]) == 2:
            lam_s = f"({last['lam_ineq'][0]:.2f};{last['lam_ineq'][1]:.2f})"
        elif len(last["lam_ineq"]) == 1:
            lam_s = f"{last['lam_ineq'][0]:.3f}"
        else:
            lam_s = f"{last['lam_eq'][0]:.3f}"
        viol_val = last["g"] if "g" in last else last["viol"]
        print(f"{name:<28} | {len(rws):>8} | {last['total_n_Phi']:>6} | {last['total_n_gradPhi']:>6} | {ans_s:>19} | {viol_val:>10.6f} | {last['F']:>9.4f} | {lam_s:>18}")

    
    # ГРАФІКИ ДЛЯ ЗВІТУ (БАР'ЄР / ЗОВНІШНІЙ ШТРАФ ТА КРИВА МЕЖА "КОЛО")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Лівий графік: лінії рівня F, обмеження g1 та траєкторії зовнішнього штрафу і бар'єра
    xs = np.linspace(-6, 7, 250)
    ys = np.linspace(-6, 5, 250)
    X1, X2 = np.meshgrid(xs, ys)
    Z = 4.0*X1**2 + 2.0*X2**2 + 2.0*X1*X2 + 2.0*X1 - 10.0*X2

    cs = ax1.contour(X1, X2, Z, levels=20, colors="steelblue", alpha=0.55, linewidths=0.9)
    ax1.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

    feas1 = (2.0*X1 - X2) >= 6.0
    ax1.contourf(X1, X2, feas1.astype(int), levels=[0.5, 1.5], colors=["#b7efb0"], alpha=0.35)
    lx1 = np.linspace(-6, 7, 200)
    ax1.plot(lx1, 2.0*lx1 - 6.0, "g-", linewidth=2.0, label="2x1 - x2 = 6")

    traj_ext = np.vstack([x0_base] + [r["x"] for r in rws_ineq])
    traj_bar = np.vstack([x_int0] + [r["x"] for r in rws_bar])
    ax1.plot(traj_ext[:, 0], traj_ext[:, 1], "o--", color="navy", linewidth=1.7, markersize=4, label="Зовнішній штраф x(r)")
    ax1.plot(traj_bar[:, 0], traj_bar[:, 1], "s-", color="purple", linewidth=1.7, markersize=4, label="Лог-бар'єр x(mu)")
    ax1.scatter([x_star_1[0]], [x_star_1[1]], color="darkgreen", s=65, zorder=6, label="x* (одне обмеження)")

    ax1.set_xlim(-6, 7)
    ax1.set_ylim(-6, 5)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Зовнішній штраф vs Лог-бар'єр")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", fontsize=7.5)

    # Правий графік: крива межа x1^2 + x2^2 >= 16 та траєкторії з двох стартів
    cs2 = ax2.contour(X1, X2, Z, levels=20, colors="steelblue", alpha=0.55, linewidths=0.9)
    ax2.clabel(cs2, inline=True, fontsize=7, fmt="%.0f")

    feas_c = (X1**2 + X2**2) >= R_CIRCLE**2
    ax2.contourf(X1, X2, feas_c.astype(int), levels=[0.5, 1.5], colors=["#b7efb0"], alpha=0.35)
    theta = np.linspace(0, 2*np.pi, 300)
    ax2.plot(R_CIRCLE*np.cos(theta), R_CIRCLE*np.sin(theta), "g-", linewidth=2.0, label="Коло x1^2 + x2^2 = 16")

    colors_c = ["crimson", "darkorange"]
    for idx, (st, xf_c, rws_c) in enumerate(circle_runs):
        tr_c = np.vstack([st] + [r["x"] for r in rws_c])
        ax2.plot(tr_c[:, 0], tr_c[:, 1], "o--", color=colors_c[idx], linewidth=1.7, markersize=4,
                 label=f"Старт ({st[0]:.0f};{st[1]:.0f}) -> F={F(xf_c):.2f}")

    ax2.set_xlim(-6, 7)
    ax2.set_ylim(-6, 5)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_title("Нелінійна межа (поза колом R=4) з двох стартів")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=7.5)

    plt.tight_layout()
    plt.show()
