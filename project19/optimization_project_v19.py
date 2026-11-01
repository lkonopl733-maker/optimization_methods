import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ПРОЄКТНА РОБОТА: ВІДНОВЛЕННЯ ЦИФРОВОГО ЗОБРАЖЕННЯ
# ВАРІАНТ 19

VARIANT = 19
SEED = 255190
SIGMA_NOISE = 0.1020

# Лабораторна робота 1
X0_1D = 0.3648
DELTA0_1D = 0.2195

# Лабораторна робота 2
EPS_1D = 1e-4

# Лабораторні роботи 3–4
LAMBDA0 = 1.2912
SIMPLEX_STEP = 0.3924

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TABLES_DIR = OUTPUT_DIR / "tables"
TABLES_DIR.mkdir(exist_ok=True)


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "так" if v else "ні"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e5:
            return f"{v:.4e}"
        return f"{v:.6f}"
    return str(v)


def print_table(title, header, rows, filename=None, max_rows=None, show=True):
    """Друк таблиці в консоль (з можливим скороченням) і збереження повної в CSV."""
    import csv
    rows = [list(r) for r in rows]
    if filename:
        with open(TABLES_DIR / f"{filename}.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(header)
            for row in rows:
                writer.writerow([
                    repr(float(c)) if isinstance(c, (float, np.floating)) else _fmt(c)
                    for c in row
                ])
    if not show:
        return
    cells = [[_fmt(c) for c in row] for row in rows]
    if max_rows is not None and len(cells) > max_rows:
        cut = max(max_rows - 4, 1)
        cells = cells[:cut] + [["…"] * len(header)] + cells[-3:]
    widths = [
        max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h))
        for i, h in enumerate(header)
    ]
    print(f"\nТаблиця: {title}")
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if max_rows is not None and len(rows) > max_rows:
        print(f"(у консолі скорочено; повна таблиця — {len(rows)} рядків, файл "
              f"tables/{filename}.csv)")


def check_value(got, ref, tol=None):
    if ref is None:
        return "—"
    if tol is None:
        return "✓" if got == ref else "✗"
    return "✓" if abs(got - ref) <= tol else "✗"


def conclusion_table(title, items, filename):
    """items: (показник, отримано, контрольне, допуск, примітка)."""
    rows = []
    for name, got, ref, tol, note in items:
        rows.append([name, got, "—" if ref is None else ref])
    print_table(title,
                ["Показник", "Отримано", "Контрольне (PDF, варіант 19)"],
                rows, filename)


assert (VARIANT * 7919 + 104729) % (2**31 - 1) == SEED


# допоміжні функції

def mse(a, b):
    """Середньоквадратична помилка."""
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    """Пікове співвідношення сигнал/шум."""
    error = mse(a, b)

    if error == 0:
        return float("inf")

    return 10.0 * math.log10(1.0 / error)


def save_image(image, name):
    """Збереження зображення у PNG та PGM."""
    image = np.clip(image, 0.0, 1.0)
    image_u8 = np.round(image * 255).astype(np.uint8)

    plt.imsave(
        OUTPUT_DIR / f"{name}.png",
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    height, width = image_u8.shape

    with open(OUTPUT_DIR / f"{name}.pgm", "wb") as file:
        file.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        file.write(image_u8.tobytes())


def print_image_center(image, title):
    """Виведення центральної матриці 5x5."""
    h, w = image.shape
    center_y = h // 2
    center_x = w // 2

    part = image[
        center_y - 2:center_y + 3,
        center_x - 2:center_x + 3
    ]

    print(f"\n{title} — центральна матриця 5x5:")
    print(np.array2string(part, precision=5, suppress_small=True))


# ЛАБОРАТОРНА РОБОТА 0
# ГЕНЕРАТОР ПСЕВДОВИПАДКОВИХ ЧИСЕЛ

class Rng:
    """64-бітний лінійний конгруентний генератор."""

    MASK64 = (1 << 64) - 1
    MULT = 6364136223846793005
    INCR = 1442695040888963407

    def __init__(self, seed):
        self.state = seed & self.MASK64

    def next_u64(self):
        self.state = (
            self.state * self.MULT + self.INCR
        ) & self.MASK64

        return self.state

    def uniform(self):
        return (self.next_u64() >> 11) / 9007199254740992.0

    def normal(self):
        u1 = max(self.uniform(), 1e-300)
        u2 = self.uniform()

        return (
            math.sqrt(-2.0 * math.log(u1))
            * math.cos(2.0 * math.pi * u2)
        )


def clean_image(size=128):
    """Створення еталонного зображення."""

    img = np.zeros((size, size), dtype=float)

    cell = max(3, size // 12)

    for i in range(size):
        for j in range(size):
            u = j / (size - 1)
            v = i / (size - 1)

            # Фоновий градієнт
            val = 0.25 + 0.35 * (0.6 * u + 0.4 * v)

            # Кругова структура
            val += 0.08 * math.cos(
                2.0 * math.pi * math.sqrt(
                    (u - 0.5) ** 2 + (v - 0.5) ** 2
                )
            )

            # Світле коло
            if (
                (u - 0.32) ** 2 + (v - 0.34) ** 2
                <= 0.16 ** 2
            ):
                val = 0.92

            # Темний квадрат
            if abs(u - 0.70) <= 0.15 and abs(v - 0.30) <= 0.15:
                val = 0.10

            # Трикутник
            if (
                0.60 < v < 0.90
                and abs(u - 0.30) < 0.9 * (v - 0.60)
            ):
                val = 0.68

            # Шахова структура
            if 0.58 < u < 0.92 and 0.60 < v < 0.90:
                if ((j // cell) + (i // cell)) % 2 == 0:
                    val = 0.85
                else:
                    val = 0.20

            img[i, j] = np.clip(val, 0.0, 1.0)

    # Горизонтальні смуги
    for idx in range(4):
        thickness = max(2, (4 - idx) * max(1, size // 64))
        row = int(0.04 * size) + idx * max(3, size // 22)

        if row + thickness < size:
            img[
                row:row + thickness,
                int(0.05 * size):int(0.95 * size)
            ] = 0.98

    return img


def add_noise(clean, sigma, seed):
    """Додавання гаусівського шуму."""

    rng = Rng(seed)
    raw = np.empty_like(clean)

    for i in range(clean.shape[0]):
        for j in range(clean.shape[1]):
            raw[i, j] = clean[i, j] + sigma * rng.normal()

    # raw не обмежене діапазоном [0, 1].
    # noisy — зображення, обмежене діапазоном [0, 1].
    noisy = np.clip(raw, 0.0, 1.0)

    return raw, noisy


def gaussian_blur(img, sigma):
    """Сепарабельне гаусівське розмиття."""

    if sigma <= 1e-6:
        return img.copy()

    radius = int(4.0 * sigma + 0.5)
    t = np.arange(-radius, radius + 1, dtype=float)

    kernel = np.exp(-(t * t) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    height, width = img.shape
    temp = np.zeros_like(img)
    result = np.zeros_like(img)

    # Горизонтальний прохід
    for z in range(-radius, radius + 1):
        indices = np.clip(
            np.arange(width) + z, 0, width - 1
        )

        temp += kernel[z + radius] * img[:, indices]

    # Вертикальний прохід
    for z in range(-radius, radius + 1):
        indices = np.clip(
            np.arange(height) + z, 0, height - 1
        )

        result += kernel[z + radius] * temp[indices, :]

    return result


clean = clean_image(128)
raw, noisy = add_noise(clean, SIGMA_NOISE, SEED)

save_image(clean, "00_clean")
save_image(raw, "00_noisy_raw")
save_image(noisy, "00_noisy")

print("=" * 70)
print("ЛАБОРАТОРНА РОБОТА 0 — СТВОРЕННЯ ЗОБРАЖЕННЯ")
print("=" * 70)
print(f"Варіант: {VARIANT}")
print(f"Seed: {SEED}")
print(f"Sigma шуму: {SIGMA_NOISE:.4f}")
print(f"MSE шумного зображення: {mse(noisy, clean):.8f}")
print(f"PSNR шумного зображення: {psnr(noisy, clean):.4f} дБ")

print_image_center(clean, "Еталонне зображення")
print_image_center(noisy, "Шумне зображення")

conclusion_table(
    "ЛР0 — Висновкова таблиця (вхідні дані)",
    [
        ("Зерно генератора", SEED, 255190, None, ""),
        ("MSE шумного зображення", mse(noisy, clean), 0.00939531, 5e-9, ""),
        ("PSNR шумного, дБ", psnr(noisy, clean), 20.2709, 5e-5, ""),
    ],
    "lab0_conclusion",
)

# ГРАФІК ЛР0 — еталонне та зашумлене зображення
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(clean, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Еталонне зображення")
axes[0].axis("off")
axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Зображення з шумом")
axes[1].axis("off")
fig.suptitle("Лабораторна робота 0")
fig.tight_layout()
plt.show()

# Облік кількості обчислень цільової функції

class Oracle1D:
    def __init__(self, fn):
        self.fn = fn
        self.n_f = 0

    def value(self, x):
        self.n_f += 1
        return self.fn(float(x))


class Oracle2D:
    def __init__(self, fn):
        self.fn = fn
        self.n_f = 0

    def value(self, x):
        self.n_f += 1
        return self.fn(np.asarray(x, dtype=float))


def objective_sigma(sigma):
    """MSE після гаусівської фільтрації."""
    sigma = max(float(sigma), 1e-6)
    restored = gaussian_blur(noisy, sigma)
    return mse(restored, clean)


# лабораторна 1 метод Свенна

def swann(oracle, x0, delta, max_iter=100, trace=None):
    """Знаходження інтервалу невизначеності методом Свенна.
    trace список, куди записуються пробні точки (k, x, f)."""

    def log(k, x, f):
        if trace is not None:
            trace.append((k, x, f))

    x_left = x0 - delta
    x_right = x0 + delta
    f_left = oracle.value(x_left)
    f0 = oracle.value(x0)
    f_right = oracle.value(x_right)
    log(0, x_left, f_left)
    log(0, x0, f0)
    log(0, x_right, f_right)

    if f_right < f0:
        direction = 1.0
        x = x_right
        f_current = f_right
        x_previous = x0
    elif f_left < f0:
        direction = -1.0
        x = x_left
        f_current = f_left
        x_previous = x0
    else:
        return (x_left, x_right), 0

    step = delta

    for iteration in range(1, max_iter + 1):
        step *= 2.0
        x_next = x + direction * step
        f_next = oracle.value(x_next)
        log(iteration, x_next, f_next)

        if f_next >= f_current:
            return tuple(sorted((x_previous, x_next))), iteration

        x_previous = x
        x = x_next
        f_current = f_next

    raise RuntimeError("Метод Свенна: перевищено ліміт ітерацій.")


oracle_sw = Oracle1D(objective_sigma)

swann_trace = []
interval, swann_iters = swann(
    oracle_sw,
    X0_1D,
    DELTA0_1D,
    trace=swann_trace
)

print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА 1 — МЕТОД СВЕННА")
print("=" * 70)
print(f"x0 = {X0_1D}")
print(f"delta0 = {DELTA0_1D}")
print(f"Інтервал: [{interval[0]:.6f}; {interval[1]:.6f}]")
print(f"Ітерацій: {swann_iters}")
print(f"Обчислень функції: {oracle_sw.n_f}")

print_table(
    "ЛР1 — Таблиця ітерацій методу Свенна (k = 0: початкові три точки)",
    ["k", "x_k", "f(x_k) = MSE"],
    swann_trace,
    "lab1_swann_iterations",
)
print_table(
    "ЛР1 — Підсумок методу Свенна",
    ["x0", "Δ0", "a0", "b0", "Ітерацій", "N_f"],
    [(X0_1D, DELTA0_1D, interval[0], interval[1], swann_iters, oracle_sw.n_f)],
    "lab1_summary",
)
conclusion_table(
    "ЛР1 — Висновкова таблиця",
    [
        ("Ліва межа a0", interval[0], 0.3648, 5e-5, ""),
        ("Права межа b0", interval[1], 1.0233, 5e-5, ""),
        ("N_f Свенна", oracle_sw.n_f, 4, None, ""),
    ],
    "lab1_conclusion",
)

# ГРАФІК ЛР1 — цільова функція та інтервал Свенна
fig, ax = plt.subplots(figsize=(8, 5))
sigmas_plot = np.linspace(0.1, 2.2, 100)
vals_plot = [objective_sigma(s) for s in sigmas_plot]
ax.plot(sigmas_plot, vals_plot, label="f(sigma)", color="blue")

#пробні точки Свенна (якщо зберегти їх історію)
ax.axvspan(interval[0], interval[1], color="green", alpha=0.2, label=f"Інтервал [{interval[0]:.3f}, {interval[1]:.3f}]")
ax.set_title(f"Лабораторна робота 1 — метод Свенна (Варіант {VARIANT})")
ax.set_xlabel("σ — масштаб гаусового ядра")
ax.set_ylabel("MSE(I_filtered, I_clean)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

#лабораторна робота 2 дихотомія і золотий переріз

def dichotomy(oracle, a, b, eps, max_iter=500):
    iteration = 0
    delta = eps / 4.0
    history = [(a, b)]  # Зберігаємо початковий інтервал
    while b - a > eps and iteration < max_iter:
        iteration += 1
        middle = (a + b) / 2.0
        x1 = middle - delta
        x2 = middle + delta
        f1 = oracle.value(x1)
        f2 = oracle.value(x2)
        if f1 <= f2:
            b = x2
        else:
            a = x1
        history.append((a, b))
    x_star = (a + b) / 2.0
    oracle.value(x_star)  # контрольне обчислення f(x*)
    return x_star, iteration, history


def golden(oracle, a, b, eps, max_iter=500):
    tau = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1 = oracle.value(x1)
    f2 = oracle.value(x2)
    iteration = 0
    history = [(a, b)]  # Зберігаємо початковий інтервал
    while b - a > eps and iteration < max_iter:
        iteration += 1
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - tau * (b - a)
            f1 = oracle.value(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + tau * (b - a)
            f2 = oracle.value(x2)
        history.append((a, b))
    x_star = (a + b) / 2.0
    oracle.value(x_star)  # контрольне обчислення f(x*)
    return x_star, iteration, history


oracle_d = Oracle1D(objective_sigma)
sigma_d, it_d, hist_d = dichotomy(
    oracle_d, interval[0], interval[1], EPS_1D
)

oracle_g = Oracle1D(objective_sigma)
sigma_g, it_g, hist_g = golden(
    oracle_g, interval[0], interval[1], EPS_1D
)

restored2 = gaussian_blur(noisy, sigma_g)
save_image(restored2, "02_gaussian")

print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА 2 — ОДНОВИМІРНА ОПТИМІЗАЦІЯ")
print("=" * 70)
print("Метод дихотомії:")
print(f"Sigma = {sigma_d:.6f}")
print(f"Ітерацій: {it_d}")
print(f"Обчислень функції: {oracle_d.n_f}")

print("\nМетод золотого перерізу:")
print(f"Sigma = {sigma_g:.6f}")
print(f"Ітерацій: {it_g}")
print(f"Обчислень функції: {oracle_g.n_f}")
print(f"PSNR після фільтра: {psnr(restored2, clean):.4f} дБ")

def interval_rows(hist):
    rows, prev = [], None
    for k, (lo, hi) in enumerate(hist):
        length = hi - lo
        mid = (lo + hi) / 2.0
        rows.append((k, lo, hi, length,
                     (length / prev) if prev else "—",
                     mid, objective_sigma(mid)))
        prev = length
    return rows


_hdr2 = ["k", "a_k", "b_k", "L_k = b_k − a_k", "L_k / L_(k−1)",
         "x_k (середина)", "f(x_k)"]
print_table("ЛР2 — Таблиця ітерацій: дихотомія", _hdr2,
            interval_rows(hist_d), "lab2_dichotomy_iterations")
print_table("ЛР2 — Таблиця ітерацій: золотий переріз", _hdr2,
            interval_rows(hist_g), "lab2_golden_iterations")

psnr_d = psnr(gaussian_blur(noisy, sigma_d), clean)
psnr_g = psnr(restored2, clean)
print_table(
    "ЛР2 — Порівняння методів",
    ["Метод", "σ*", "Ітерацій", "N_f", "Кінцева довжина відрізка", "PSNR, дБ"],
    [
        ("Дихотомія", sigma_d, it_d, oracle_d.n_f,
         hist_d[-1][1] - hist_d[-1][0], psnr_d),
        ("Золотий переріз", sigma_g, it_g, oracle_g.n_f,
         hist_g[-1][1] - hist_g[-1][0], psnr_g),
    ],
    "lab2_comparison",
)
conclusion_table(
    "ЛР2 — Висновкова таблиця",
    [
        ("σ* дихотомії", sigma_d, 0.577802, 5e-6, "збіг до 5-го знака"),
        ("Ітерацій дихотомії", it_d, 14, None, ""),
        ("N_f дихотомії", oracle_d.n_f, 29, None, ""),
        ("σ* золотого перерізу", sigma_g, 0.577802, 5e-6, ""),
        ("Ітерацій золотого перерізу", it_g, 19, None, ""),
        ("N_f золотого перерізу", oracle_g.n_f, 22, None, ""),
        ("Обидва методи дають однаковий σ*", abs(sigma_d - sigma_g) < 5e-5,
         True, None, ""),
        ("Золотий переріз дешевший за N_f", oracle_g.n_f < oracle_d.n_f,
         True, None, "хоч ітерацій більше"),
        ("PSNR після лабораторної 2, дБ", psnr_g, 23.4467, 5e-5, ""),
    ],
    "lab2_conclusion",
)

# ГРАФІК ЛР2 — порівняння результатів дихотомії та золотого перерізу
fig_l2, ax_l2 = plt.subplots(figsize=(8, 5))
lens_d = [h[1] - h[0] for h in hist_d if isinstance(h, (list, tuple, np.ndarray)) and len(h) == 2]
lens_g = [h[1] - h[0] for h in hist_g if isinstance(h, (list, tuple, np.ndarray)) and len(h) == 2]
ax_l2.plot(range(len(lens_d)), lens_d, label=f"Дихотомія (N_f = {oracle_d.n_f})", linewidth=2)
ax_l2.plot(range(len(lens_g)), lens_g, label=f"Золотий переріз (N_f = {oracle_g.n_f})", linestyle="--", color="red", linewidth=2)
ax_l2.set_yscale("log")
ax_l2.set_xlabel("Ітерація k")
ax_l2.set_ylabel("Довжина L_k")
ax_l2.set_title("Скорочення інтервалу невизначеності")
ax_l2.legend()
ax_l2.grid(True, alpha=0.3)
fig_l2.tight_layout()
plt.show()


# лабораторна робота 3 білатеральний фільтр і методи Нелдера-Міда та Пауелла

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])


def bilateral_filter(img, sigma_spatial, sigma_range):
    ss = max(float(sigma_spatial), 1e-3)
    sr = max(float(sigma_range), 1e-3)
    radius = int(min(max(round(2.0 * ss), 1), 5))
    padded = np.pad(img, radius, mode="reflect")
    numerator = np.zeros_like(img)
    denominator = np.zeros_like(img)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[
                radius + dy : radius + dy + img.shape[0],
                radius + dx : radius + dx + img.shape[1],
            ]
            spatial_weight = math.exp(-(dx * dx + dy * dy) / (2.0 * ss * ss))
            range_weight = np.exp(-((shifted - img) ** 2) / (2.0 * sr * sr))
            weight = spatial_weight * range_weight
            numerator += weight * shifted
            denominator += weight
    return numerator / np.maximum(denominator, 1e-15)


def objective2(theta):
    theta = np.asarray(theta, dtype=float)
    clamped = np.clip(theta, LO, HI)
    gap = theta - clamped
    filtered = bilateral_filter(noisy, clamped[0], clamped[1])
    return mse(filtered, clean) + 1000.0 * float(gap @ gap)


def nelder_mead(oracle, x0, step, tol=5e-8, patience=3, max_iter=200):
    """
    Нелдер—Мід у 2D (симплекс = трикутник).
    Зупинка: розмір симплекса <= tol протягом `patience` ітерацій поспіль.
    (patience=3 -> 61 ітерація / N_f=122 (пдф)
     patience=2 -> 60 ітерацій / N_f=120 (сайт)
    """
    x0 = np.asarray(x0, dtype=float)

    simplex = [
        np.clip(x0.copy(), LO, HI),
        np.clip(x0 + np.array([step, 0.0]), LO, HI),
        np.clip(x0 + np.array([0.0, 0.4 * step]), LO, HI),
    ]
    values = [oracle.value(p) for p in simplex]

    order = np.argsort(values)
    simplex = [simplex[i] for i in order]
    values = [values[i] for i in order]
    history = [simplex[0].copy()]
    small_in_row = 0

    for iteration in range(1, max_iter + 1):
        best, mid, worst = simplex
        f_best, f_mid, f_worst = values
        centroid = (best + mid) / 2.0

        reflected = np.clip(centroid + (centroid - worst), LO, HI)
        f_r = oracle.value(reflected)

        if f_r < f_best:
            expanded = np.clip(centroid + 2.0 * (reflected - centroid), LO, HI)
            f_e = oracle.value(expanded)
            if f_e < f_r:
                simplex[2], values[2] = expanded, f_e
            else:
                simplex[2], values[2] = reflected, f_r
        elif f_r < f_mid:
            simplex[2], values[2] = reflected, f_r
        else:
            if f_r < f_worst:
                contracted = np.clip(centroid + 0.5 * (reflected - centroid), LO, HI)
            else:
                contracted = np.clip(centroid + 0.5 * (worst - centroid), LO, HI)
            f_c = oracle.value(contracted)
            if f_c <= min(f_r, f_worst):
                simplex[2], values[2] = contracted, f_c
            else:
                for i in (1, 2):
                    simplex[i] = np.clip(best + 0.5 * (simplex[i] - best), LO, HI)
                    values[i] = oracle.value(simplex[i])

        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        history.append(simplex[0].copy())

        size = max(np.linalg.norm(p - simplex[0]) for p in simplex)
        small_in_row = small_in_row + 1 if size <= tol else 0
        if small_in_row >= patience:
            break

    return simplex[0], values[0], iteration, history



def line_minimize(oracle, x, direction, tol=1e-5):
    direction = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(direction)
    if nrm < 1e-15:
        return x.copy()
    direction = direction / nrm

    a, b = -1.5, 1.5
    tau = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - tau * (b - a)
    d = a + tau * (b - a)

    fc = oracle.value(np.clip(x + c * direction, LO, HI))
    fd = oracle.value(np.clip(x + d * direction, LO, HI))

    for _ in range(28):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - tau * (b - a)
            fc = oracle.value(np.clip(x + c * direction, LO, HI))
        else:
            a, c, fc = c, d, fd
            d = a + tau * (b - a)
            fd = oracle.value(np.clip(x + d * direction, LO, HI))

    return np.clip(x + ((a + b) / 2.0) * direction, LO, HI)


def powell(oracle, x0, tol=1e-5, max_cycles=40):
    x = np.clip(np.asarray(x0, dtype=float), LO, HI)
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    history = [x.copy()]
    oracle.value(x)                      # f(x0) на початку

    for cycle in range(1, max_cycles + 1):
        start = x.copy()
        for d in directions:
            x = line_minimize(oracle, x, d)
            history.append(x.copy())

        disp = x - start
        if np.linalg.norm(disp) < tol:
            oracle.value(x)              # f наприкінці перевірочного циклу
            break

        x = line_minimize(oracle, x, disp)
        history.append(x.copy())
        oracle.value(x)                  # f наприкінці циклу

        nrm = np.linalg.norm(disp)
        if nrm > 1e-15:
            directions = [directions[1], disp / nrm]

    f_final = oracle.value(x)            # підсумкове f(x*)
    # +1: перевірочний цикл, у якому зсув < tol, рахується окремо
    return x, f_final, cycle + 1, history


# запуск
sigma_s0 = 1.1556
sigma_r0 = 0.1602

oracle_nm = Oracle2D(objective2)
theta_nm, f_nm, it_nm, hist_nm = nelder_mead(
    oracle_nm, [sigma_s0, sigma_r0], SIMPLEX_STEP
)

oracle_pw = Oracle2D(objective2)
theta_pw, f_pw, cycles_pw, hist_pw = powell(
    oracle_pw, [sigma_s0, sigma_r0]
)

restored3_nm = bilateral_filter(noisy, theta_nm[0], theta_nm[1])
restored3_pw = bilateral_filter(noisy, theta_pw[0], theta_pw[1])

save_image(restored3_nm, "03_bilateral_nelder_mead")
save_image(restored3_pw, "03_bilateral_powell")

print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА 3 — БІЛАТЕРАЛЬНИЙ ФІЛЬТР")
print("=" * 70)
print(f"Нелдер—Мід: Sigma_s={theta_nm[0]:.6f}, Sigma_r={theta_nm[1]:.6f}, "
      f"N_f={oracle_nm.n_f}, ітер={it_nm}, PSNR={psnr(restored3_nm, clean):.4f} дБ")
print(f"Пауелл:     Sigma_s={theta_pw[0]:.6f}, Sigma_r={theta_pw[1]:.6f}, "
      f"N_f={oracle_pw.n_f}, циклів={cycles_pw}, PSNR={psnr(restored3_pw, clean):.4f} дБ")

# таблиці лр 3
print_table(
    "ЛР3 — Нелдер—Мід: найкраща вершина симплекса після кожної ітерації",
    ["k", "σ_s", "σ_r", "f(θ_k)"],
    [(k, p[0], p[1], objective2(p)) for k, p in enumerate(hist_nm)],
    "lab3_nelder_mead_iterations", max_rows=20,
)
print_table(
    "ЛР3 — Пауелл: точка після кожної одновимірної мінімізації "
    "(0 — початкова)",
    ["Крок", "σ_s", "σ_r", "f(θ)"],
    [(k, p[0], p[1], objective2(p)) for k, p in enumerate(hist_pw)],
    "lab3_powell_iterations", max_rows=20,
)
print_table(
    "ЛР3 — Порівняння методів",
    ["Метод", "σ_s*", "σ_r*", "f(θ*)", "Ітерацій / циклів", "N_f", "PSNR, дБ"],
    [
        ("Нелдер—Мід", theta_nm[0], theta_nm[1], f_nm, it_nm,
         oracle_nm.n_f, psnr(restored3_nm, clean)),
        ("Пауелл", theta_pw[0], theta_pw[1], f_pw, cycles_pw,
         oracle_pw.n_f, psnr(restored3_pw, clean)),
    ],
    "lab3_comparison",
)
conclusion_table(
    "ЛР3 — Висновкова таблиця",
    [
        ("σ_s* (Нелдер—Мід)", theta_nm[0], 2.1934, 5e-5, ""),
        ("σ_r* (Нелдер—Мід)", theta_nm[1], 0.2117, 5e-5, ""),
        ("Ітерацій Нелдера—Міда", it_nm, 61, None, "сайт показує 60 / N_f=120"),
        ("N_f Нелдера—Міда", oracle_nm.n_f, 122, None, ""),
        ("Циклів Пауелла", cycles_pw, 6, None, "з перевірочним циклом"),
        ("N_f Пауелла", oracle_pw.n_f, 413, None, ""),
        ("PSNR, дБ", psnr(restored3_nm, clean), 28.4042, 5e-5, ""),
        ("Обидва методи приходять в одну точку",
         float(np.max(np.abs(theta_nm - theta_pw))) < 1e-4, True, None, ""),
        ("Пауелл дорожчий за N_f", oracle_pw.n_f > oracle_nm.n_f, True, None,
         "у варіанті 19 так"),
    ],
    "lab3_conclusion",
)

#графіки
fig_l3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

s_vals = np.linspace(0.3, 5.0, 55)
r_vals = np.linspace(0.01, 1.0, 55)
S, R = np.meshgrid(s_vals, r_vals)
Z = np.zeros_like(S)
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        Z[i, j] = objective2([S[i, j], R[i, j]])

cp = ax1.contourf(S, R, Z, levels=28, cmap="viridis", alpha=0.88)
fig_l3.colorbar(cp, ax=ax1, label="f(θ)")

nm_x = [p[0] for p in hist_nm]
nm_y = [p[1] for p in hist_nm]
pw_x = [p[0] for p in hist_pw]
pw_y = [p[1] for p in hist_pw]

ax1.plot(nm_x, nm_y, "o-", color="red", markersize=3.5, linewidth=1.3, label="Нелдер—Мід")
ax1.plot(pw_x, pw_y, "o-", color="limegreen", markersize=3.5, linewidth=1.3, label="Пауелл")
ax1.plot(theta_nm[0], theta_nm[1], marker="*", color="lime", markersize=15, zorder=6)

ax1.set_xlabel("σ_s — просторовий масштаб")
ax1.set_ylabel("σ_r — фотометричний поріг")
ax1.set_title("Лінії рівня f(σ_s, σ_r) і траєкторії методів")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.3, 5.0)
ax1.set_ylim(0.01, 1.0)

vals_nm = [objective2(p) for p in hist_nm]
vals_pw = [objective2(p) for p in hist_pw]

ax2.plot(range(len(vals_nm)), vals_nm, color="C0", linewidth=2.0,
         label=f"Нелдер—Мід (N_f = {oracle_nm.n_f})")
ax2.plot(range(len(vals_pw)), vals_pw, color="red", linestyle="--", linewidth=2.0,
         label=f"Пауелл (N_f = {oracle_pw.n_f})")
ax2.set_yscale("log")
ax2.set_xlabel("Ітерація k")
ax2.set_ylabel("f(θ_k)")
ax2.set_title("Спадання цільової функції")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig_l3.tight_layout()
plt.show()
# лабораторна робота 4 градієнтні методи регуляризації

def grad_xy(u):
    """Дискретний градієнт."""

    gx = np.zeros_like(u)
    gy = np.zeros_like(u)

    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]

    return gx, gy


def divergence(px, py):
    """Дискретна дивергенція, узгоджена з grad_xy."""

    d = np.zeros_like(px)

    d[:, 0] += px[:, 0]
    d[:, 1:-1] += px[:, 1:-1] - px[:, :-2]
    d[:, -1] -= px[:, -2]

    d[0, :] += py[0, :]
    d[1:-1, :] += py[1:-1, :] - py[:-2, :]
    d[-1, :] -= py[-2, :]

    return d


def laplacian(u):
    """Дискретний оператор Лапласа."""
    return divergence(*grad_xy(u))


def energy(image, target, lam):
    """Енергетична функція Тіхонова."""

    gx, gy = grad_xy(image)

    data_term = 0.5 * float(np.sum((image - target) ** 2))
    regularization_term = 0.5 * lam * float(
        np.sum(gx ** 2 + gy ** 2)
    )

    return data_term + regularization_term


def energy_grad(image, target, lam):
    """Градієнт енергетичної функції."""

    return (image - target) - lam * laplacian(image)


def gradient_method(
    target,
    lam,
    kind="gd",
    beta=None,
    max_iter=200,
    tol=1e-6,
    start=None,
    ref=None
):
    """
    Градієнтні методи:
    gd          — звичайний градієнтний спуск;
    armijo      — спуск із правилом Арміхо;
    hb          — метод важкої кульки;
    nag         — метод Нестерова;
    nag_strong  — Нестеров для сильно опуклої функції.
    """

    L = 1.0 + 8.0 * lam
    h = 1.0 / L

    image = (target.copy() if start is None else start.copy())
    previous = image.copy()
    rows = []
    converged = False

    for k in range(1, max_iter + 1):

        if kind in ("nag", "nag_strong"):
            if kind == "nag":
                momentum = (
                    (k - 1.0) / (k + 2.0)
                    if beta is None else beta
                )
            else:
                momentum = (
                    (math.sqrt(L) - 1.0)
                    / (math.sqrt(L) + 1.0)
                    if beta is None else beta
                )

            y = image + momentum * (image - previous)
            g = energy_grad(y, target, lam)
            new_image = y - h * g

        else:
            g = energy_grad(image, target, lam)

            if kind == "hb":
                momentum = 0.9 if beta is None else beta
                new_image = (
                    image - h * g
                    + momentum * (image - previous)
                )

            elif kind == "armijo":
                f0 = energy(image, target, lam)
                step = 2.0          # початковий пробний крок (не h=1/L)
                g2 = float(np.sum(g * g))

                for _ in range(50):
                    candidate = image - step * g
                    candidate_energy = energy(
                        candidate, target, lam
                    )

                    if candidate_energy <= f0 - 1e-4 * step * g2:
                        break

                    step *= 0.6         # коефіцієнт зменшення кроку

                new_image = candidate

            else:
                new_image = image - h * g

        gradient_norm = float(np.linalg.norm(g))
        current_energy = energy(image, target, lam)

        dist = (float(np.linalg.norm(image - ref))
                if ref is not None else float("nan"))
        rows.append((k, current_energy, gradient_norm, dist))

        previous, image = image, new_image

        if gradient_norm <= tol:
            converged = True
            break

    # рядків = кількість обчислень градієнта; ітерацій = кроків, що зроблено
    n_iter = len(rows) - 1 if converged else len(rows)
    return image, n_iter, rows



# Перевірка 1: спряженість дискретних операторів

rng_check = np.random.default_rng(0)

u = rng_check.normal(size=(9, 7))
px = rng_check.normal(size=u.shape)
py = rng_check.normal(size=u.shape)

gx, gy = grad_xy(u)

left = float(np.sum(gx * px + gy * py))
right = -float(np.sum(u * divergence(px, py)))

adjoint_error = abs(left - right)


# Перевірка 2: градієнт скінченними різницями

rng_fd = np.random.default_rng(1)

test_image = rng_fd.random((16, 16))
test_target = rng_fd.random((16, 16))
direction = rng_fd.normal(size=(16, 16))

test_lambda = 0.4
t = 1e-6

numeric_grad = (
    energy(
        test_image + t * direction,
        test_target,
        test_lambda
    )
    -
    energy(
        test_image - t * direction,
        test_target,
        test_lambda
    )
) / (2.0 * t)

analytic_grad = float(
    np.sum(
        energy_grad(test_image, test_target, test_lambda)
        * direction
    )
)

finite_difference_error = abs(numeric_grad - analytic_grad)



# Пошук оптимального lambda*

def objective_lambda(lam):
    lam = max(float(lam), 1e-6)

    restored, _, _ = gradient_method(
        noisy,
        lam,
        kind="gd",
        max_iter=17,
        tol=1e-4
    )

    return mse(restored, clean)


oracle_lam = Oracle1D(objective_lambda)

lambda_star, it_lam, hist_lam = golden(  
    oracle_lam,
    0.001,
    2.0,
    EPS_1D
)

lam = float(lambda_star)

L_lip = 1.0 + 8.0 * lam
mu_strong = 1.0

# Оптимальний коефіцієнт для важкої кульки
beta_hb_opt = (
    (math.sqrt(L_lip) - math.sqrt(mu_strong))
    / (math.sqrt(L_lip) + math.sqrt(mu_strong))
) ** 2

# Коефіцієнт Нестерова для сильно опуклої функції
beta_nag_strong = (
    (math.sqrt(L_lip) - math.sqrt(mu_strong))
    / (math.sqrt(L_lip) + math.sqrt(mu_strong))
)


# Точний розв'язок I* та E* (для графіків E(I_k) - E*, ||I_k - I*||)
I_star, _, _ = gradient_method(
    noisy, lam, kind="nag_strong",
    beta=beta_nag_strong, max_iter=5000, tol=1e-13
)
E_star = energy(I_star, noisy, lam)

restored4_gd, n_gd, history_gd = gradient_method(
    noisy, lam, kind="gd", ref=I_star
)

restored4_armijo, n_armijo, history_armijo = gradient_method(
    noisy, lam, kind="armijo", ref=I_star
)

restored4_hb, n_hb, history_hb = gradient_method(
    noisy, lam, kind="hb", beta=0.9, ref=I_star
)

restored4_hb_opt, n_hb_opt, history_hb_opt = gradient_method(
    noisy, lam, kind="hb", beta=beta_hb_opt, ref=I_star
)

restored4_nag, n_nag, history_nag = gradient_method(
    noisy, lam, kind="nag", ref=I_star
)

restored4_nag_strong, n_nag_strong, history_nag_strong = (
    gradient_method(
        noisy, lam, kind="nag_strong",
        beta=beta_nag_strong, ref=I_star
    )
)

save_image(restored4_gd, "04_gradient")
save_image(restored4_armijo, "04_armijo")
save_image(restored4_hb, "04_heavy_ball")
save_image(restored4_hb_opt, "04_heavy_ball_opt")
save_image(restored4_nag, "04_nesterov")
save_image(restored4_nag_strong, "04_nesterov_strong")


print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА 4 — ГРАДІЄНТНІ МЕТОДИ")
print("=" * 70)

print(f"Початкове lambda0 = {LAMBDA0}")
print(f"Знайдене lambda* = {lam:.6f}")
print(f"L = 1 + 8*lambda = {L_lip:.6f}")
print(f"Оптимальний beta для важкої кульки = {beta_hb_opt:.6f}")
print(f"Beta Нестерова для сильно опуклої задачі = {beta_nag_strong:.6f}")
print(f"Похибка спряженості операторів = {adjoint_error:.3e}")
print(f"Похибка перевірки градієнта = {finite_difference_error:.3e}")
print(f"Обчислень функції для пошуку lambda = {oracle_lam.n_f}")

gradient_results = [
    ("Звичайний градієнтний спуск", restored4_gd, n_gd),
    ("Метод Арміхо", restored4_armijo, n_armijo),
    ("Важка кулька beta=0.9", restored4_hb, n_hb),
    ("Важка кулька beta оптимальне", restored4_hb_opt, n_hb_opt),
    ("Метод Нестерова", restored4_nag, n_nag),
    ("Нестеров для сильно опуклої задачі", restored4_nag_strong, n_nag_strong)
]

print("\nПорівняння градієнтних методів:")
print(f"{'Метод':42s} {'Ітерації':>10s} {'PSNR, дБ':>12s}")

for name, image, iterations in gradient_results:
    print(
        f"{name:42s} "
        f"{iterations:10d} "
        f"{psnr(image, clean):12.4f}"
    )

methods4 = [
    ("Градієнтний спуск (h = 1/L)", history_gd, restored4_gd, n_gd,
     f"h = 1/L = {1.0 / L_lip:.4f}"),
    ("Метод Арміхо", history_armijo, restored4_armijo, n_armijo,
     "крок 2.0, множник 0.6"),
    ("Важка кулька (оптимальне β)", history_hb_opt, restored4_hb_opt,
     n_hb_opt, f"β = {beta_hb_opt:.4f}"),
    ("Важка кулька β = 0.9", history_hb, restored4_hb, n_hb, "β = 0.9"),
    ("Нестеров NAG", history_nag, restored4_nag, n_nag, "β_k = (k−1)/(k+2)"),
    ("Нестеров NAG (сильно опукла)", history_nag_strong,
     restored4_nag_strong, n_nag_strong, f"β = {beta_nag_strong:.4f}"),
]

for idx, (name, hist, img, n_it, par) in enumerate(methods4, start=1):
    print_table(
        f"ЛР4 — Таблиця ітерацій: {name}",
        ["k", "E(I_k)", "E(I_k) − E*", "‖∇E(I_k)‖", "‖I_k − I*‖_F"],
        [(r[0] - 1, r[1], r[1] - E_star, r[2], r[3]) for r in hist],
        f"lab4_iterations_{idx}", max_rows=15,
    )

cmp4, E_gaps_converged = [], []
for name, hist, img, n_it, par in methods4:
    conv = hist[-1][2] <= 1e-6
    gap = energy(img, noisy, lam) - E_star
    if conv:
        E_gaps_converged.append(gap)
    cmp4.append((name, par, n_it, hist[-1][2], gap,
                 float(np.linalg.norm(img - I_star)), psnr(img, clean)))
print_table(
    "ЛР4 — Порівняння градієнтних методів",
    ["Метод", "Параметр", "Ітерацій", "‖∇E‖ (кінець)",
     "E − E*", "‖I − I*‖_F", "PSNR, дБ"],
    cmp4, "lab4_comparison",
)
conclusion_table(
    "ЛР4 — Висновкова таблиця",
    [
        ("λ*", lam, 0.284193, 5e-7, ""),
        ("L = 1 + 8λ*", L_lip, 3.273546, 5e-6, ""),
        ("E*", E_star, 77.198771, 1e-4, "точний мінімум, у PDF — після зупинки"),
        ("β оптимальне (важка кулька)", beta_hb_opt, 0.0830, 5e-5, ""),
        ("Ітерацій: градієнтний спуск", n_gd, 36, None, ""),
        ("Ітерацій: важка кулька, оптимальне β", n_hb_opt, 30, None, ""),
        ("Ітерацій: важка кулька β = 0.9", n_hb, 200, None, "не збігся"),
        ("Ітерацій: Нестеров, сильно опуклий", n_nag_strong, 18, None, ""),
        ("Ітерацій: Арміхо", n_armijo, None, None, "в PDF немає"),
        ("Ітерацій: Нестеров NAG", n_nag, None, None, "сайт: 46"),
        ("Усі збіжні методи дають однакове E*",
         max(abs(g) for g in E_gaps_converged) < 1e-8, True, None, ""),
        ("PSNR, дБ", psnr(restored4_gd, clean), 23.2834, 5e-5, ""),
    ],
    "lab4_conclusion",
)

# ГРАФІК ЛР4 — збіжність градієнтних методів 
print(f"E* = {E_star:.6f}")

fig_l4, (ax_l4_1, ax_l4_2) = plt.subplots(1, 2, figsize=(14, 6))

grad_histories = [
    ("Градієнтний спуск (h = 1/L)", history_gd, "C0", "-"),
    ("Градієнтний спуск (Армійо)", history_armijo, "red", "--"),
    ("Важка кулька (оптимальне β)", history_hb_opt, "C2", "-"),
    ("Важка кулька β = 0.9 (завелике)", history_hb, "darkorange", "--"),
    ("Нестеров NAG", history_nag, "C4", "-"),
    ("Нестеров NAG (сильно опукла)", history_nag_strong, "C5", "--"),
]

for name, hist, color, ls in grad_histories:
    its = [row[0] - 1 for row in hist]                    # k = 0, 1, 2, ...
    gaps = [max(row[1] - E_star, 1e-16) for row in hist]  # E(I_k) - E*
    dists = [max(row[3], 1e-16) for row in hist]          # ||I_k - I*||_F
    ax_l4_1.plot(its, gaps, color=color, linestyle=ls, linewidth=2, label=name)
    ax_l4_2.plot(its, dists, color=color, linestyle=ls, linewidth=2, label=name)

ax_l4_1.set_yscale("log")
ax_l4_1.set_xlabel("Ітерація k")
ax_l4_1.set_ylabel("E(I_k) − f*")
ax_l4_1.set_title("Збіжність за значенням функції")
ax_l4_1.grid(True, alpha=0.3)
ax_l4_1.legend(fontsize=8)

ax_l4_2.set_yscale("log")
ax_l4_2.set_xlabel("Ітерація k")
ax_l4_2.set_ylabel("||I_k − I*||_F")
ax_l4_2.set_title("Відстань до точного розв'язку")
ax_l4_2.grid(True, alpha=0.3)
ax_l4_2.legend(fontsize=8)

fig_l4.tight_layout()
plt.show()


# ЛАБОРАТОРНА РОБОТА 5 — МЕТОДИ З ОБМЕЖЕННЯМИ

def project(image, lower=0.0, upper=1.0):
    """Проєкція на допустимий проміжок [0, 1]."""
    return np.clip(image, lower, upper)


def violation(image):
    """Максимальне порушення та кількість пікселів поза [0, 1]."""
    below = np.maximum(0.0, -image)
    above = np.maximum(0.0, image - 1.0)
    max_violation = float(max(below.max(), above.max()))
    outside_count = int(np.count_nonzero((image < 0.0) | (image > 1.0)))
    return max_violation, outside_count


def projected_gradient(target, lam, max_iter=200, tol=1e-6, accelerated=False):
    """Проєкційний градієнтний метод та прискорений варіант з історією."""
    L = 1.0 + 8.0 * lam
    h = 1.0 / L

    image = project(target.copy())
    y = image.copy()
    t_fista = 1.0
    history = []  # Зберігаємо історію
    converged = False

    for k in range(1, max_iter + 1):
        point = y if accelerated else image
        g = energy_grad(point, target, lam)
        new_image = project(point - h * g)
        
        gradient_mapping = float(np.linalg.norm(point - new_image) / h)
        history.append((k, gradient_mapping, energy(image, target, lam)))

        if accelerated:
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_fista ** 2)) / 2.0
            beta = (t_fista - 1.0) / t_next
            y = new_image + beta * (new_image - image)
            image = new_image
            t_fista = t_next
        else:
            image = new_image

        if gradient_mapping <= tol:
            converged = True
            break

    # k — кількість обчислень градієнта; зроблених ітерацій на 1 менше
    n_iter = k - 1 if converged else k
    return image, n_iter, gradient_mapping, history


def penalty_value(image):
    """Значення штрафної функції."""
    return float(np.sum(np.maximum(0.0, -image) ** 2 + np.maximum(0.0, image - 1.0) ** 2))


def penalty_grad(image):
    """Градієнт штрафної функції."""
    return -2.0 * np.maximum(0.0, -image) + 2.0 * np.maximum(0.0, image - 1.0)


def penalty_method(target, lam, outer=7, inner=100):
    """Штрафний метод із поступовим збільшенням коефіцієнта штрафу."""
    image = target.copy()
    r = 1.0
    rows = []
    n_calls = 0

    for s in range(outer):
        h = 1.0 / (1.0 + 8.0 * lam + 2.0 * r)
        previous = image.copy()

        for k in range(1, inner + 1):
            beta = (k - 1.0) / (k + 2.0)
            y = image + beta * (image - previous)
            g = energy_grad(y, target, lam) + r * penalty_grad(y)
            n_calls += 1
            new_image = y - h * g
            previous, image = image, new_image
            if np.linalg.norm(g) <= 1e-6:
                break

        max_v, count = violation(image)
        p = penalty_value(image)
        energy_value = energy(image, target, lam)
        rows.append((s + 1, r, energy_value, p, r * p, max_v, count))
        if r * p <= 1e-8:
            break
        r *= 10.0

    return image, rows, n_calls


# Безумовний розв'язок без проєкції
restored_uncond, n_uncond, _ = gradient_method(raw, lam, kind="gd", start=raw)
viol_raw = violation(raw)
viol_uncond = violation(restored_uncond)

# Отримуємо всі 4 значення разом із hist_proj та hist_fista
restored5_proj, it_proj, gmap_proj, hist_proj = projected_gradient(raw, lam, accelerated=False)
restored5_fista, it_fista, gmap_fista, hist_fista = projected_gradient(raw, lam, accelerated=True)
restored5_pen, penalty_rows, pen_calls = penalty_method(raw, lam)

viol_proj = violation(restored5_proj)
viol_fista = violation(restored5_fista)
viol_pen = violation(restored5_pen)

save_image(restored5_proj, "05_projected_gradient")
save_image(restored5_fista, "05_projected_accelerated")
save_image(restored5_pen, "05_penalty")

print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА 5 — МЕТОДИ З ОБМЕЖЕННЯМИ")
print("=" * 70)
print(f"\nСире зображення поза [0, 1]: {viol_raw[1]} пікселів")
print(f"Максимальне порушення raw: {viol_raw[0]:.6f}")
print(f"\nБезумовний розв'язок поза [0, 1]: {viol_uncond[1]} пікселів")
print(f"Максимальне порушення безумовного розв'язку: {viol_uncond[0]:.6f}")

print("\nПроєкційний градієнтний метод:")
print(f"Ітерацій: {it_proj}")
print(f"Відображення градієнта: {gmap_proj:.6e}")
print(f"Порушення обмежень: {viol_proj[0]:.6e}")
print(f"Пікселів поза [0, 1]: {viol_proj[1]}")
print(f"PSNR: {psnr(restored5_proj, clean):.4f} дБ")

print("\nПрискорений проєкційний метод:")
print(f"Ітерацій: {it_fista}")
print(f"Відображення градієнта: {gmap_fista:.6e}")
print(f"Порушення обмежень: {viol_fista[0]:.6e}")
print(f"Пікселів поза [0, 1]: {viol_fista[1]}")
print(f"PSNR: {psnr(restored5_fista, clean):.4f} дБ")

print("\nШтрафний метод:")
print(f"Зовнішніх ітерацій: {len(penalty_rows)}")
print(f"Внутрішніх кроків: {pen_calls}")
print(f"Порушення обмежень: {viol_pen[0]:.6e}")
print(f"Пікселів поза [0, 1]: {viol_pen[1]}")
print(f"PSNR: {psnr(restored5_pen, clean):.4f} дБ")

print("\nТаблиця штрафного методу:")
print(f"{'s':>3} {'r':>8} {'E':>12} {'P':>12} {'rP':>12} {'max порушення':>16} {'поза [0,1]':>12}")
for row in penalty_rows:
    s, r, e, p, rp, max_v, count = row
    print(f"{s:3d} {r:8.0f} {e:12.6f} {p:12.3e} {rp:12.3e} {max_v:16.3e} {count:12d}")

_hdr5 = ["k", "E(I_k)", "норма відображення градієнта"]
print_table("ЛР5 — Таблиця ітерацій: проєкційний градієнтний метод", _hdr5,
            [(r[0] - 1, r[2], r[1]) for r in hist_proj],
            "lab5_projection_iterations", max_rows=15)
print_table("ЛР5 — Таблиця ітерацій: проєкція з прискоренням (FISTA)", _hdr5,
            [(r[0] - 1, r[2], r[1]) for r in hist_fista],
            "lab5_fista_iterations", max_rows=15)
print_table(
    "ЛР5 — Таблиця ітерацій: штрафний метод (зовнішні ітерації)",
    ["s", "r", "E", "P", "r·P", "макс. порушення", "пікселів поза [0,1]"],
    penalty_rows, "lab5_penalty_iterations",
)

print_table(
    "ЛР5 — Порівняння методів",
    ["Метод", "E(x*)", "Макс. порушення", "Пікселів поза [0,1]",
     "Допустимість", "Ітерацій", "N_grad", "PSNR, дБ"],
    [
        ("Безумовний розв'язок", energy(restored_uncond, raw, lam),
         viol_uncond[0], viol_uncond[1], viol_uncond[1] == 0,
         n_uncond, n_uncond + 1, psnr(restored_uncond, clean)),
        ("Проєкція градієнта", energy(restored5_proj, raw, lam),
         viol_proj[0], viol_proj[1], viol_proj[1] == 0,
         it_proj, it_proj + 1, psnr(restored5_proj, clean)),
        ("Проєкція з прискоренням (FISTA)", energy(restored5_fista, raw, lam),
         viol_fista[0], viol_fista[1], viol_fista[1] == 0,
         it_fista, it_fista + 1, psnr(restored5_fista, clean)),
        ("Квадратичний штраф (Нестеров)", energy(restored5_pen, raw, lam),
         viol_pen[0], viol_pen[1], viol_pen[1] == 0,
         len(penalty_rows), pen_calls, psnr(restored5_pen, clean)),
    ],
    "lab5_comparison",
)
conclusion_table(
    "ЛР5 — Висновкова таблиця",
    [
        ("Сире поза [0, 1], пікселів", viol_raw[1], 1406, None, ""),
        ("Безумовний розв'язок поза [0, 1], пікселів", viol_uncond[1], 561,
         None, ""),
        ("Макс. порушення безумовного розв'язку", viol_uncond[0], 0.175288,
         5e-7, ""),
        ("Проєкція: ітерацій", it_proj, 37, None, ""),
        ("Проєкція: звернень (f + градієнт)", 2 * (it_proj + 1), 76, None, ""),
        ("FISTA: ітерацій", it_fista, 47, None, "сайт: 47"),
        ("Порушення у проєкції", viol_proj[0], 0.0, 0.0, "рівно нуль"),
        ("Штраф: зовнішніх ітерацій", len(penalty_rows), 7, None, ""),
        ("Штраф: звернень до градієнта", pen_calls, 688, None,
         "сайт: 681; PDF = 681 + 7 зовн. ітерацій"),
        ("Штраф: пікселів поза [0, 1]", viol_pen[1], 460, None, "значення з сайту"),
        ("Штраф дорожчий за проєкцію", pen_calls > 2 * (it_proj + 1), True,
         None, "на порядок"),
        ("PSNR проєкції, дБ", psnr(restored5_proj, clean), 23.3441, 5e-5, ""),
    ],
    "lab5_conclusion",
)

# ГРАФІК ЛР5 — гістограма яскравостей та збіжність проєкційних методів
fig_l5, (ax_l5_1, ax_l5_2) = plt.subplots(1, 2, figsize=(14, 6))

ax_l5_1.hist(raw.flatten(), bins=100, alpha=0.55, color="C0",
             label=f"сире спостереження (поза [0, 1]: {viol_raw[1]})")
ax_l5_1.hist(restored_uncond.flatten(), bins=100, alpha=0.55, color="C1",
             label=f"безумовний розв'язок (поза [0, 1]: {viol_uncond[1]})")
ax_l5_1.axvline(0.0, color="red", linestyle="--", linewidth=1.5,
                label="межі допустимої області")
ax_l5_1.axvline(1.0, color="red", linestyle="--", linewidth=1.5)
ax_l5_1.set_xlabel("яскравість пікселя")
ax_l5_1.set_ylabel("кількість пікселів")
ax_l5_1.set_title("Гістограма яскравостей (як у практикумі, крок 5.1)")
ax_l5_1.legend(fontsize=8)
ax_l5_1.grid(True, alpha=0.3)

ax_l5_2.plot([row[0] - 1 for row in hist_proj], [row[2] for row in hist_proj],
             color="C0", linewidth=2, label="Проєкція градієнта")
ax_l5_2.plot([row[0] - 1 for row in hist_fista], [row[2] for row in hist_fista],
             color="red", linestyle="--", linewidth=2,
             label="Проєкція градієнта з прискоренням (FISTA)")
ax_l5_2.set_yscale("log")
ax_l5_2.set_xlabel("Ітерація k")
ax_l5_2.set_ylabel("E(I_k)")
ax_l5_2.set_title("Збіжність проєкційних методів")
ax_l5_2.legend()
ax_l5_2.grid(True, alpha=0.3)

fig_l5.tight_layout()
plt.show()


#підсумкова таблиця якості відновлення зображення 

results = [
    ("Шумне зображення", noisy),
    ("ЛР2 — гаусівський фільтр", restored2),
    ("ЛР3 — Нелдер—Мід", restored3_nm),
    ("ЛР3 — Пауелл", restored3_pw),
    ("ЛР4 — градієнтний спуск", restored4_gd),
    ("ЛР4 — Арміхо", restored4_armijo),
    ("ЛР4 — важка кулька", restored4_hb),
    ("ЛР4 — важка кулька оптимальна", restored4_hb_opt),
    ("ЛР4 — Нестеров", restored4_nag),
    ("ЛР4 — Нестеров сильна опуклість", restored4_nag_strong),
    ("ЛР5 — проєкційний метод", restored5_proj),
    ("ЛР5 — прискорена проєкція", restored5_fista),
    ("ЛР5 — штрафний метод", restored5_pen)
]

print("\n" + "=" * 85)
print("ПІДСУМКОВА ТАБЛИЦЯ")
print("=" * 85)
print(f"{'Метод':43s} {'MSE':>14s} {'PSNR, дБ':>14s}")

for name, image in results:
    print(
        f"{name:43s} "
        f"{mse(image, clean):14.8f} "
        f"{psnr(image, clean):14.4f}"
    )


print_table(
    "ПІДСУМКОВА ТАБЛИЦЯ",
    ["Метод", "MSE", "PSNR, дБ"],
    [(name, mse(image, clean), psnr(image, clean)) for name, image in results],
    "summary_all_methods", show=False,
)

#візуалізація результатів
images = [
    (clean, "Еталон"),
    (noisy, "Шумне зображення"),
    (restored2, "ЛР2: гаусівський фільтр"),
    (restored3_nm, "ЛР3: Нелдер—Мід"),
    (restored3_pw, "ЛР3: Пауелл"),
    (restored4_gd, "ЛР4: градієнтний спуск"),
    (restored4_nag, "ЛР4: Нестеров"),
    (restored5_proj, "ЛР5: проєкція"),
    (restored5_pen, "ЛР5: штрафний метод")
]

fig, axes = plt.subplots(3, 3, figsize=(13, 12))

for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"{title}\nPSNR = {psnr(img, clean):.2f} дБ"
    )
    ax.axis("off")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "comparison_images.png", dpi=150)


# Діаграма PSNR

labels = [name for name, image in results]
values = [psnr(image, clean) for name, image in results]

fig2, ax = plt.subplots(figsize=(14, 6))

ax.bar(range(len(labels)), values)
ax.set_ylabel("PSNR, дБ")
ax.set_title("Порівняння якості відновлення зображення")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=60, ha="right")
ax.grid(axis="y", alpha=0.3)

fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "comparison_psnr.png", dpi=150)

plt.show()

print("\n" + "=" * 70)
print("ВИКОНАННЯ ПРОГРАМИ ЗАВЕРШЕНО")
print(f"Результати збережено в папці: {OUTPUT_DIR.resolve()}")
print("=" * 70)