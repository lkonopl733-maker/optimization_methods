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

def swann(oracle, x0, delta, max_iter=100):
    """Знаходження інтервалу невизначеності методом Свенна."""

    f0 = oracle.value(x0)

    x_right = x0 + delta
    f_right = oracle.value(x_right)

    if f_right >= f0:
        x_left = x0 - delta
        f_left = oracle.value(x_left)

        if f_left >= f0:
            return (x_left, x_right), 0

        direction = -1.0
        x = x_left
        f_current = f_left
        x_previous = x0

    else:
        direction = 1.0
        x = x_right
        f_current = f_right
        x_previous = x0

    step = delta

    for iteration in range(1, max_iter + 1):
        step *= 2.0
        x_next = x + direction * step
        f_next = oracle.value(x_next)

        if f_next >= f_current:
            return tuple(sorted((x_previous, x_next))), iteration

        x_previous = x
        x = x_next
        f_current = f_next

    raise RuntimeError("Метод Свенна: перевищено ліміт ітерацій.")


oracle_sw = Oracle1D(objective_sigma)

interval, swann_iters = swann(
    oracle_sw,
    X0_1D,
    DELTA0_1D
)

print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА 1 — МЕТОД СВЕННА")
print("=" * 70)
print(f"x0 = {X0_1D}")
print(f"delta0 = {DELTA0_1D}")
print(f"Інтервал: [{interval[0]:.6f}; {interval[1]:.6f}]")
print(f"Ітерацій: {swann_iters}")
print(f"Обчислень функції: {oracle_sw.n_f}")


#лабораторна робота 2 дихотомія і золотий переріз

def dichotomy(oracle, a, b, eps, max_iter=500):
    """Метод дихотомії."""

    iteration = 0
    delta = eps / 4.0

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

    return (a + b) / 2.0, iteration


def golden(oracle, a, b, eps, max_iter=500):
    """Метод золотого перерізу."""

    tau = (math.sqrt(5.0) - 1.0) / 2.0

    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)

    f1 = oracle.value(x1)
    f2 = oracle.value(x2)

    iteration = 0

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

    return (a + b) / 2.0, iteration


oracle_d = Oracle1D(objective_sigma)
sigma_d, it_d = dichotomy(
    oracle_d, interval[0], interval[1], EPS_1D
)

oracle_g = Oracle1D(objective_sigma)
sigma_g, it_g = golden(
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


# лабораторна робота 3 білатеральний фільтр і методи Нелдера-Міда та Пауелла

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])


def bilateral_filter(img, sigma_spatial, sigma_range):
    """Білатеральний фільтр."""

    ss = max(float(sigma_spatial), 1e-3)
    sr = max(float(sigma_range), 1e-3)

    radius = int(min(max(round(2.0 * ss), 1), 5))

    padded = np.pad(img, radius, mode="reflect")

    numerator = np.zeros_like(img)
    denominator = np.zeros_like(img)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):

            shifted = padded[
                radius + dy:radius + dy + img.shape[0],
                radius + dx:radius + dx + img.shape[1]
            ]

            spatial_weight = math.exp(
                -(dx * dx + dy * dy) / (2.0 * ss * ss)
            )

            range_weight = np.exp(
                -((shifted - img) ** 2) / (2.0 * sr * sr)
            )

            weight = spatial_weight * range_weight

            numerator += weight * shifted
            denominator += weight

    return numerator / np.maximum(denominator, 1e-15)


def objective2(theta):
    """Цільова функція для налаштування білатерального фільтра."""

    theta = np.asarray(theta, dtype=float)
    clamped = np.clip(theta, LO, HI)

    gap = theta - clamped

    filtered = bilateral_filter(
        noisy,
        clamped[0],
        clamped[1]
    )

    return mse(filtered, clean) + 1000.0 * float(gap @ gap)


def nelder_mead(oracle, x0, step, tol=1e-7, max_iter=200):
    """Метод Нелдера—Міда."""

    x0 = np.asarray(x0, dtype=float)

    simplex = [
        np.clip(x0, LO, HI),
        np.clip(x0 + [step, 0.0], LO, HI),
        np.clip(x0 + [0.0, 0.4 * step], LO, HI)
    ]

    values = [oracle.value(point) for point in simplex]

    for iteration in range(1, max_iter + 1):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        best, middle, worst = simplex
        f_best, f_middle, f_worst = values

        center = (best + middle) / 2.0

        # Відбиття
        reflected = np.clip(
            center + (center - worst), LO, HI
        )
        f_reflected = oracle.value(reflected)

        if f_reflected < f_best:
            # Розтягування
            expanded = np.clip(
                center + 2.0 * (reflected - center), LO, HI
            )
            f_expanded = oracle.value(expanded)

            if f_expanded < f_reflected:
                simplex[2] = expanded
                values[2] = f_expanded
            else:
                simplex[2] = reflected
                values[2] = f_reflected

        elif f_reflected < f_middle:
            simplex[2] = reflected
            values[2] = f_reflected

        else:
            # Зовнішнє або внутрішнє стиснення
            if f_reflected < f_worst:
                contracted = np.clip(
                    center + 0.5 * (reflected - center), LO, HI
                )
                f_contracted = oracle.value(contracted)

                if f_contracted <= f_reflected:
                    simplex[2] = contracted
                    values[2] = f_contracted
                else:
                    for i in (1, 2):
                        simplex[i] = np.clip(
                            best + 0.5 * (simplex[i] - best), LO, HI
                        )
                        values[i] = oracle.value(simplex[i])

            else:
                contracted = np.clip(
                    center + 0.5 * (worst - center), LO, HI
                )
                f_contracted = oracle.value(contracted)

                if f_contracted < f_worst:
                    simplex[2] = contracted
                    values[2] = f_contracted
                else:
                    for i in (1, 2):
                        simplex[i] = np.clip(
                            best + 0.5 * (simplex[i] - best), LO, HI
                        )
                        values[i] = oracle.value(simplex[i])

        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        size = max(
            np.linalg.norm(point - simplex[0])
            for point in simplex
        )

        if (
            size <= tol
            and max(values) - min(values) <= tol
        ):
            break

    best_index = int(np.argmin(values))

    return simplex[best_index], values[best_index], iteration


def line_minimize(oracle, x, direction, tol=1e-5):
    """Мінімізація вздовж заданого напрямку."""

    a, b = -1.0, 1.0
    tau = (math.sqrt(5.0) - 1.0) / 2.0

    c = b - tau * (b - a)
    d = a + tau * (b - a)

    fc = oracle.value(np.clip(x + c * direction, LO, HI))
    fd = oracle.value(np.clip(x + d * direction, LO, HI))

    for _ in range(100):
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

    alpha = (a + b) / 2.0

    return np.clip(x + alpha * direction, LO, HI)


def powell(oracle, x0, tol=1e-5, max_cycles=40):
    """Метод Пауелла."""

    x = np.clip(np.asarray(x0, dtype=float), LO, HI)

    directions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0])
    ]

    for cycle in range(1, max_cycles + 1):
        start = x.copy()

        for direction in directions:
            x = line_minimize(oracle, x, direction)

        displacement = x - start

        if np.linalg.norm(displacement) < tol:
            break

        x = line_minimize(oracle, x, displacement)

        norm_displacement = np.linalg.norm(displacement)

        if norm_displacement > 1e-15:
            directions = [
                directions[1],
                displacement / norm_displacement
            ]

    return x, oracle.value(x), cycle


sigma_s0 = min(max(2.0 * sigma_g, 0.5), 4.0)
sigma_r0 = min(max(0.15 + 0.1 * SIGMA_NOISE, 0.02), 0.9)

oracle_nm = Oracle2D(objective2)
theta_nm, f_nm, it_nm = nelder_mead(
    oracle_nm,
    [sigma_s0, sigma_r0],
    SIMPLEX_STEP
)

oracle_pw = Oracle2D(objective2)
theta_pw, f_pw, cycles_pw = powell(
    oracle_pw,
    [sigma_s0, sigma_r0]
)

restored3_nm = bilateral_filter(
    noisy, theta_nm[0], theta_nm[1]
)

restored3_pw = bilateral_filter(
    noisy, theta_pw[0], theta_pw[1]
)

save_image(restored3_nm, "03_bilateral_nelder_mead")
save_image(restored3_pw, "03_bilateral_powell")

print("\n" + "=" * 70)
print("ЛАБОРАТОРНА РОБОТА 3 — БІЛАТЕРАЛЬНИЙ ФІЛЬТР")
print("=" * 70)

print("\nМетод Нелдера—Міда:")
print(f"Sigma spatial = {theta_nm[0]:.6f}")
print(f"Sigma range = {theta_nm[1]:.6f}")
print(f"Цільова функція = {f_nm:.8f}")
print(f"Ітерацій = {it_nm}")
print(f"Обчислень функції = {oracle_nm.n_f}")
print(f"PSNR = {psnr(restored3_nm, clean):.4f} дБ")

print("\nМетод Пауелла:")
print(f"Sigma spatial = {theta_pw[0]:.6f}")
print(f"Sigma range = {theta_pw[1]:.6f}")
print(f"Цільова функція = {f_pw:.8f}")
print(f"Циклів = {cycles_pw}")
print(f"Обчислень функції = {oracle_pw.n_f}")
print(f"PSNR = {psnr(restored3_pw, clean):.4f} дБ")


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
    start=None
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
                step = h
                g2 = float(np.sum(g * g))

                for _ in range(50):
                    candidate = image - step * g
                    candidate_energy = energy(
                        candidate, target, lam
                    )

                    if candidate_energy <= f0 - 1e-4 * step * g2:
                        break

                    step *= 0.5

                new_image = candidate

            else:
                new_image = image - h * g

        gradient_norm = float(np.linalg.norm(g))
        current_energy = energy(image, target, lam)

        rows.append((k, current_energy, gradient_norm))

        previous, image = image, new_image

        if gradient_norm <= tol:
            break

    return image, len(rows), rows



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
        max_iter=80,
        tol=1e-4
    )

    return mse(restored, clean)


oracle_lam = Oracle1D(objective_lambda)

lambda_star, it_lam = golden(
    oracle_lam,
    0.05,
    min(3.0, 2.0 * LAMBDA0),
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


restored4_gd, n_gd, history_gd = gradient_method(
    noisy, lam, kind="gd"
)

restored4_armijo, n_armijo, history_armijo = gradient_method(
    noisy, lam, kind="armijo"
)

restored4_hb, n_hb, history_hb = gradient_method(
    noisy, lam, kind="hb", beta=0.9
)

restored4_hb_opt, n_hb_opt, history_hb_opt = gradient_method(
    noisy, lam, kind="hb", beta=beta_hb_opt
)

restored4_nag, n_nag, history_nag = gradient_method(
    noisy, lam, kind="nag"
)

restored4_nag_strong, n_nag_strong, history_nag_strong = (
    gradient_method(
        noisy, lam, kind="nag_strong",
        beta=beta_nag_strong
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


# лабораторна робота 5 проєкційний градієнтний та штрафний методи

def project(image, lower=0.0, upper=1.0):
    """Проєкція на допустимий проміжок [0, 1]."""
    return np.clip(image, lower, upper)


def violation(image):
    """Максимальне порушення та кількість пікселів поза [0, 1]."""

    below = np.maximum(0.0, -image)
    above = np.maximum(0.0, image - 1.0)

    max_violation = float(max(below.max(), above.max()))

    outside_count = int(
        np.count_nonzero((image < 0.0) | (image > 1.0))
    )

    return max_violation, outside_count


def projected_gradient(
    target,
    lam,
    max_iter=200,
    tol=1e-6,
    accelerated=False
):
    """Проєкційний градієнтний метод та прискорений варіант."""

    L = 1.0 + 8.0 * lam
    h = 1.0 / L

    image = project(target.copy())
    y = image.copy()
    t_fista = 1.0
    gradient_mapping = float("inf")

    for k in range(1, max_iter + 1):
        point = y if accelerated else image

        g = energy_grad(point, target, lam)
        new_image = project(point - h * g)

        # Норма відображення градієнта
        gradient_mapping = float(
            np.linalg.norm(point - new_image) / h
        )

        if accelerated:
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_fista ** 2)) / 2.0
            beta = (t_fista - 1.0) / t_next

            y = new_image + beta * (new_image - image)
            image = new_image
            t_fista = t_next
        else:
            image = new_image

        if gradient_mapping <= tol:
            break

    return image, k, gradient_mapping


def penalty_value(image):
    """Значення штрафної функції."""

    return float(
        np.sum(
            np.maximum(0.0, -image) ** 2
            + np.maximum(0.0, image - 1.0) ** 2
        )
    )


def penalty_grad(image):
    """Градієнт штрафної функції."""

    return (
        -2.0 * np.maximum(0.0, -image)
        + 2.0 * np.maximum(0.0, image - 1.0)
    )


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

            g = (
                energy_grad(y, target, lam)
                + r * penalty_grad(y)
            )

            n_calls += 1
            new_image = y - h * g

            previous, image = image, new_image

            if np.linalg.norm(g) <= 1e-6:
                break

        max_v, count = violation(image)
        p = penalty_value(image)
        energy_value = energy(image, target, lam)

        rows.append((
            s + 1,
            r,
            energy_value,
            p,
            r * p,
            max_v,
            count
        ))

        if r * p <= 1e-8:
            break

        r *= 10.0

    return image, rows, n_calls


# Безумовний розв'язок без проєкції
restored_uncond, n_uncond, _ = gradient_method(
    raw,
    lam,
    kind="gd",
    start=raw
)

viol_raw = violation(raw)
viol_uncond = violation(restored_uncond)

# Проєкційний метод
restored5_proj, it_proj, gmap_proj = projected_gradient(
    raw, lam, accelerated=False
)

# Прискорений проєкційний метод
restored5_fista, it_fista, gmap_fista = projected_gradient(
    raw, lam, accelerated=True
)

# Штрафний метод
restored5_pen, penalty_rows, pen_calls = penalty_method(
    raw, lam
)

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
print(
    f"{'s':>3} {'r':>8} {'E':>12} {'P':>12} "
    f"{'rP':>12} {'max порушення':>16} {'поза [0,1]':>12}"
)

for row in penalty_rows:
    s, r, e, p, rp, max_v, count = row

    print(
        f"{s:3d} {r:8.0f} {e:12.6f} {p:12.3e} "
        f"{rp:12.3e} {max_v:16.3e} {count:12d}"
    )


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