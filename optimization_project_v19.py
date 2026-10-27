import math
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ПРОЄКТНА РОБОТА: ВІДНОВЛЕННЯ ЦИФРОВОГО ЗОБРАЖЕННЯ
# Варіант 19
# ============================================================

VARIANT = 19
SEED = 255190
SIGMA_NOISE = 0.1020

# Початкові параметри для методу Свенна
X0_1D = 0.3648
DELTA0_1D = 0.2195
EPS_1D = 1e-4

# Параметри для наступних лабораторних
LAMBDA0 = 1.2912
SIMPLEX_STEP = 0.3924

assert (VARIANT * 7919 + 104729) % (2**31 - 1) == SEED


# ============================================================
# ЛАБОРАТОРНА РОБОТА 0
# СТВОРЕННЯ ЕТАЛОННОГО ЗОБРАЖЕННЯ ТА ДОДАВАННЯ ШУМУ
# ============================================================

class Rng:
    """Генератор псевдовипадкових чисел."""

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
        return (self.next_u64() >> 11) * (
            1.0 / 9007199254740992.0
        )

    def normal(self):
        u1 = max(self.uniform(), 1e-300)
        u2 = self.uniform()

        return math.sqrt(-2.0 * math.log(u1)) * math.cos(
            2.0 * math.pi * u2
        )


def clean_image(size=128):
    """Створення еталонного зображення."""

    img = np.zeros((size, size), dtype=float)

    for i in range(size):
        for j in range(size):
            u = j / (size - 1)
            v = i / (size - 1)

            # Фоновий градієнт
            val = 0.25 + 0.35 * (0.6 * u + 0.4 * v)

            # Кругова структура
            val += 0.08 * math.cos(
                2 * math.pi * math.sqrt(
                    (u - 0.5) ** 2 + (v - 0.5) ** 2
                )
            )

            # Світле коло
            if (u - 0.32) ** 2 + (v - 0.34) ** 2 <= 0.16 ** 2:
                val = 0.92

            # Темний квадрат
            if abs(u - 0.70) <= 0.15 and abs(v - 0.30) <= 0.15:
                val = 0.10

            # Трикутник
            if 0.60 < v < 0.90 and abs(u - 0.30) < 0.9 * (v - 0.60):
                val = 0.68

            # Шахова структура
            if 0.58 < u < 0.92 and 0.60 < v < 0.90:
                cell = max(3, size // 12)

                if ((j // cell) + (i // cell)) % 2 == 0:
                    val = 0.85
                else:
                    val = 0.20

            img[i, j] = np.clip(val, 0, 1)

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

    # raw — зображення без обмеження діапазону;
    # noisy — зображення з пікселями в межах [0, 1].
    return raw, np.clip(raw, 0, 1)


def mse(a, b):
    """Середньоквадратична помилка."""
    return float(np.mean((a - b) ** 2))


def psnr(a, b):
    """Пікове співвідношення сигнал/шум."""
    error = mse(a, b)

    if error == 0:
        return float("inf")

    return 10 * math.log10(1.0 / error)


def gaussian_blur(img, sigma):
    """Сепарабельне гаусівське розмиття."""

    if sigma <= 1e-6:
        return img.copy()

    radius = int(4.0 * sigma + 0.5)
    t = np.arange(-radius, radius + 1, dtype=float)

    kernel = np.exp(-(t * t) / (2 * sigma * sigma))
    kernel /= kernel.sum()

    height, width = img.shape
    temp = np.zeros_like(img)
    result = np.zeros_like(img)

    # Горизонтальний прохід
    for z in range(-radius, radius + 1):
        indices = np.clip(
            np.arange(width) + z,
            0,
            width - 1
        )

        temp += kernel[z + radius] * img[:, indices]

    # Вертикальний прохід
    for z in range(-radius, radius + 1):
        indices = np.clip(
            np.arange(height) + z,
            0,
            height - 1
        )

        result += kernel[z + radius] * temp[indices, :]

    return result


# Створення зображень
clean = clean_image(128)
raw, noisy = add_noise(clean, SIGMA_NOISE, SEED)

print("=" * 65)
print("ЛАБОРАТОРНА РОБОТА 0")
print("=" * 65)
print(f"Варіант: {VARIANT}")
print(f"Seed: {SEED}")
print(f"Середньоквадратична помилка шумного зображення: {mse(noisy, clean):.8f}")
print(f"PSNR шумного зображення: {psnr(noisy, clean):.4f} дБ")


# ============================================================
# ДОПОМІЖНІ КЛАСИ ДЛЯ ПІДРАХУНКУ ОБЧИСЛЕНЬ
# ============================================================

class Oracle1D:
    """Функція для одновимірної оптимізації з підрахунком викликів."""

    def __init__(self, fn):
        self.fn = fn
        self.n_f = 0

    def value(self, x):
        self.n_f += 1
        return self.fn(float(x))


def objective_sigma(sigma):
    """Цільова функція для пошуку оптимального sigma."""

    sigma = max(float(sigma), 1e-6)
    restored = gaussian_blur(noisy, sigma)

    return mse(restored, clean)


# ============================================================
# ЛАБОРАТОРНА РОБОТА 1
# МЕТОД СВЕННА
# ============================================================

def swann(oracle, x0, delta, max_iter=100):
    """Пошук інтервалу невизначеності методом Свенна."""

    f0 = oracle.value(x0)
    fr = oracle.value(x0 + delta)

    if fr >= f0:
        fl = oracle.value(x0 - delta)

        if fl >= f0:
            return (x0 - delta, x0 + delta), 0

        direction = -1.0
    else:
        direction = 1.0

    x_prev = x0
    x = x0 + direction * delta
    step = direction * delta

    for k in range(1, max_iter + 1):
        step *= 2
        x_next = x + step

        f_next = oracle.value(x_next)
        f_current = oracle.value(x)

        if f_next >= f_current:
            return tuple(sorted((x_prev, x_next))), k

        x_prev = x
        x = x_next

    raise RuntimeError("Метод Свенна: перевищено ліміт ітерацій.")


oracle1 = Oracle1D(objective_sigma)

interval, swann_iters = swann(
    oracle1,
    X0_1D,
    DELTA0_1D
)

print("\n" + "=" * 65)
print("ЛАБОРАТОРНА РОБОТА 1 — МЕТОД СВЕННА")
print("=" * 65)
print(f"Початкове наближення x0 = {X0_1D}")
print(f"Початковий крок delta0 = {DELTA0_1D}")
print(f"Знайдений інтервал: [{interval[0]:.4f}; {interval[1]:.4f}]")
print(f"Кількість ітерацій: {swann_iters}")
print(f"Кількість обчислень функції: {oracle1.n_f}")


# ============================================================
# ЛАБОРАТОРНА РОБОТА 2
# МЕТОД ДИХОТОМІЇ ТА МЕТОД ЗОЛОТОГО ПЕРЕРІЗУ
# ============================================================

def dichotomy(oracle, a, b, eps, max_iter=500):
    """Метод дихотомії."""

    iteration = 0
    delta = eps / 4

    while b - a > eps and iteration < max_iter:
        iteration += 1

        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        f1 = oracle.value(x1)
        f2 = oracle.value(x2)

        if f1 <= f2:
            b = x2
        else:
            a = x1

    return (a + b) / 2, iteration


def golden(oracle, a, b, eps, max_iter=500):
    """Метод золотого перерізу."""

    tau = (math.sqrt(5) - 1) / 2

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

    return (a + b) / 2, iteration


oracle_d = Oracle1D(objective_sigma)
sigma_d, it_d = dichotomy(
    oracle_d,
    interval[0],
    interval[1],
    EPS_1D
)

oracle_g = Oracle1D(objective_sigma)
sigma_g, it_g = golden(
    oracle_g,
    interval[0],
    interval[1],
    EPS_1D
)

restored2 = gaussian_blur(noisy, sigma_g)

print("\n" + "=" * 65)
print("ЛАБОРАТОРНА РОБОТА 2 — ОДНОВИМІРНА ОПТИМІЗАЦІЯ")
print("=" * 65)
print(f"Дихотомія: sigma = {sigma_d:.6f}")
print(f"Ітерацій: {it_d}")
print(f"Обчислень функції: {oracle_d.n_f}")
print()
print(f"Золотий переріз: sigma = {sigma_g:.6f}")
print(f"Ітерацій: {it_g}")
print(f"Обчислень функції: {oracle_g.n_f}")
print(f"PSNR після гаусівського фільтра: {psnr(restored2, clean):.4f} дБ")


# ============================================================
# ЛАБОРАТОРНА РОБОТА 3
# БІЛАТЕРАЛЬНИЙ ФІЛЬТР, НЕЛДЕР—МІД ТА ПАУЕЛЛ
# ============================================================

LO = np.array([0.3, 0.01])
HI = np.array([5.0, 1.0])


def bilateral_filter(img, sigma_spatial, sigma_range):
    """Білатеральна фільтрація зображення."""

    ss = max(float(sigma_spatial), 1e-3)
    sr = max(float(sigma_range), 1e-3)

    radius = int(min(max(round(2 * ss), 1), 5))

    padded = np.pad(
        img,
        radius,
        mode="reflect"
    )

    numerator = np.zeros_like(img)
    denominator = np.zeros_like(img)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):

            shifted = padded[
                radius + dy:radius + dy + img.shape[0],
                radius + dx:radius + dx + img.shape[1]
            ]

            # Просторова вага
            spatial_weight = math.exp(
                -(dx * dx + dy * dy) / (2 * ss * ss)
            )

            # Вага за різницею інтенсивності
            range_weight = np.exp(
                -((shifted - img) ** 2) / (2 * sr * sr)
            )

            weight = spatial_weight * range_weight

            numerator += weight * shifted
            denominator += weight

    return numerator / denominator


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

    return mse(filtered, clean) + 1000 * float(gap @ gap)


class Oracle2D:
    """Цільова функція для двовимірної оптимізації."""

    def __init__(self, fn):
        self.fn = fn
        self.n_f = 0

    def value(self, x):
        self.n_f += 1
        return self.fn(np.asarray(x, dtype=float))


def nelder_mead(oracle, x0, step, tol=1e-7, max_iter=200):
    """Метод деформованого багатогранника Нелдера—Міда."""

    x0 = np.asarray(x0, dtype=float)

    simplex = [
        x0,
        np.clip(x0 + [step, 0], LO, HI),
        np.clip(x0 + [0, 0.4 * step], LO, HI)
    ]

    values = [oracle.value(point) for point in simplex]

    for k in range(1, max_iter + 1):

        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        best, middle, worst = simplex
        f_best, f_middle, f_worst = values

        # Центр ваги двох найкращих вершин
        center = (best + middle) / 2

        # Відбиття
        reflected = np.clip(
            center + (center - worst),
            LO,
            HI
        )

        f_reflected = oracle.value(reflected)

        if f_reflected < f_best:

            # Розтягування
            expanded = np.clip(
                center + 2 * (reflected - center),
                LO,
                HI
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
            # Стиснення
            if f_reflected < f_worst:
                point = reflected
            else:
                point = worst

            contracted = np.clip(
                center + 0.5 * (point - center),
                LO,
                HI
            )

            f_contracted = oracle.value(contracted)

            if f_contracted < min(f_reflected, f_worst):
                simplex[2] = contracted
                values[2] = f_contracted

            else:
                # Зменшення симплекса
                for i in (1, 2):
                    simplex[i] = np.clip(
                        best + 0.5 * (simplex[i] - best),
                        LO,
                        HI
                    )

                    values[i] = oracle.value(simplex[i])

        best_index = int(np.argmin(values))
        best_point = simplex[best_index]

        size = max(
            np.linalg.norm(point - best_point)
            for point in simplex
        )

        if size <= tol and max(values) - min(values) <= tol:
            break

    best_index = int(np.argmin(values))

    return simplex[best_index], values[best_index], k


def line_minimize(oracle, x, direction, tol=1e-5):
    """Одновимірна мінімізація вздовж заданого напрямку."""

    a, b = -1.0, 1.0
    tau = (math.sqrt(5) - 1) / 2

    c = b - tau * (b - a)
    d = a + tau * (b - a)

    fc = oracle.value(x + c * direction)
    fd = oracle.value(x + d * direction)

    for _ in range(60):

        if b - a < tol:
            break

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - tau * (b - a)
            fc = oracle.value(x + c * direction)

        else:
            a, c, fc = c, d, fd
            d = a + tau * (b - a)
            fd = oracle.value(x + d * direction)

    alpha = (a + b) / 2

    return np.clip(x + alpha * direction, LO, HI)


def powell(oracle, x0, tol=1e-5, max_cycles=40):
    """Метод Пауелла."""

    x = np.asarray(x0, dtype=float)

    directions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0])
    ]

    for cycle in range(1, max_cycles + 1):

        start = x.copy()

        # Мінімізація вздовж поточних напрямків
        for direction in directions:
            x = line_minimize(oracle, x, direction)

        displacement = x - start

        if np.linalg.norm(displacement) < tol:
            break

        # Додатковий пошук у напрямку переміщення
        x = line_minimize(oracle, x, displacement)

        directions = [
            directions[1],
            displacement / (np.linalg.norm(displacement) + 1e-15)
        ]

    return x, oracle.value(x), cycle


# Початкові параметри білатерального фільтра
sigma_s0 = min(max(2 * sigma_g, 0.5), 4.0)
sigma_r0 = min(max(0.15 + 0.1 * SIGMA_NOISE, 0.02), 0.9)

# Нелдер—Мід
oracle_nm = Oracle2D(objective2)

theta_nm, f_nm, it_nm = nelder_mead(
    oracle_nm,
    [sigma_s0, sigma_r0],
    SIMPLEX_STEP
)

# Пауелл
oracle_pw = Oracle2D(objective2)

theta_pw, f_pw, cycles_pw = powell(
    oracle_pw,
    [sigma_s0, sigma_r0]
)

restored3 = bilateral_filter(
    noisy,
    theta_nm[0],
    theta_nm[1]
)

print("\n" + "=" * 65)
print("ЛАБОРАТОРНА РОБОТА 3 — БІЛАТЕРАЛЬНИЙ ФІЛЬТР")
print("=" * 65)
print("Метод Нелдера—Міда:")
print(f"sigma spatial = {theta_nm[0]:.6f}")
print(f"sigma range = {theta_nm[1]:.6f}")
print(f"Значення цільової функції = {f_nm:.8f}")
print(f"Кількість ітерацій = {it_nm}")
print(f"Обчислень функції = {oracle_nm.n_f}")
print(f"PSNR = {psnr(restored3, clean):.4f} дБ")

print("\nМетод Пауелла:")
print(f"sigma spatial = {theta_pw[0]:.6f}")
print(f"sigma range = {theta_pw[1]:.6f}")
print(f"Значення цільової функції = {f_pw:.8f}")
print(f"Кількість циклів = {cycles_pw}")
print(f"Обчислень функції = {oracle_pw.n_f}")


# ============================================================
# ЛАБОРАТОРНА РОБОТА 4
# ГРАДІЄНТНІ МЕТОДИ РЕГУЛЯРИЗАЦІЇ
# ============================================================

def grad_xy(u):
    """Обчислення дискретного градієнта."""

    gx = np.zeros_like(u)
    gy = np.zeros_like(u)

    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]

    return gx, gy


def divergence(px, py):
    """Обчислення дискретної дивергенції."""

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
    """Енергетична функція регуляризації."""

    gx, gy = grad_xy(image)

    data_term = 0.5 * float(np.sum((image - target) ** 2))
    regularization_term = 0.5 * lam * float(
        np.sum(gx * gx + gy * gy)
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
    tol=1e-6
):
    """Градієнтні методи: звичайний, важка кулька, Нестеров, Арміхо."""

    L = 1 + 8 * lam
    h = 1 / L

    image = target.copy()
    previous = image.copy()
    rows = []

    for k in range(1, max_iter + 1):

        if kind == "nag":
            # Метод Нестерова
            momentum = (
                (k - 1) / (k + 2)
                if beta is None
                else beta
            )

            y = image + momentum * (image - previous)
            g = energy_grad(y, target, lam)
            new_image = y - h * g

        else:
            g = energy_grad(image, target, lam)

            if kind == "hb":
                # Метод важкої кульки
                momentum = 0.9 if beta is None else beta

                new_image = (
                    image
                    - h * g
                    + momentum * (image - previous)
                )

            elif kind == "armijo":
                # Градієнтний спуск з правилом Арміхо
                f0 = energy(image, target, lam)
                step = h
                g2 = float(np.sum(g * g))

                for _ in range(40):
                    candidate = image - step * g

                    if energy(candidate, target, lam) <= (
                        f0 - 1e-4 * step * g2
                    ):
                        break

                    step *= 0.5

                new_image = candidate

            else:
                # Звичайний градієнтний спуск
                new_image = image - h * g

        gradient_norm = float(np.linalg.norm(g))

        rows.append((
            k,
            energy(image, target, lam),
            gradient_norm
        ))

        previous, image = image, new_image

        if gradient_norm <= tol:
            break

    return image, len(rows), rows


# Перевірка спряженості дискретних операторів
rng_check = np.random.default_rng(123)

u = rng_check.normal(size=(16, 16))
px = rng_check.normal(size=u.shape)
py = rng_check.normal(size=u.shape)

gx, gy = grad_xy(u)

adjoint_error = abs(
    float(
        np.sum(gx * px + gy * py)
        + np.sum(u * divergence(px, py))
    )
)

# Обчислення відновлених зображень
lam = LAMBDA0

restored4_gd, n_gd, history_gd = gradient_method(
    noisy, lam, kind="gd"
)

restored4_armijo, n_armijo, history_armijo = gradient_method(
    noisy, lam, kind="armijo"
)

restored4_hb, n_hb, history_hb = gradient_method(
    noisy, lam, kind="hb"
)

restored4_nag, n_nag, history_nag = gradient_method(
    noisy, lam, kind="nag"
)

restored4 = restored4_nag

print("\n" + "=" * 65)
print("ЛАБОРАТОРНА РОБОТА 4 — ГРАДІЄНТНІ МЕТОДИ")
print("=" * 65)
print(f"Параметр регуляризації lambda = {lam}")
print(f"Похибка перевірки спряженості операторів: {adjoint_error:.3e}")

print("\nЗвичайний градієнтний спуск:")
print(f"Ітерацій: {n_gd}")
print(f"PSNR: {psnr(restored4_gd, clean):.4f} дБ")

print("\nГрадієнтний спуск з правилом Арміхо:")
print(f"Ітерацій: {n_armijo}")
print(f"PSNR: {psnr(restored4_armijo, clean):.4f} дБ")

print("\nМетод важкої кульки:")
print(f"Ітерацій: {n_hb}")
print(f"PSNR: {psnr(restored4_hb, clean):.4f} дБ")

print("\nМетод Нестерова:")
print(f"Ітерацій: {n_nag}")
print(f"PSNR: {psnr(restored4_nag, clean):.4f} дБ")


# ============================================================
# ЛАБОРАТОРНА РОБОТА 5
# МЕТОД ПРОЄКЦІЇ ТА ШТРАФНИЙ МЕТОД
# ============================================================

def project(image, lower=0.0, upper=1.0):
    """Проєкція значень пікселів на проміжок [0, 1]."""

    return np.clip(image, lower, upper)


def violation(image):
    """Максимальне порушення обмежень."""

    below = np.maximum(0.0, -image)
    above = np.maximum(0.0, image - 1.0)

    max_violation = float(
        max(below.max(), above.max())
    )

    outside_count = int(
        np.count_nonzero((image < 0) | (image > 1))
    )

    return max_violation, outside_count


def projected_gradient(
    target,
    lam,
    max_iter=200,
    tol=1e-6,
    accelerated=False
):
    """Проєкційний градієнтний спуск та прискорений варіант."""

    L = 1 + 8 * lam
    h = 1 / L

    image = project(target.copy())
    previous = image.copy()
    y = image.copy()

    for k in range(1, max_iter + 1):

        point = y if accelerated else image

        g = energy_grad(point, target, lam)
        new_image = project(point - h * g)

        gradient_mapping = float(
            np.linalg.norm(new_image - point) / h
        )

        if accelerated:
            beta = (k - 1) / (k + 2)
            y = new_image + beta * (new_image - image)
            previous, image = image, new_image
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
        -2 * np.maximum(0.0, -image)
        + 2 * np.maximum(0.0, image - 1.0)
    )


def penalty_method(target, lam, outer=7, inner=100):
    """Штрафний метод з поступовим збільшенням штрафу."""

    image = target.copy()
    r = 1.0
    rows = []

    for s in range(outer):

        h = 1 / (1 + 8 * lam + 2 * r)
        previous = image.copy()

        for k in range(1, inner + 1):

            beta = (k - 1) / (k + 2)
            y = image + beta * (image - previous)

            g = (
                energy_grad(y, target, lam)
                + r * penalty_grad(y)
            )

            new_image = y - h * g

            previous, image = image, new_image

            if np.linalg.norm(g) <= 1e-6:
                break

        max_v, count = violation(image)
        p = penalty_value(image)
        energy_value = energy(image, target, lam)

        rows.append((
            s,
            r,
            energy_value,
            p,
            r * p,
            max_v,
            count
        ))

        if r * p <= 1e-8:
            break

        r *= 10

    return image, rows


# Виконання методів лабораторної роботи 5
restored5_proj, it_proj, gmap = projected_gradient(
    raw,
    lam,
    accelerated=False
)

restored5_fista, it_fista, gmap_f = projected_gradient(
    raw,
    lam,
    accelerated=True
)

restored5_pen, penalty_rows = penalty_method(
    raw,
    lam
)

viol_proj = violation(restored5_proj)
viol_fista = violation(restored5_fista)
viol_pen = violation(restored5_pen)

print("\n" + "=" * 65)
print("ЛАБОРАТОРНА РОБОТА 5 — МЕТОДИ З ОБМЕЖЕННЯМИ")
print("=" * 65)

print("\nПроєкційний градієнтний метод:")
print(f"Ітерацій: {it_proj}")
print(f"Максимальне порушення: {viol_proj[0]:.6g}")
print(f"Кількість пікселів поза [0, 1]: {viol_proj[1]}")
print(f"PSNR: {psnr(restored5_proj, clean):.4f} дБ")

print("\nПрискорений проєкційний метод:")
print(f"Ітерацій: {it_fista}")
print(f"Максимальне порушення: {viol_fista[0]:.6g}")
print(f"Кількість пікселів поза [0, 1]: {viol_fista[1]}")
print(f"PSNR: {psnr(restored5_fista, clean):.4f} дБ")

print("\nШтрафний метод:")
print(f"Кількість зовнішніх ітерацій: {len(penalty_rows)}")
print(f"Максимальне порушення: {viol_pen[0]:.6g}")
print(f"Кількість пікселів поза [0, 1]: {viol_pen[1]}")
print(f"PSNR: {psnr(restored5_pen, clean):.4f} дБ")

print("\nТаблиця штрафного методу:")
print("s | r | E | P | rP | max порушення | поза [0, 1]")

for row in penalty_rows:
    print(
        "%d | %.0f | %.6f | %.3e | %.3e | %.3e | %d"
        % row
    )


# ============================================================
# ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
# ============================================================

images = [
    (clean, "Еталон"),
    (noisy, "Шумне зображення"),
    (restored2, "ЛР2: гаусівський фільтр"),
    (restored3, "ЛР3: білатеральний фільтр"),
    (restored4_gd, "ЛР4: градієнтний спуск"),
    (restored4_nag, "ЛР4: метод Нестерова"),
    (restored5_proj, "ЛР5: проєкційний метод"),
    (restored5_pen, "ЛР5: штрафний метод")
]

fig, axes = plt.subplots(2, 4, figsize=(15, 8))

for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"{title}\nPSNR = {psnr(img, clean):.2f} дБ"
    )
    ax.axis("off")

fig.tight_layout()


# ============================================================
# ПОРІВНЯННЯ ЯКОСТІ ВІДНОВЛЕННЯ
# ============================================================

labels = [
    "Шумне",
    "ЛР2",
    "ЛР3 НМ",
    "ЛР4 ГС",
    "ЛР4 Нестеров",
    "ЛР5 проєкція",
    "ЛР5 штраф"
]

values = [
    psnr(noisy, clean),
    psnr(restored2, clean),
    psnr(restored3, clean),
    psnr(restored4_gd, clean),
    psnr(restored4_nag, clean),
    psnr(restored5_proj, clean),
    psnr(restored5_pen, clean)
]

fig2, ax = plt.subplots(figsize=(11, 5))

ax.bar(labels, values)
ax.set_ylabel("PSNR, дБ")
ax.set_title("Порівняння якості відновлення зображення")
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=20)

fig2.tight_layout()

plt.show()

print("\nВиконання програми завершено.")