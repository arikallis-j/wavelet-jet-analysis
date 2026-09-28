import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe

YELLOW = '#ffff99'
GREEN = '#7fc87f'
BLUE = '#385eb1'

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ============================================================
# Левая панель: изотропный спектр E(k) — как на твоей картинке
# ============================================================
ax = axes[0]

k = np.logspace(-1, 1.5, 1000)
k_f = 1.4
k_nu = 20.0

# Плавный переход между -5/3 и -3 через сигмоиду в лог-пространстве
def smooth_spectrum(k, k_f, k_nu, width=0.1):
    log_k = np.log10(k)
    log_kf = np.log10(k_f)
    log_knu = np.log10(k_nu)
    
    # Сигмоида для перехода между двумя степенными законами
    sigma = 1 / (1 + np.exp(-(log_k - log_kf) / width))
    
    E_inverse = k**(-5/3)
    E_forward = k**(-3) * k_f**(-5/3 + 3)  # нормировка чтобы стыковались
    
    E = (1 - sigma) * E_inverse + sigma * E_forward
    
    # Экспоненциальный спад после k_nu
    E *= np.exp(-((k / k_nu)**5))
    
    return E

E = smooth_spectrum(k, k_f, k_nu)
ax.loglog(k, E, 'b-', lw=2, color=BLUE)

# Вертикальная линия форсинга
ax.axvline(k_f, color='gray', ls='--', lw=1)
ax.axvline(k_nu, color='gray', ls='--', lw=1)
ax.text(k_f * 1.1, 1e-2, r'$k_f$', fontsize=12)

# Подписи степенных законов
ax.text(0.2, 0.05, r'$\varepsilon^{2/3}k^{-5/3}$', fontsize=12, color='black')
ax.text(2.0, 1e-3, r'$\eta^{2/3}k^{-3}$', fontsize=12, color='black')

# Стрелки переноса энергии — ax.annotate
ax.annotate('', xy=(0.2, 0.02), xytext=(0.5, 0.02),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.25, 0.025, r'$\varepsilon$', color='red', fontsize=12)

ax.annotate('', xy=(3.0, 5e-4), xytext=(1.5, 5e-4),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(2.0, 6e-4, r'$\eta$', color='red', fontsize=12)

ax.set_xlabel('Wavenumber', fontsize=12)
ax.set_ylabel('Energy', fontsize=12)
ax.set_title('Isotropic spectrum', fontsize=13)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.minorticks_off()
ax.set_ylim(1e-10, 1e2)

for spine in ax.spines.values():
    spine.set_visible(False)

# И рисуешь оси вручную как стрелки
ax.annotate('', xy=(1, 0), xytext=(0, 0),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-', color='black', lw=1.5))
ax.annotate('', xy=(0, 1), xytext=(0, 0),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='-', color='black', lw=1.5))

# ============================================================
# Правая панель: 2D спектр с гантелькой на (kx, ky)
# ============================================================
ax = axes[1]

# Параметры для гантельки — условные
eps = 1.0
beta = 1.0
# Граница гантельки: beta * |kx| / k^2 = eps^(1/3) * k^(2/3)
# => |kx| = eps^(1/3) * k^(8/3) / beta
# Параметризуем через угол phi
phi = np.linspace(0, 2*np.pi, 1000)
k_vals = np.linspace(0.1, 5, 500)

# Строим границу гантельки в (kx, ky)
# из условия: beta * kx / k^2 = eps^(1/3) * k^(2/3)
# kx = eps^(1/3) * k^(8/3) / beta, k = sqrt(kx^2 + ky^2)
# Проще всего: для каждого угла phi найти k на границе
# beta * cos(phi) / k = eps^(1/3) * k^(2/3)
# => k^(5/3) = beta * cos(phi) / eps^(1/3)
# => k = (beta * |cos(phi)| / eps^(1/3))^(3/5)

phi_plot = np.linspace(-np.pi/2, np.pi/2, 1000)
k_boundary = 1.0

kx_boundary = 1.2 * k_boundary * np.cos(phi_plot)
ky_boundary = 1.2 * k_boundary * np.sin(phi_plot)
# Симметрия: отражаем на kx < 0
kx_full = np.concatenate([kx_boundary, -kx_boundary[::-1]])
ky_full = np.concatenate([ky_boundary, ky_boundary[::-1]])

# Заливка внутренней области (нелинейный режим)
# ax.fill(kx_full, ky_full, color='lightblue', alpha=0.5, label='Linear (wave)')

# Заливка внешней области — через большой прямоугольник минус внутренность
# Проще: просто закрасить фон и поверх нарисовать внутренность
ax.set_facecolor('white')  # внешняя область = волновой режим
ax.fill(kx_full, ky_full, color='lightgreen', alpha=0.8)

# Граница гантельки
ax.plot(kx_full, ky_full, 'k-', lw=1.5, color=GREEN)



kx_boundary = 1.1 * k_boundary * np.cos(phi_plot)
ky_boundary = 1.1 * k_boundary * np.sin(phi_plot)
# Симметрия: отражаем на kx < 0
kx_full = np.concatenate([kx_boundary, -kx_boundary[::-1]])
ky_full = np.concatenate([ky_boundary, ky_boundary[::-1]])

# Заливка внутренней области (нелинейный режим)
# ax.fill(kx_full, ky_full, color='lightblue', alpha=0.5, label='Linear (wave)')

# Заливка внешней области — через большой прямоугольник минус внутренность
# Проще: просто закрасить фон и поверх нарисовать внутренность
ax.set_facecolor('white')  # внешняя область = волновой режим
ax.fill(kx_full, ky_full, color='white')
ax.fill(kx_full, ky_full, color='lightblue', alpha=0.6, label='Nonlinear (turbulence)')


# Граница гантельки
ax.plot(kx_full, ky_full, 'k-', lw=1.5, color=BLUE)


kx_boundary = k_boundary * (np.cos(phi_plot))**(8/5)
ky_boundary = k_boundary * np.sin(phi_plot) * (np.cos(phi_plot))**(3/5)


# Симметрия: отражаем на kx < 0
kx_full = np.concatenate([kx_boundary, -kx_boundary[::-1]])
ky_full = np.concatenate([ky_boundary, ky_boundary[::-1]])

# Заливка внутренней области (нелинейный режим)
ax.fill(kx_full, ky_full, color='white', alpha=0.5, label='Nothing')

# Заливка внешней области — через большой прямоугольник минус внутренность
# Проще: просто закрасить фон и поверх нарисовать внутренность
ax.set_facecolor('white')  # внешняя область = волновой режим
ax.fill(kx_full, ky_full, color='white', alpha=0.8)

# Граница гантельки
ax.plot(kx_full, ky_full, 'k-', lw=1.5, color='black')


kx_boundary = 0.4 * k_boundary * np.cos(phi_plot)
ky_boundary = 0.4 * k_boundary * np.sin(phi_plot)
# Симметрия: отражаем на kx < 0
kx_full = np.concatenate([kx_boundary, -kx_boundary[::-1]])
ky_full = np.concatenate([ky_boundary, ky_boundary[::-1]])

# Заливка внутренней области (нелинейный режим)
# ax.fill(kx_full, ky_full, color='lightblue', alpha=0.5, label='Linear (wave)')

# Заливка внешней области — через большой прямоугольник минус внутренность
# Проще: просто закрасить фон и поверх нарисовать внутренность
ax.set_facecolor('white')  # внешняя область = волновой режим
ax.fill(kx_full, ky_full, color='pink', alpha=0.8, label='Linear (wave)')

# Граница гантельки
ax.plot(kx_full, ky_full, 'k--', lw=1.5, color='red')


kx_boundary = k_boundary * (np.cos(phi_plot))**(8/5)
ky_boundary = k_boundary * np.sin(phi_plot) * (np.cos(phi_plot))**(3/5)


# Симметрия: отражаем на kx < 0
kx_full = np.concatenate([kx_boundary, -kx_boundary[::-1]])
ky_full = np.concatenate([ky_boundary, ky_boundary[::-1]])

# Заливка внутренней области (нелинейный режим)

ax.fill(kx_full, ky_full, color='white', alpha=1.0)



# # Накопление энергии у kx ~ 0 — штриховка вертикальной полосы
# ax.fill_betweenx(np.linspace(-5, 5, 100), -0.3, 0.3,
#                  alpha=0.3, color='red', hatch='///',
#                  label='Energy accumulation')

# Оси и подписи
ax.axhline(0, color='k', lw=0.5)
ax.axvline(0, color='k', lw=0.5)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r'$k_x$', fontsize=12)
ax.set_ylabel(r'$k_y$', fontsize=12)
# ax.set_title(r'2D spectrum with $\beta$-effect', fontsize=13)

# Стрелка — направление переноса энергии к kx~0
ax.annotate('', xy=(0.1, 2.0), xytext=(2.0, 2.0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.8, 2.3, r'energy', color='red', fontsize=11)

# ax.legend(loc='upper right', fontsize=10)
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

ax.minorticks_off()

# plt.tight_layout()
plt.savefig('schematic.svg')  # потом доделываешь в Inkscape
plt.show()