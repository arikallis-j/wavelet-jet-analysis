## 1. Да, Uchida et al. 2023 действительно использовали `pyqg`

Причём это важно уточнить: в статье использовался **двухслойный QG `pyqg`**, а не специально однослойный barotropic `BTModel`.

Авторы прямо пишут, что QG simulation была рассчитана с помощью **pseudo-spectral `pyqg`**, а затем:

* Fourier analysis выполнялся через `xrft`;
* wavelet transforms — через `xwavelet`;
* Jupyter notebooks для запуска `pyqg` и анализа опубликованы отдельно. ([Wiley Online Library][1])

В их конкретной постановке:

$$
\beta=0,
$$

была **двухслойная QG-система** с ($R_d=100$) km, stochastic ring forcing и bottom drag; симуляция использовалась как контролируемый тест для сравнения wavelet- и Fourier-методов. ([Wiley Online Library][1])

Это значит, что есть очень хорошая возможность буквально построить **первый этап твоего проекта поверх той же экосистемы**, на которой была продемонстрирована методология Uchida:

$
\boxed{\text{pyqg} \rightarrow \text{xrft} \rightarrow \text{xwavelet}}
$

а уже затем:

$
\boxed{\text{Oceananigans} \rightarrow \text{тот же анализ}}
$

---

# 2. И да: ты правильно поняла смысл предыдущего плана

По сути, большая часть этапов 0–6 в первоначальной версии была **не исследованием струй как таковых, а верификацией того, что твоя реализация QG через Oceananigans действительно реализует нужную физику и выдаёт данные пригодного качества**.

И здесь действительно возникает естественный вопрос:

> Зачем сначала строить сложную validation architecture вокруг Oceananigans, если у нас есть уже готовая QG-модель, на которой можно сначала отладить всю методологическую цепочку?

Ответ: **незачем**.

Гораздо рациональнее сначала сделать полностью работающий reference pipeline:

$$
\boxed{
\texttt{pyqg}
\rightarrow
\texttt{NetCDF/xarray}
\rightarrow
\texttt{xrft}
\rightarrow
\texttt{xwavelet}
\rightarrow
\text{jets / spectra / diagnostics}
}
$$

И только после этого:

$$
\boxed{
\texttt{Oceananigans}
\rightarrow
\text{те же output}
\rightarrow
\text{тот же Python pipeline}
}
$$

Тогда ты разделяешь две совершенно разные проблемы.

### Проблема A — методология исследования

> Умею ли я корректно провести симуляцию → Fourier → wavelet → jet analysis?

Её решает `pyqg`.

### Проблема B — численная реализация

> Воспроизводит ли Oceananigans ту же физику?

Её решает сравнение Oceananigans с уже работающим reference pipeline.

Это **гораздо чище с точки зрения научной верификации**.

---

# 3. Я бы теперь полностью перестроила план

Получается две большие фазы:

```text
PHASE I — Reference QG methodology
        pyqg
          ↓
        xrft
          ↓
      xwavelet
          ↓
      jet diagnostics
          ↓
   validated analysis pipeline

PHASE II — Oceananigans implementation
        Oceananigans
             ↓
       same diagnostics
             ↓
      comparison with pyqg
             ↓
      validated QG engine
             ↓
       production study
```

И вот уже после этого имеет смысл заниматься физической задачей о зональных струях.

---

# 4. Новый полный план

## Фаза I. Reference implementation на `pyqg`

### 1. Математическая спецификация reference experiment

**Цель:** определить точную QG-задачу, которую сначала решает `pyqg`, а потом должна воспроизвести Oceananigans.

**Задачи:**

* Зафиксировать целевое уравнение:
  $$
  \partial_t q+J(\psi,q)+\beta\psi_x = -rq+\mathcal F+\mathcal D.
  $$
* Определить, нужна ли нам именно **однослойная barotropic QG**:
  $$
  q=\nabla^2\psi.
  $$
* Зафиксировать:
  $$
  L,\ N,\ \beta,\ r,\ k_f,\Delta k_f,A_f.
  $$
* Определить expected regime:

  * forcing scale;
  * Rhines scale;
  * drag scale;
  * dissipation scale.
* Зафиксировать random seed и правила статистического усреднения.

**Результат:**
Полностью формализованная reference-конфигурация.

---

## 2. Запуск минимальной `pyqg`-симуляции

**Цель:** убедиться, что reference solver вообще работает в нашей постановке.

**Задачи:**

* Создать `pyqg.BTModel`.
* Задать periodic domain.
* Включить β.
* Добавить linear drag.
* Настроить stochastic narrow-band forcing.
* Запустить короткий spin-up.

**Результат:**
Рабочий QG dataset.

**Критерий:**
Энергия и enstrophy конечны, нет blow-up, forcing/drag работают.

---

## 3. Проверка физики reference QG

**Цель:** убедиться, что `pyqg` даёт именно тот режим, который нам нужен.

**Задачи:**

* Проверить forcing spectrum.
* Проверить ($E(k)$) и ($Z(k)$).
* Проверить energy/enstrophy balance.
* Оценить ($L_\beta$).
* Посмотреть появление anisotropy / jets при выбранной β.
* Проверить влияние ($r$).

**Результат:**
Получена физически понятная reference simulation.

---

# Фаза II. Reference analysis pipeline

## 4. NetCDF → xarray

**Цель:** полностью отделить simulation от анализа.

**Задачи:**

* Читать output `pyqg` через `xarray`.
* Нормализовать координаты и units.
* Сделать стандартный dataset schema.

Например концептуально:

$
(x,y,t,u,v,\omega,\psi).
$

**Результат:**
Любой QG output приводится к единому `xarray.Dataset`.

---

## 5. Fourier pipeline через `xrft`

**Цель:** получить полностью рабочую спектральную диагностику.

**Задачи:**

* 2D Fourier transform;
* ($E(k_x,k_y)$);
* ($Z(k_x,k_y)$);
* radial averaging;
* ($E(k)$);
* ($Z(k)$);
* Parseval check;
* spectral evolution in time.

**Результат:**
Validated Fourier pipeline.

---

## 6. Jet diagnostics

**Цель:** научиться автоматически определять зональные струи.

**Задачи:**

* $$
  U(y,t)=\langle u\rangle_x;
  $$
* визуализация ($U(y)$);
* определение jet spacing;
* число струй;
* comparison with ($L_\beta$);
* time-averaging.

**Результат:**
Рабочая количественная jet diagnostic.

---

## 7. Wavelet pipeline через `xwavelet`

**Цель:** воспроизвести методологию Uchida.

**Задачи:**

* Запустить `xwavelet` на тех же данных.
* Построить wavelet-based wavenumber spectra.
* Сопоставить их с Fourier spectra.
* Исследовать spatially localized anisotropy.
* Повторить ключевые diagnostics из Uchida.

Это особенно удобно, потому что в статье сами авторы сообщают, что использовали именно `xwavelet` для wavelet transforms и `xrft` для Fourier transforms. ([Wiley Online Library][1])

**Результат:**
Полностью рабочий pipeline

$$
\text{QG simulation}
\rightarrow
\text{Fourier}
\rightarrow
\text{Wavelet}.
$$

---

## 8. Validation against Uchida

**Цель:** удостовериться, что наш pipeline не просто «работает», а методологически совпадает с опубликованным.

**Задачи:**

* Взять их опубликованный пример/конфигурацию либо максимально близкую постановку.
* Воспроизвести:
  [
  E(k), Z(k)
  ]
  и wavelet spectra.
* Сравнить характерные slopes и scale ranges.
* Проверить agreement Fourier ↔ wavelet.

Статья прямо описывает, что wavelet spectra хорошо согласовывались с canonical Fourier spectra; одновременно wavelets дают локальную информацию об anisotropy. ([Wiley Online Library][1])

**Результат:**
Можно уверенно сказать:

> «Наш Python analysis pipeline воспроизводит опубликованную методологию Uchida».

На этом **Фаза I заканчивается**.

---

# Фаза III. Перенос dynamics на Oceananigans

Вот только теперь появляется твоя текущая техническая работа с Julia.

## 9. Минимальная Oceananigans QG-like модель

**Цель:** заставить Oceananigans воспроизвести reference `pyqg`.

**Задачи:**

* `NonhydrostaticModel`;
* periodic (x,y);
* flat (z);
* Coriolis;
* β-plane;
* forcing;
* drag;
* выбранная advection scheme.

**Результат:**
Минимальная Oceananigans configuration с той же математической постановкой, что и reference.

---

## 10. Oceananigans output

**Цель:** заставить Oceananigans выдавать **тот же формат данных**, который уже умеет анализировать Python pipeline.

**Задачи:**

* NetCDFWriter;
* ($u$,$v$,$\omega$,$\psi$);
* coordinates;
* metadata;
* время.

**Результат:**

```text
pyqg.nc
Oceananigans.nc
```

оба читаются одной и той же Python-функцией.

Это, кстати, очень важный архитектурный момент: **не надо писать второй analysis pipeline для Julia**.

---

## 11. Oceananigans monitoring

**Цель:** обеспечить эксплуатацию модели.

**Задачи:**

* callback;
* CFL;
* ($E$);
* ($Z$);
* checkpoint;
* JLD2 при необходимости.

**Результат:**
Длительный запуск безопасен и контролируем.

---

# Фаза IV. Oceananigans ↔ pyqg validation

## 12. Геострофический/f-plane test

Проверяем primitive-equation side:

$$
f_0\rightarrow\text{geostrophic balance}.
$$

**Результат:**
Корректная rotational dynamics.

---

## 13. β-plane comparison

Одинаковые параметры:

$$
\beta,L,U,N
$$

в `pyqg` и Oceananigans.

Сравниваем:

* ($E(t)$);
* ($Z(t)$);
* ($E(k)$);
* ($Z(k)$);
* ($U(y)$);
* anisotropy.

**Результат:**
Oceananigans воспроизводит reference QG dynamics.

---

## 14. Forcing comparison

Одинаковая:

$
k_f,\Delta k_f,\epsilon_{\rm in}.
$

Проверяем forcing spectrum и статистическую injection rate.

**Результат:**
Forcing эквивалентен.

---

## 15. Drag comparison

Одинаковый ($r$).

Сравниваем:

$$
T_r,\quad E(k),\quad E(t).
$$

**Результат:**
Equivalent large-scale damping.

---

## 16. Advection/conservation comparison

Сравнить схемы Oceananigans с reference `pyqg`.

**Результат:**
Выбрана схема, которая лучше всего воспроизводит reference spectral dynamics.

После этого конфигурация Oceananigans **замораживается**.

---

# Фаза V. Production

## 17. Production QG simulations

Теперь впервые начинается собственно основная серия исследования.

**Задачи:**

* sweep по ($\beta$);
* sweep по forcing amplitude;
* sweep по ($r$);
* возможно, sweep по ($k_f$);
* достаточно длинные statistically stationary runs.

---

## 18. Joint Fourier + wavelet + jet analysis

Для каждого эксперимента:

$$
\boxed{
E(k_x,k_y)
\rightarrow
E(k)
\rightarrow
\text{wavelet spectra}
\rightarrow
U(y)
\rightarrow
\text{jet statistics}
}
$$

И уже здесь исследовать:

* возникновение jets;
* jet spacing;
* Rhines-scale relation;
* spectral anisotropy;
* localization of anisotropy;
* transfer between scales.

---

# 5. Получается очень важное изменение философии проекта

Теперь архитектура выглядит так:

```text
                 REFERENCE
                    │
                 pyqg
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
        xrft              xwavelet
          │                   │
          └─────────┬─────────┘
                    ↓
              Jet diagnostics
                    │
          VALIDATED METHODOLOGY
                    │
                    ↓
              Oceananigans
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
   Numerical validation   Same analysis
          │                   │
          └─────────┬─────────┘
                    ↓
          Oceananigans validated
                    │
                    ↓
             PRODUCTION STUDY
```

И это, на мой взгляд, **намного лучше первоначального плана**.

Самое главное — теперь мы не пытаемся одновременно решить три задачи:

1. научиться запускать QG;
2. научиться анализировать QG;
3. доказать корректность Oceananigans.

Сначала полностью решаем **(1)+(2)** на уже готовом QG solver, а затем используем результат как эталон для **(3)**.

Причём тут есть ещё один очень красивый момент: Uchida действительно оставили нам практически готовый methodological reference — `pyqg` + `xrft` + `xwavelet` + опубликованные notebooks. ([Wiley Online Library][1])

Так что я бы теперь считала **первой настоящей задачей проекта не Oceananigans, а воспроизведение reference experiment Uchida на `pyqg` и получение из него полностью автоматического Python pipeline**. После этого переход на Oceananigans становится уже не исследованием «а правильно ли я вообще что-то делаю?», а довольно чистым **numerical-model validation problem**.

[1]: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003399?utm_source=chatgpt.com "Wavelet‐Based Wavenumber Spectral Estimate of Eddy Kinetic Energy: Idealized Quasi‐Geostrophic Flow - Uchida - 2023 - Journal of Advances in Modeling Earth Systems - Wiley Online Library"
