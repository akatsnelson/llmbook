"""Энергия ускорителей из главы 15 «Устройство GPU»: ватты, джоули на токен, PUE и обучение.

Все числа — паспортные (мощность из спецификаций, GPU-часы из открытых отчётов), а не замер:
на Mac без sudo мощность не снять, а на чужом кластере тем более. Поэтому каждая оценка
по паспортной мощности — верхняя граница для карты, которая работает не на пределе.
Файл сам проверяет свои утверждения. Источники чисел — в сносках раздела 15.11
«Энергия и питание ускорителей» главы 15.

Запуск из корня репозитория: python labs/ch15/energy_budget.py
"""
from __future__ import annotations

GIB = 2**30
KWH = 3.6e6  # джоулей в киловатт-часе


# --- 1. Мощность и пик на ватт ----------------------------------------------------------
# (карта, мощность в Вт по спецификации, плотный пик BF16 в Тфлоп/с из таблицы ускорителей)
CARDS = [
    ("GeForce RTX 4090", 450, 165),
    ("GeForce RTX 5090", 575, 210),
    ("A100 SXM 80 ГБ", 400, 312),
    ("H100 SXM", 700, 989),
    ("H200 SXM", 700, 989),
    ("B200 (HGX)", 1000, 2250),
    ("B300 (GB300)", 1400, 2500),
    ("AMD Instinct MI355X", 1400, 2500),
    ("GB10 (DGX Spark)", 140, 100),
]


def tflops_per_watt(watts: float, tflops: float) -> float:
    return tflops / watts


def check_power_table() -> None:
    printed = {"GeForce RTX 4090": 0.37, "GeForce RTX 5090": 0.37, "A100 SXM 80 ГБ": 0.78,
               "H100 SXM": 1.41, "H200 SXM": 1.41, "B200 (HGX)": 2.25, "B300 (GB300)": 1.79,
               "AMD Instinct MI355X": 1.79, "GB10 (DGX Spark)": 0.71}
    print("карта                 Вт   Тфлоп/с BF16 на Вт")
    for name, w, p in CARDS:
        r = tflops_per_watt(w, p)
        print(f"{name:<20}{w:5d}  {r:6.2f}")
        assert round(r, 2) == printed[name], (name, r)   # совпадает с таблицей книги
    # A100 -> B200: пик на ватт вырос почти втрое, а мощность карты — в 2,5 раза
    assert 2.8 < 2250 / 1000 / (312 / 400) < 2.95
    # B300 на 1100 Вт: в GB300 NVL72 Lenovo пик BF16 2500 Тфлоп/с, в HGX B300 — 2250, как у B200
    assert round(2500 / 1100, 2) == 2.27 and round(2250 / 1100, 2) == 2.05
    # пикоджоули на операцию при пике и на байт при полной полосе у H100: верхние границы
    pj_flop = 700 / 989e12 * 1e12
    pj_byte = 700 / 3.35e12 * 1e12
    print(f"H100 на пике: {pj_flop:.2f} пДж на операцию, {pj_byte:.0f} пДж на байт "
          f"({pj_byte / 8:.0f} пДж на бит)")
    assert abs(pj_flop - 0.71) < 0.01 and abs(pj_byte - 209) < 1


# --- 2. Сервер и стойка ------------------------------------------------------------------
def check_server_and_rack() -> None:
    dgx_h100, dgx_b200 = 10.2e3, 14.3e3            # Вт, максимум сервера по NVIDIA
    gpu_share_h100 = 8 * 700 / dgx_h100
    gpu_share_b200 = 8 * 1000 / dgx_b200
    print(f"доля GPU в мощности сервера: DGX H100 {gpu_share_h100:.0%}, "
          f"DGX B200 {gpu_share_b200:.0%}")
    assert 0.5 < gpu_share_h100 < 0.6 and 0.5 < gpu_share_b200 < 0.6
    for name, rack in (("GB200 NVL72", 120e3), ("GB300 NVL72", 135e3)):
        print(f"{name}: {rack / 1e3:.0f} кВт на стойку, {rack / 72:.0f} Вт на GPU со всем прочим")
    assert 1600 < 120e3 / 72 < 1700


# --- 3. Джоули на токен ------------------------------------------------------------------
def joules_per_token(watts: float, seconds: float, tokens: float) -> float:
    """Энергия = мощность x время; делим на число токенов, сделанных за это время."""
    return watts * seconds / tokens


def check_tokens() -> dict:
    # Глава 14: модель на 8 млрд параметров в Q4_K_M на H100 (3,35 ТБ/с), кэш 1 ГБ на диалог
    bw, w = 3.35e12, 700
    t1 = (4.56 + 1) * GIB / bw                     # один диалог: веса + свой кэш
    t16 = (4.56 + 16) * GIB / bw                   # 16 диалогов: веса один раз
    j1, j16 = joules_per_token(w, t1, 1), joules_per_token(w, t16, 16)
    j_in = joules_per_token(w, 0.27, 8192)         # обработка входа 8192 токенов, 0,27 с
    print(f"H100, шаг одного диалога {t1 * 1e3:.2f} мс: {j1:.2f} Дж на токен")
    print(f"H100, шаг 16 диалогов {t16 * 1e3:.2f} мс: {j16:.2f} Дж на токен")
    print(f"H100, вход 8192 токена за 0,27 с: {j_in:.3f} Дж на токен")
    assert abs(t1 * 1e3 - 1.78) < 0.01 and abs(t16 * 1e3 - 6.59) < 0.01
    assert 4.3 < j1 / j16 < 4.4                    # пакет в 16 дешевле в 4,3 раза
    assert j_in < j16 / 10                         # вход ещё на порядок дешевле
    # Перевод корпуса BookLM на GB10: около 3 историй в секунду по 760 токенов, DGX Spark до 240 Вт
    j_tr = joules_per_token(240, 1, 3 * 760)
    print(f"GB10, перевод корпуса: {j_tr:.3f} Дж на токен")
    assert 0.10 < j_tr < 0.11
    return {"j1": j1, "j16": j16}


# --- 4. Ограничение мощности: модель с явными допущениями -------------------------------
def capped(f: float, t_mem: float, t_comp: float, p_static: float, p_dyn: float):
    """Частота f в долях от номинальной.

    Допущения: динамическая мощность ~ f^3 (напряжение снижают вместе с частотой, P ~ C V^2 f),
    статическая не зависит от f; время — большее из чтения памяти (от частоты SM не
    зависит) и счёта (растёт как 1/f). Это учебная модель, а не замер.
    """
    t = max(t_mem, t_comp / f)
    busy = (t_comp / f) / t                        # доля времени, когда считают
    power = p_static + p_dyn * f**3 * busy
    return t, power, power * t


def check_power_capping() -> None:
    print("частота  генерация: время, энергия   вход: время, энергия")
    rows = []
    for f in (1.0, 0.8, 0.6):
        dec = capped(f, t_mem=1.0, t_comp=0.1, p_static=200, p_dyn=500)   # упор в память
        pre = capped(f, t_mem=0.1, t_comp=1.0, p_static=200, p_dyn=500)   # упор в счёт
        rows.append((dec, pre))
        print(f"{f:5.0%}    {dec[0]:6.2f} {dec[2]:7.0f}            {pre[0]:6.2f} {pre[2]:7.0f}")
    (d1, p1), _, (d6, p6) = rows
    assert d6[0] == d1[0]                          # генерация не замедлилась
    assert d6[2] < d1[2]                           # и стала дешевле
    assert p6[0] > 1.6 * p1[0]                     # вход замедлился в 1/0,6 раза
    # POLCA (A100, инференс): до 20 % мощности ценой не больше 7 % скорости — модель
    # согласуется по знаку: упор в память почти не чувствует частоты.


# --- 5. Дата-центр: PUE и гигаватт ------------------------------------------------------
def facility_energy(it_energy: float, pue: float) -> float:
    return it_energy * pue


def check_datacenter() -> None:
    for name, pue in (("Google, 2025", 1.09), ("средний по опросу Uptime, 2025", 1.54)):
        over = facility_energy(1.0, pue) - 1.0
        print(f"{name}: на 1 кВт·ч вычислений ещё {over:.2f} кВт·ч")
    racks = 1e9 / (1.2 * 135e3)                    # 1 ГВт площадки, PUE 1,2, стойки GB300 NVL72
    print(f"1 ГВт при PUE 1,2: {racks:,.0f} стоек NVL72, {racks * 72:,.0f} GPU")
    assert 6000 < racks < 6300


# --- 6. Энергия обучения -----------------------------------------------------------------
def training_energy_kwh(gpu_hours: float, watts: float, pue: float = 1.0) -> float:
    return gpu_hours * watts / 1e3 * pue


def check_training() -> None:
    # Llama 3.1, карточка модели: GPU-часы на H100-80GB с TDP 700 Вт
    llama = {"8B": 1.46e6, "70B": 7.0e6, "405B": 30.84e6}
    total = sum(llama.values())
    assert abs(total - 39.3e6) < 0.01e6
    e405 = training_energy_kwh(llama["405B"], 700)
    e_all = training_energy_kwh(total, 700)
    print(f"Llama 3.1 405B: {e405 / 1e6:.1f} ГВт·ч, все три: {e_all / 1e6:.1f} ГВт·ч, "
          f"с PUE 1,1: {e_all * 1.1 / 1e6:.1f} ГВт·ч")
    # сверка с бюджетом: 3,8e25 операций при 400 Тфлоп/с на GPU (таблица 4 отчёта)
    need = 3.8e25 / 400e12 / 3600
    print(f"405B по бюджету C = 3,8e25: {need / 1e6:.1f} млн GPU-часов из {llama['405B'] / 1e6:.2f}")
    pj = e405 * KWH / 3.8e25 * 1e12
    print(f"энергия на полезную операцию: {pj:.2f} пДж против {700 / 989e12 * 1e12:.2f} на пике")
    assert 26 < need / 1e6 < 27 and 1.9 < pj < 2.1
    # BLOOM: 1,08 млн GPU-часов A100 (400 Вт) -> 433 МВт·ч в статье
    e_bloom = training_energy_kwh(1_082_990, 400)
    print(f"BLOOM: {e_bloom / 1e3:.0f} МВт·ч по TDP")
    assert round(e_bloom / 1e3) == 433
    # BookLM на GB10: 25,5 мин, DGX Spark целиком до 240 Вт; перевод корпуса — 68 958 историй
    e_booklm = training_energy_kwh(25.5 / 60, 240)
    e_translate = training_energy_kwh(68_958 / 3 / 3600, 240)
    print(f"BookLM: обучение не больше {e_booklm:.2f} кВт·ч, перевод корпуса около "
          f"{e_translate:.1f} кВт·ч, в {e_translate / e_booklm:.0f} раз больше")
    assert e_booklm < 0.11 and 14 < e_translate / e_booklm < 16
    # бюджет главы 7: условный ускоритель 100 Тфлоп/с, MFU 40 %; положим ему 400 Вт
    seconds = 5.26e15 / (100e12 * 0.40)
    print(f"BookLM на условном ускорителе: {seconds:.0f} с, {400 * seconds / 1e3:.0f} кДж")


if __name__ == "__main__":
    check_power_table()
    check_server_and_rack()
    check_tokens()
    check_power_capping()
    check_datacenter()
    check_training()
    print("все проверки прошли")
