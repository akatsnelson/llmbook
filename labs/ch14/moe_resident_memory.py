"""Резидентная память смеси экспертов: BookLM-MoE и gpt-oss-120b в BF16 и MXFP4.

Дополнение к подразделу «Форматы с плавающей точкой» главы 14 (раздел 14.5 «Квантование»).
В книге оставлены мысль «память определяют все эксперты, скорость — только активные»
и итог по gpt-oss-120b; здесь подсчёт до байта для BookLM-MoE из главы 5.

Запуск из корня репозитория: python labs/ch14/moe_resident_memory.py
"""


def resident_bytes(expert_params, other_params, expert_bits, other_bits=16):
    """Байты в памяти: эксперты в expert_bits бит на параметр, остальное в other_bits."""
    return expert_params * expert_bits / 8 + other_params * other_bits / 8


# --- BookLM-MoE: E=4 эксперта, k=2 на токен, d=256, d_ff=768, 6 блоков
# 6 блоков * 4 эксперта * 3 матрицы SwiGLU * (256 * 768) чисел в матрице
expert = 6 * 4 * 3 * 256 * 768
total = 17_245_440                      # все параметры BookLM-MoE
other = total - expert                  # внимание, нормализации, таблица токенов, маршрутизатор
assert expert == 14_155_776 and other == 3_089_664
assert 6 * 256 * 4 == 6144              # маршрутизатор: d -> E в каждом блоке
active = other + expert // 2            # k=2 из E=4 экспертов работают на токене
assert active == 10_167_552

# обе матрицы эксперта делятся на блоки MXFP4 по 32 нацело
assert 256 % 32 == 0 and 768 % 32 == 0

bf16 = resident_bytes(expert, other, 16)
mx = resident_bytes(expert, other, 4.25)        # (32*4 + 8) / 32 = 4.25 бита
assert bf16 == 34_490_880                       # 32.9 МБ
assert mx == 14_155_776 * 17 / 32 + 3_089_664 * 2 == 13_699_584   # 13.1 МБ
MiB = 2 ** 20
assert round(bf16 / MiB, 1) == 32.9 and round(mx / MiB, 1) == 13.1
dense_booklm = 6_622_464 * 2                    # плотная BookLM в BF16: 12.63 МБ
print(f"BookLM-MoE: BF16 {bf16 / MiB:.1f} MiB, MXFP4 {mx / MiB:.1f} MiB, "
      f"dense BookLM {dense_booklm / MiB:.2f} MiB")

# --- gpt-oss-120b (карточка модели): эксперты 114.71e9, остальное 2.12e9
GB, GiB = 1e9, 2 ** 30
oss_bf16 = resident_bytes(114.71e9, 2.12e9, 16)
oss_mx = resident_bytes(114.71e9, 2.12e9, 4.25)
assert oss_bf16 > 80 * GB > oss_mx              # на ускоритель 80 ГБ влезает только MXFP4
assert round(oss_bf16 / GB, 1) == 233.7         # три ускорителя по 80 ГБ
assert round(oss_mx / GB, 1) == 65.2
# Карточка даёт 60.8 ГБ: маршрутизаторы и смещения экспертов лежат в BF16,
# а здесь они посчитаны вместе с экспертами по 4.25 бита.
assert round(oss_mx / GiB, 1) == 60.7
print(f"gpt-oss-120b: BF16 {oss_bf16 / GB:.1f} GB, MXFP4 {oss_mx / GB:.1f} GB")
