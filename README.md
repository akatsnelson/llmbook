# Большие языковые модели: код, ноутбуки и модель BookLM

Код к книге А. И. Кацнельсона и В. А. Колесник «Большие языковые модели». Здесь лежит всё, чем в книге
получены числа опытов: сквозная модель BookLM с весами и корпусом, все листинги книги вместе с их
выводом, полные версии длинных листингов, опыты на BookLM и 22 ноутбука-лаборатории. Текста книги в
репозитории нет.

**BookLM** — маленький русский сказочник на 6,6 млн параметров с той же архитектурой, что у больших
моделей вроде LLaMA. Книга собирает её по частям, обучает на переведённых детских историях, дообучает
выполнять просьбы и на ней же показывает почти каждый механизм. Модель отвечает на обычном процессоре
за секунду и пишет связные, но по-детски корявые сказки: путает род и падежи, иначе понимает редкие
слова в просьбе (подробнее — в [models/booklm/README.md](models/booklm/README.md), раздел «Чего от неё ждать»).

## С чего начать

1. Установите окружение (раздел «Установка» ниже) и проверьте его: `python labs/ch01/check_setup.py`.
2. Поговорите с моделью в браузере:
   `python booklm/serve_chat.py --base models/booklm/booklm-base.pt --instruct models/booklm/booklm-instruct.pt --tokenizer models/booklm`
   и откройте http://localhost:8080.
3. Откройте первый ноутбук, где модель собирается руками: `notebooks/04_attention_block.ipynb`.
4. Дальше идите по главам: карта ниже.

## Карта по главам

| Глава | Тема | Листинги | Опыты на BookLM | Ноутбуки |
|---:|---|---|---|---|
| 1 | Введение | `labs/ch01` | `booklm/experiments/ch01.py` | `00_booklm_contract` |
| 2 | История ИИ и языковых моделей | `labs/ch02` | `ch02.py` | — |
| 3 | Токенизация и эмбеддинги | `labs/ch03` | `ch03.py` | `03_tokenizer` |
| 4 | Архитектура трансформера | `labs/ch04` | `ch04.py` | `04_attention_block` |
| 5 | Архитектуры современных LLM | `labs/ch05` | `ch05*.py` | `05_booklm_forward` |
| 6 | Мультимодальные модели | `labs/ch06` | `ch06*.py` | `06_pluggable_modalities`, `06_audio_spectrogram_by_hand` |
| 7 | Предобучение | `labs/ch07` | `ch07*.py` | `07_pretraining` |
| 8 | Дообучение | `labs/ch08` | `ch08*.py` | `08_lora_sft` |
| 9 | Обучение с подкреплением и оценка | `labs/ch09` | `ch09*.py` | — |
| 10 | Работа с промптом | `labs/ch10` | `ch10*.py` | `10_constrained_decoding` |
| 11 | Контекст и память | `labs/ch11` | `ch11*.py` | `11_memory` |
| 12 | Поиск по документам (RAG) | `labs/ch12` | `ch12*.py` | `12_rag` |
| 13 | Агенты и инструменты | `labs/ch13` | `ch13*.py` | `13_tools` |
| 14 | Инференс | `labs/ch14` | `ch14*.py` | `14_*` (шесть ноутбуков) |
| 15 | Устройство GPU | `labs/ch15`, `labs/15_cuda_path` | `ch15*.py` | `15_gpu_execution_by_hand` |
| 16 | Распределённое обучение | `labs/ch16` | `ch16*.py` | `16_zero_by_hand` |
| 17 | Выпуск и сопровождение модели | `labs/ch17` | `ch17*.py` | `17_artifact_lineage` |
| 18 | Безопасность LLM-приложений | `labs/ch18` | `ch18*.py` | `18_security_boundaries` |

Опыты лежат в `booklm/experiments/`, ноутбуки — в `notebooks/`. Список опытов файла и что каждый печатает
показывает `python -m booklm.experiments.ch05 --help`.

## Как найти код к разделу книги

QR-коды и ссылки на полях книги ведут прямо на файлы этого репозитория. Если под рукой только номер главы,
ищите так:

| Что в книге | Где в репозитории |
|---|---|
| любой листинг книги с его выводом | `listings/chNN/<файл главы>.py`: все листинги главы по порядку, вывод из книги — комментарием под листингом |
| длинный листинг, сокращённый в тексте | `labs/chNN/`, где NN — номер главы; в начале файла сказано, какой раздел он дополняет |
| рубрика «А теперь на нашей модели» | `booklm/experiments/chNN*.py`; запуск — `python -m booklm.experiments.chNN` |
| ноутбук главы | `notebooks/NN_*.ipynb`; карта по главам — [notebooks/README.md](notebooks/README.md) |
| сама модель BookLM | `booklm/model.py`; загрузка, генерация, перплексия — `booklm/lab.py` |
| веса, токенизатор, корпус | `models/booklm/`; описание — [models/booklm/README.md](models/booklm/README.md) |
| как связаны файлы модели между главами | [BOOKLM.md](BOOKLM.md) |

Файлы `labs/chNN/*_checks.py` повторяют вычисления листингов главы, но вместо `print` в них стоят `assert`.
Запустите такой файл после правки кода, и он скажет, какое из чисел книги перестало сходиться.

Эталонные выводы опытов лежат в `models/booklm/experiments/`: в `*.txt` — то, что опыт печатает, в `*.csv` —
таблицы, по которым нарисованы графики. Там же небольшие веса опытов (адаптеры LoRA, модели со смесью
экспертов). Многие опыты при запуске перезаписывают свой файл; сравнить свой запуск с эталоном можно
командой `git diff models/booklm/experiments`.

## Установка

Нужны Python 3.10 или новее и [Git LFS](https://git-lfs.com) для весов.

```bash
git clone https://github.com/akatsnelson/llmbook.git
cd llmbook
git lfs pull                      # веса и корпус, около 350 МБ
python3 -m venv .venv
source .venv/bin/activate         # в Windows: .venv\Scripts\activate
pip install -r requirements-extra.txt
python labs/ch01/check_setup.py   # проверка окружения
```

`requirements.txt` — минимум для большинства глав (PyTorch, tokenizers, NumPy, JupyterLab).
`requirements-extra.txt` добавляет пакеты для экспорта и форматов: transformers, safetensors, ONNX,
onnxruntime и gguf. На Linux без видеокарты PyTorch легче поставить с CPU-индекса:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Запуск

Все команды выполняются из корня репозитория.

```bash
python labs/ch04/attention_checks.py          # одна лабораторная
python -m booklm.experiments.ch03             # все опыты главы 3
python -m booklm.experiments.ch03 ch03_bpe_merges   # один опыт
jupyter lab notebooks/                        # ноутбуки
python scripts/check_notebooks.py --execute   # все ноутбуки разом
python scripts/check_listings.py listings/ch04/04-innovations.py   # листинги главы и сверка с книгой
```

Листинги одного файла опираются друг на друга, поэтому запускайте их по порядку: целиком или ячейками
сверху вниз (разметка `# %%` понятна VS Code и Jupyter). `scripts/check_listings.py` исполняет их и
сверяет напечатанное с выводом из книги.

BookLM загружается одной строкой:

```python
from booklm.lab import load, generate
tok, model = load("instruct")
print(generate(tok, model, "Напиши сказку про кота и рыбу.", chat=True, seed=1))
```

## Железо и время

Почти всё считается на обычном процессоре без сети: лабораторные и опыты — за секунды или минуты, все 22
ноутбука — минут за пятнадцать, из них около десяти уходит на ноутбук главы 8.

Видеокарта NVIDIA нужна в трёх местах: `labs/15_cuda_path` (CUDA C++), обучение самой BookLM
(`booklm/train.py`, `booklm/sft_lora.py`) и перевод корпуса (`booklm/translate_llm.py`). Для книги их
повторять не нужно: готовые веса лежат в `models/booklm/`.

Несколько опытов обучают модели и идут долго, а два скачивают модели Qwen. Время — на процессоре
Apple M2 Max:

| Файл | Время | Сеть |
|---|---|---|
| `labs/ch07/mup_width_transfer.py` | около полутора часов на 6 процессах; готовые точки лежат в `mup_width_transfer.json`, и без них файл только печатает таблицу | — |
| `labs/ch05/moe_train.py` | около часа (около 5 минут на RTX 5090); готовые веса уже в `models/` | — |
| `labs/ch08/dora_booklm.py` | около часа | — |
| `booklm/experiments/ch07_muon_midtraining.py ch07_muon` | около 45 минут на 8 потоках | — |
| `labs/ch08/merge_booklm.py` | около 30 минут | — |
| `booklm/experiments/ch09_knowledge_gaps.py ch09_gaps_patch` | около 25 минут | — |
| `labs/ch08/merge_mini_barrier.py` | около 20 минут | — |
| `labs/ch05/moe_prune_merge.py`, `labs/ch05/masked_diffusion_tales.py` | 10–15 минут | — |
| `labs/ch10/sparse_autoencoder.py` | около 10 минут | — |
| `labs/ch05/attention_sinks.py` | около 5 минут | скачивает Qwen3-0.6B и Qwen3.5-0.8B-Base, около 3,3 ГБ |
| `labs/ch03/scaffold_tokens.py` | несколько минут | скачивает токенизатор Qwen3, около 11 МБ |

## Воспроизводимость

Числа в книге получены на torch 2.14.1 и transformers 5.22.2, в основном на процессоре Apple M2 Max.
На другой версии библиотек или другом процессоре последние знаки чисел могут отличаться: суммы с
плавающей точкой складываются в другом порядке. Обычно это видно только в последнем знаке, но жадная
генерация на границе двух почти равных токенов может пойти другим путём. Где мы это видели, проверка в
файле сравнивает общее, а не каждый символ.

## Если что-то не работает

- `torch.load` падает с `invalid load key` или `UnpicklingError`: веса не скачаны, вместо них лежат
  указатели Git LFS. Выполните `git lfs install` и `git lfs pull`.
- `pip install` отвечает `externally-managed-environment`: вы ставите пакеты в системный Python.
  Создайте окружение `.venv`, как в разделе «Установка».
- `git lfs pull` отвечает, что превышена квота трафика: GitHub ограничивает скачивание больших файлов
  в месяц. Попробуйте позже.
- Число в последнем знаке не совпало с книгой: см. «Воспроизводимость». Если расходится больше, проверьте
  версии Python, NumPy и PyTorch: их печатает `labs/ch01/check_setup.py`.

## Лицензии

Код, веса BookLM, адаптеры и токенизатор — MIT ([LICENSE](LICENSE)).

Корпус BookLM в `models/booklm/data/` — перевод на русский набора TinyStories (Eldan, Li, 2023), выпущенного
под CDLA-Sharing-1.0, и распространяется на тех же условиях. Подробности — в `models/booklm/README.md`.
