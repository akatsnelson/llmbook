"""Листинги главы 14 «Инференс» книги «Большие языковые модели», файл 10-ollama.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch14/10-ollama.py
"""

# %% Листинг «Два шаблона BookLM в Ollama»
# Проверка этот листинг не запускает: ему нужны сеть, видеокарта, сервер или внешний файл.
# import subprocess, tempfile, textwrap
# from pathlib import Path
# from booklm.experiments.ch14 import write_gguf, ollama, TEMPLATE_OK
#
# tmp = Path(tempfile.gettempdir())
# write_gguf("instruct", tmp / "booklm-instruct-Q8_0.gguf", "Q8_0")
# broken = TEMPLATE_OK.replace("<|assistant|>", "")
# ask = "Напиши сказку про кота, который боялся темноты."
# for name, template in [("booklm:bad", broken), ("booklm:q8", TEMPLATE_OK)]:
#     (tmp / "Modelfile").write_text(
#         f'FROM ./booklm-instruct-Q8_0.gguf\nTEMPLATE """{template}"""\n'
#         "PARAMETER stop <eos>\nPARAMETER num_ctx 512\n")
#     subprocess.run(["ollama", "create", name, "-f", "Modelfile"],
#                    cwd=tmp, check=True, capture_output=True)
#     r = ollama("/api/chat", model=name, stream=False,
#                options={"temperature": 0},
#                messages=[{"role": "user", "content": ask}])
#     print(f"{name}, вход {r['prompt_eval_count']} токенов:")
#     text = textwrap.shorten(r["message"]["content"], 130)
#     print(textwrap.fill(text, 72, initial_indent="  ",
#                         subsequent_indent="  "))
