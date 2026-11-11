"""Глава 18, раздел 18.1 «Модель угроз», подраздел «Закрытый фрагмент и кэш ответов»:
листинг «Закрытый фрагмент и ключ кэша» (полная версия).

В книге оставлена сокращённая версия; этот файл самодостаточен и
выполняется в ноутбуке главы notebooks/18_security_boundaries.ipynb.

Фильтр прав до поиска не пускает закрытый фрагмент customer-secret к
пользователю public ни в ответе, ни в цитате. Третий канал — кэш: если ключ
кэша содержит только текст вопроса, public получает ответ финансиста целиком.
Проверка двусторонняя: чужое не вышло, своё по-прежнему читается.

Запуск из корня репозитория: python labs/ch18/acl_cache_leak.py
"""

CHUNKS = [  # (chunk_id, access, text): уменьшенный корпус 12_rag.ipynb
    ("params",   "public",  "BookLM содержит 6 622 464 параметра"),
    ("kv",       "public",  "KV-кэш BookLM на 512 токенов занимает 768 КБ"),
    ("vllm",     "public",  "vLLM размещает KV-кэш страницами"),
    ("deploy17", "public",  "deploy-17 запускает booklm-rag-v1 на H100-80GB"),
    ("deploy09", "public",  "архивный deploy-09 запускал BookLM на A100-40GB"),
    ("secret",   "finance", "лимит клиента 900 000 токенов; KV-кэш BookLM 512"),
]


def visible(roles):                     # права до поиска, как в главе 12
    return [c for c in CHUNKS if c[1] == "public" or c[1] in roles]


def search(query, corpus, k=2):         # учебный поиск: число общих слов
    words = set(query.lower().split())
    hits = [(len(words & set(t.lower().split())), cid, t)
            for cid, _, t in corpus]
    return sorted([h for h in hits if h[0] > 0],
                  key=lambda h: (-h[0], h[1]))[:k]


def answer(query, roles, cache, cache_key):
    key = cache_key(query, roles)
    if key not in cache:                # промах: ищем в разрешённом виде
        hits = search(query, visible(roles))
        cache[key] = {"text": " / ".join(t for _, _, t in hits),
                      "cites": [cid for _, cid, _ in hits]}
    return cache[key]


def good_key(query, roles):             # роли и версия индекса в ключе
    return (query, frozenset(roles), "index-v17")


def bad_key(query, roles):              # "оптимизация": ключ по тексту
    return query


def crossed(result):                    # вышел ли фрагмент к public?
    return "900 000" in result["text"] or "secret" in result["cites"]


assert len(visible({"public"})) == 5
assert len(visible({"public", "finance"})) == 6
q = "лимит клиента KV-кэш BookLM 512 токенов"
for cache_key, leak_expected in ((good_key, False), (bad_key, True)):
    cache = {}
    fin = answer(q, {"public", "finance"}, cache, cache_key)  # финансист
    pub = answer(q, {"public"}, cache, cache_key)             # затем public
    assert "secret" in fin["cites"]      # финансист читает, как раньше
    assert crossed(pub) == leak_expected
    if not leak_expected:
        assert pub["cites"][0] == "kv"   # public тоже получил ответ

if __name__ == "__main__":
    print("acl_cache_leak: все проверки пройдены")
