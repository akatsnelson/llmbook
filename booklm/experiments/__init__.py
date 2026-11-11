"""Опыты на обученной BookLM для разделов «А теперь на нашей модели», по файлу на главу.

Каждый файл chNN*.py держит словарь EXPERIMENTS «имя опыта -> функция» и кончается вызовом run():

    python -m booklm.experiments.ch03                    # все опыты файла
    python -m booklm.experiments.ch03 ch03_bpe_merges    # один опыт
    python -m booklm.experiments.ch03 --help             # список опытов

Запускать нужно из корня репозитория. Старое имя опыта остаётся в словаре вторым ключом той же
функции (псевдоним), поэтому старые команды тоже работают.
"""
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "booklm" / "experiments"


def _module_name():
    """booklm.experiments.chNN для примера команды в справке."""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return spec.name
    return "booklm.experiments." + Path(sys.argv[0]).stem


def _groups(experiments):
    """[(функция, [имя, псевдоним, ...]), ...] в порядке первого вхождения функции."""
    order = {}
    for name, fn in experiments.items():
        order.setdefault(fn, []).append(name)
    return list(order.items())


def _canonical(experiments, name):
    """Основное имя опыта (первый ключ его функции) или None. Принимает и «-» вместо «_»."""
    for variant in (name, name.replace("-", "_"), name.replace("_", "-")):
        if variant in experiments:
            fn = experiments[variant]
            return next(n for n, f in experiments.items() if f is fn)
    return None


def usage(experiments, file=sys.stdout):
    """Печатает список опытов с первой строкой документации каждого и пример команды."""
    module = _module_name()
    print(f"Опыты файла {module.replace('.', '/')}.py:", file=file)
    for fn, names in _groups(experiments):
        doc = (fn.__doc__ or "").strip().splitlines()
        line = f"  {names[0]}"
        if doc:
            line += f" — {doc[0].strip()}"
        if names[1:]:
            line += f" (старое имя: {', '.join(names[1:])})"
        print(line, file=file)
    first = _groups(experiments)[0][1][0] if experiments else "<имя>"
    one, all_ = f"python -m {module} {first}", f"python -m {module}"
    print("\nЗапуск из корня репозитория:", file=file)
    print(f"  {all_:<{len(one)}}   # все опыты по очереди", file=file)
    print(f"  {one}   # один опыт", file=file)
    print("Что печатает каждый опыт и куда сохраняет вывод, сказано в начале файла.", file=file)


def pick(experiments, argv=None):
    """Имена опытов из командной строки, проверенные по словарю; без имён — все опыты по разу.

    --help и -h печатают список опытов и завершают программу; неизвестное имя — тоже, с кодом 2.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if any(a in ("-h", "--help") for a in argv):
        usage(experiments)
        sys.exit(0)
    if not argv:
        return [names[0] for _, names in _groups(experiments)]
    names = []
    for a in argv:
        name = _canonical(experiments, a)
        if name is None:
            print(f"Нет опыта с именем {a!r}.\n", file=sys.stderr)
            usage(experiments, file=sys.stderr)
            sys.exit(2)
        names.append(name)
    return names


def _value(text):
    """Строку параметра в число, если это число."""
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


class _Tee(io.TextIOBase):
    """Печатает на экран и одновременно копит текст для файла."""

    def __init__(self, screen):
        self.screen, self.buf = screen, io.StringIO()

    def write(self, s):
        self.screen.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.screen.flush()


def run(experiments, argv=None, *, save=None, args=False):
    """Запускает опыты, названные в командной строке (без имён — все).

    save=True — копия вывода каждого опыта в models/booklm/experiments/<имя с дефисами>.txt;
    save={имя: "файл.txt"} — свои имена файлов, опыты без ключа не сохраняются; None — только экран.
    args=True — первое слово имя опыта, остальные слова его параметры: «k=v» именованные,
    прочие по порядку (числа превращаются в int или float).
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    params, kwargs = [], {}
    if args and argv and not any(a in ("-h", "--help") for a in argv):
        argv, rest = argv[:1], argv[1:]
        for a in rest:
            if "=" in a:
                k, v = a.split("=", 1)
                kwargs[k] = _value(v)
            else:
                params.append(_value(a))
    names = pick(experiments, argv)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        fn = experiments[name]
        target = (name.replace("_", "-") + ".txt") if save is True else (save or {}).get(name)
        if not target:
            fn(*params, **kwargs)
            continue
        tee = _Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            fn(*params, **kwargs)
        (OUT / target).write_text(tee.buf.getvalue(), encoding="utf-8")
