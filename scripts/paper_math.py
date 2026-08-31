#!/usr/bin/env python3
"""Перевод формул из LaTeX в HTML для вёрстки рукописи.

Зачем свой преобразователь. Готовые движки (MathJax, KaTeX) подключаются
скриптом с внешнего адреса, а в опубликованной странице это запрещено политикой
безопасности; печать в PDF идёт через старый движок, который не понимает MathML.
Поэтому формулы собираются из обычного HTML: дроби — вложенными блоками с
чертой, индексы — sub и sup, буквы — юникодом.

Охвачено ровно то, что встречается в рукописи. Незнакомая команда не молчит, а
остаётся видимой в тексте — так ошибка сразу бросается в глаза при вычитке.
"""
import re

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
    "epsilon": "ε", "varepsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "pi": "π", "rho": "ρ", "sigma": "σ", "Sigma": "Σ", "tau": "τ",
    "phi": "φ", "varphi": "φ", "psi": "ψ", "omega": "ω", "Omega": "Ω",
}
FUNCS = ("cos", "sin", "tan", "exp", "log", "ln", "min", "max", "arg")
WIDE = {"quad": "\u2003", "qquad": "\u2003\u2003", "thinspace": "\u2009"}
OPS = {
    "cdot": "·", "times": "×", "in": "∈", "approx": "≈", "le": "≤", "ge": "≥",
    "neq": "≠", "pm": "±", "to": "→", "dots": "…", "ldots": "…", "cdots": "⋯",
    "infty": "∞", "partial": "∂", "nabla": "∇", "propto": "∝", "sim": "∼",
    # \circ в надстрочном положении — знак градуса. Без него 50^\circ
    # печаталось как «50circ»: в рукописи, ушедшей на рецензию, так и было.
    "circ": "°", "degree": "°", "prime": "′",
}
# Рукописные начертания (𝒢, 𝒩, ℱ) лежат в блоке математических символов, а у
# Times их нет — при печати они молча исчезают. Поэтому обычный курсив:
# в контексте формулы множество узлов и окрестность и так однозначны.
CAL = {}
SPACES = {r"\,": " ", r"\;": " ", r"\ ": " ", r"\!": "", r"\quad": " ",
          r"\qquad": "  "}


def _brace(s, i):
    """Возвращает содержимое {...}, начиная с позиции открывающей скобки."""
    assert s[i] == "{"
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _arg(s, i):
    """Аргумент команды: {...}, целая команда \\имя или один символ.

    Случай с командой обязателен: до 31.08.2026 бралась ровно одна литера, и
    в `50^\\circ` аргументом оказывалась обратная косая черта, а `circ` уезжало
    в текст курсивом. В рукописи, ушедшей на рецензию, так и стояло «50circ».
    """
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return ("", i)
    if s[i] == "{":
        return _brace(s, i)
    if s[i] == "\\":
        j = i + 1
        while j < len(s) and s[j].isalpha():
            j += 1
        if j > i + 1:
            # Команда с аргументом (\mathsf{T}) забирается целиком, иначе
            # аргумент убегал бы из индекса наружу.
            if j < len(s) and s[j] == "{":
                _, k = _brace(s, j)
                return (s[i:k], k)
            return (s[i:j], j)
    return (s[i], i + 1)


def tex_to_html(t: str) -> str:
    """LaTeX-фрагмент → HTML."""
    out, i, n = [], 0, len(t)
    while i < n:
        ch = t[i]

        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+)", t[i:])
            if not m:
                for k, v in SPACES.items():
                    if t.startswith(k, i):
                        out.append(v); i += len(k); break
                else:
                    out.append(t[i + 1] if i + 1 < n else ""); i += 2
                continue
            cmd, i = m.group(1), i + m.end()

            if cmd == "frac":
                a, i = _arg(t, i); b, i = _arg(t, i)
                out.append(f'<span class="frac"><span class="num">{tex_to_html(a)}</span>'
                           f'<span class="den">{tex_to_html(b)}</span></span>')
            elif cmd in ("mathrm", "text", "mathsf", "operatorname"):
                a, i = _arg(t, i)
                out.append(f'<span class="up">{tex_to_html(a)}</span>')
            elif cmd == "mathbf":
                a, i = _arg(t, i)
                out.append(f'<b class="up">{tex_to_html(a)}</b>')
            elif cmd == "mathcal":
                a, i = _arg(t, i)
                out.append(CAL.get(a, f"<i>{a}</i>"))
            elif cmd in ("hat", "widehat"):
                a, i = _arg(t, i)
                out.append(tex_to_html(a) + "̂")
            elif cmd in ("bar", "overline"):
                a, i = _arg(t, i)
                out.append(f'<span class="ovl">{tex_to_html(a)}</span>')
            elif cmd == "sqrt":
                a, i = _arg(t, i)
                out.append(f'√<span class="ovl">{tex_to_html(a)}</span>')
            elif cmd == "underbrace":
                a, i = _arg(t, i)
                lab = ""
                if i < n and t[i] == "_":
                    b, i = _arg(t, i + 1)
                    lab = tex_to_html(b)
                out.append(f'<span class="ubr"><span class="ubr-v">{tex_to_html(a)}</span>'
                           f'<span class="ubr-l">{lab}</span></span>')
            elif cmd == "sum":
                sub = sup = ""
                while i < n and t[i] in "_^":
                    k = t[i]; a, i = _arg(t, i + 1)
                    if k == "_": sub = tex_to_html(a)
                    else: sup = tex_to_html(a)
                out.append(f'<span class="big">Σ<span class="lim">'
                           f'<span class="hi">{sup}</span><span class="lo">{sub}</span></span></span>')
            elif cmd in ("left", "right", "bigl", "bigr", "Bigl", "Bigr"):
                if i < n and t[i] in "()[]|.":
                    out.append("" if t[i] == "." else t[i]); i += 1
            elif cmd == "tag":
                a, i = _arg(t, i)
                out.append(f'<span class="tag">({a})</span>')
            elif cmd in WIDE:
                out.append(WIDE[cmd])
            elif cmd in FUNCS:
                out.append(f'<span class="up">{cmd}</span>')
            elif cmd in GREEK:
                out.append(GREEK[cmd])
            elif cmd in OPS:
                out.append(OPS[cmd])
            else:
                out.append(f'<span class="unk">\\{cmd}</span>')
            continue

        if ch in "_^":
            a, i = _arg(t, i + 1)
            tag = "sub" if ch == "_" else "sup"
            out.append(f"<{tag}>{tex_to_html(a)}</{tag}>")
            continue

        if ch == "{":
            a, i = _brace(t, i)
            out.append(tex_to_html(a) if a != "," else ",")
            continue
        if ch == "}":
            i += 1
            continue

        if ch.isalpha() and ch.isascii():
            j = i
            while j < n and t[j].isalpha() and t[j].isascii():
                j += 1
            out.append(f"<i>{t[i:j]}</i>"); i = j
            continue

        out.append({"<": "&lt;", ">": "&gt;", "&": "&amp;", "-": "\u2212"}.get(ch, ch))
        i += 1

    return "".join(out)


CSS = """
.mf { display:block; text-align:center; margin:1.1em 0; position:relative; }
.mi, .mf { font-family: "Times New Roman", Times, serif; }
.mi i, .mf i { font-style: italic; }
.up, .up i { font-style: normal; }
.frac { display:inline-block; vertical-align:middle; text-align:center; margin:0 .18em; }
.frac .num { display:block; padding:0 .35em .1em; border-bottom:1px solid currentColor; }
.frac .den { display:block; padding:.1em .35em 0; }
.ovl { border-top:1px solid currentColor; padding-top:.03em; }
.big { display:inline-block; vertical-align:middle; font-size:1.35em; line-height:1; position:relative; margin:0 .1em; }
.big .lim { display:inline-block; font-size:.42em; line-height:1.05; vertical-align:middle; margin-left:.05em; }
.big .hi, .big .lo { display:block; }
.ubr { display:inline-block; text-align:center; vertical-align:bottom; }
.ubr .ubr-v { display:block; border-bottom:1px solid currentColor; padding-bottom:.1em; }
.ubr .ubr-l { display:block; font-size:.72em; font-style:normal; padding-top:.15em; }
.mf .tag { position:absolute; right:0; font-style:normal; }
.unk { color:#b00; font-family:monospace; }
sub, sup { font-size:.72em; line-height:0; }
"""

if __name__ == "__main__":
    import sys
    print(tex_to_html(sys.argv[1] if len(sys.argv) > 1 else r"\frac{a}{b}"))
