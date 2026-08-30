"""Скрипт загрузки листов матрицы высот: он порождается, а не пишется руками.

Проверяется не текст, а поведение: порождённый скрипт запускается по-настоящему
с подставным curl. Ошибка здесь дорогая и отложенная — обнаружится она не тут, а
через несколько часов на виртуалке, где данных уже не докачать.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "postproc" / "list_dem_tiles.py"


def stations(tmp_path, points):
    p = tmp_path / "st.json"
    p.write_text(json.dumps([{"lat": la, "lon": lo} for la, lo in points]))
    return p


def make_script(tmp_path, points=((55.0, 92.0),), margin="0.05"):
    """Порождает fetch.sh для заданных станций и возвращает путь к нему."""
    out = tmp_path / "fetch.sh"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--stations", str(stations(tmp_path, points)),
         "--script", str(out), "--margin-deg", margin],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return out, r.stdout


def fake_curl(tmp_path, *, missing=(), truncate=()):
    """Подставной curl: пишет настоящий gzip, кроме перечисленных листов.

    `missing` — сервер отвечает отказом (лист над морем).
    `truncate` — файл скачивается, но битый (обрыв связи).
    """
    binp = tmp_path / "bin"
    binp.mkdir(exist_ok=True)
    (binp / "curl").write_text(f"""#!/usr/bin/env bash
# аргументы: ... -o <файл> <url>
dest=""; url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) dest="$2"; shift 2;;
    -*) shift;;
    *) url="$1"; shift;;
  esac
done
name=$(basename "$url" .hgt.gz)
echo "$name" >> "{tmp_path}/calls.txt"
for m in {" ".join(missing) or "__none__"}; do [[ "$name" == "$m" ]] && exit 22; done
for m in {" ".join(truncate) or "__none__"}; do
  [[ "$name" == "$m" ]] && {{ printf '\\x1f\\x8b\\x08\\x00' > "$dest"; exit 0; }}
done
printf 'высоты' | gzip -c > "$dest"
exit 0
""")
    (binp / "curl").chmod(0o755)
    return binp


def run_fetch(script, out_dir, binp):
    env = {**os.environ, "PATH": f"{binp}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(script), str(out_dir)],
                          capture_output=True, text=True, env=env)


def calls(tmp_path):
    f = tmp_path / "calls.txt"
    return f.read_text().split() if f.exists() else []


def test_listing_alone_downloads_nothing(tmp_path):
    """Составление списка — не загрузка. Ровно на этом уже споткнулись.

    Скрипт печатает список листов и молча пишет fetch.sh; ни одного байта данных
    при этом не появляется. Поэтому он обязан сказать об этом прямым текстом.
    """
    _, stdout = make_script(tmp_path)
    assert not list(tmp_path.glob("*.hgt.gz"))
    assert "ЕЩЁ НЕ СКАЧАНЫ" in stdout


def test_generated_script_is_valid_bash(tmp_path):
    script, _ = make_script(tmp_path, points=((55.0, 92.0), (50.3, 83.8), (59.5, 98.0)))
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_every_needed_tile_is_fetched(tmp_path):
    script, stdout = make_script(tmp_path, points=((55.4, 92.4), (50.3, 83.8)))
    n = int([l for l in stdout.splitlines() if l.startswith("нужно листов")][0]
            .split(":")[1].split("(")[0])
    binp = fake_curl(tmp_path)
    r = run_fetch(script, tmp_path / "dem", binp)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(list((tmp_path / "dem").glob("*.hgt.gz"))) == n
    assert f"скачано: {n}" in r.stdout


def test_progress_is_printed_for_every_tile(tmp_path):
    """Ход загрузки виден. Без этого несколько минут молчания неотличимы
    от зависания — так и вышло на самом деле."""
    script, _ = make_script(tmp_path, points=((55.4, 92.4),))
    r = run_fetch(script, tmp_path / "dem", fake_curl(tmp_path))
    tiles = calls(tmp_path)
    assert tiles, "ни одного листа не запрошено"
    for t in tiles:
        assert t in r.stdout, f"лист {t} скачан молча"
    assert "[1/" in r.stdout, "нет счётчика"


def test_second_run_downloads_nothing(tmp_path):
    """Повторный запуск не качает заново — иначе после обрыва всё с начала."""
    script, _ = make_script(tmp_path, points=((55.4, 92.4),))
    binp = fake_curl(tmp_path)
    run_fetch(script, tmp_path / "dem", binp)
    first = len(calls(tmp_path))
    assert first > 0
    r = run_fetch(script, tmp_path / "dem", binp)
    assert len(calls(tmp_path)) == first, "листы скачаны повторно"
    assert f"уже было: {first}" in r.stdout


def test_truncated_tile_is_refetched(tmp_path):
    """Обрезанный архив НЕ считается готовым.

    Прерванная загрузка оставляет файл ненулевой длины. Проверка «файл есть»
    приняла бы его за готовый лист, и битые высоты всплыли бы уже на виртуалке
    при сборке мозаики. Целостность проверяется gzip -t.
    """
    script, _ = make_script(tmp_path, points=((55.4, 92.4),))
    dem = tmp_path / "dem"
    # первый заход: один лист приходит битым
    bad = "N55E092"
    binp = fake_curl(tmp_path, truncate=(bad,))
    r = run_fetch(script, dem, binp)
    assert not (dem / f"{bad}.hgt.gz").exists(), "битый лист остался на диске"
    assert "не найдено: 1" in r.stdout

    # второй заход, связь наладилась — лист докачивается
    (tmp_path / "calls.txt").unlink()
    r = run_fetch(script, dem, fake_curl(tmp_path))
    assert bad in calls(tmp_path), "битый лист не перекачали"
    assert gz_ok(dem / f"{bad}.hgt.gz")


def gz_ok(path):
    return subprocess.run(["gzip", "-t", str(path)], capture_output=True).returncode == 0


def test_missing_tile_does_not_stop_the_rest(tmp_path):
    """Листа нет на сервере — остальные всё равно докачиваются.

    Над водой листов не существует, и это нормально. Обрыв всей загрузки
    на первом же таком листе означал бы, что данные не собрать вовсе.
    """
    script, stdout = make_script(
        tmp_path, points=((55.4, 92.4), (50.3, 83.8), (57.6, 95.2), (52.1, 89.7)))
    n = int([l for l in stdout.splitlines() if l.startswith("нужно листов")][0]
            .split(":")[1].split("(")[0])
    assert n >= 3, "для проверки нужно хотя бы три листа"
    binp = fake_curl(tmp_path)
    # узнаём имена листов, ничего не скачивая по-настоящему
    run_fetch(script, tmp_path / "probe", binp)
    names = calls(tmp_path)
    (tmp_path / "calls.txt").unlink()

    binp = fake_curl(tmp_path, missing=(names[0],))
    r = run_fetch(script, tmp_path / "dem", binp)
    assert r.returncode == 0
    assert f"скачано: {n - 1}" in r.stdout and "не найдено: 1" in r.stdout
    assert len(calls(tmp_path)) == n, "загрузка оборвалась на отсутствующем листе"


def test_tile_urls_match_the_source_layout(tmp_path):
    """Лист лежит в подкаталоге по широте: skadi/N55/N55E092.hgt.gz."""
    script, _ = make_script(tmp_path, points=((55.4, 92.4),))
    text = script.read_text()
    assert "/skadi/N55/N55E092.hgt.gz" in text


def test_margin_widens_the_tile_set(tmp_path):
    """Запас вокруг станции берёт соседние листы — окно признаков рельефа
    выходит за клетку в 1°, и без запаса край окна оказался бы пустым."""
    _, narrow = make_script(tmp_path, points=((55.5, 92.5),), margin="0.05")
    _, wide = make_script(tmp_path, points=((55.5, 92.5),), margin="0.8")
    def count(s):
        return int([l for l in s.splitlines() if l.startswith("нужно листов")][0]
                   .split(":")[1].split("(")[0])
    assert count(narrow) == 1
    assert count(wide) == 9, "запас в 0,8° обязан захватить все восемь соседей"
