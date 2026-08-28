#!/usr/bin/env bash
# Тесты постобработки. Одна короткая команда — её и надо набирать на виртуалке.
#
# Работают и без torch: датасет, геометрия, нарезка, признаки-наблюдения, учёт,
# преобразования координат, построение графа и порядок узлов от него не зависят,
# а в conftest.py стоит заглушка на случай, когда его нет.
#
# ВАЖНО. Тесты целевой функции — веса широты, маска каналов, вес области —
# требуют настоящего torch и без него ПРОПУСКАЮТСЯ. Их обязательно надо прогнать
# на виртуалке: там они и проверяются. Если внизу написано "skipped", значит
# часть проверок не выполнялась, и полагаться на зелёный ответ нельзя.
#
#   bash scripts/run_tests.sh            — прогнать тесты
#   bash scripts/run_tests.sh --cov      — ещё и показать покрытие
#
# Про покрытие. Часть скриптов проверяется запуском подпроцессом: так видно
# поведение всей команды, а не набора функций. Без COVERAGE_PROCESS_START такие
# запуски не засчитываются, и отчёт врёт — у add_obs_lags.py при пяти сквозных
# тестах показывало 0 %. Поэтому здесь и .coveragerc с parallel.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

# Какой питон. На виртуалке в PATH стоит conda-питон, в котором нет ни pytest,
# ни torch, ни зависимостей проекта, — а окружение лежит в /data/venvs/graphcast.
# Без этого выбора команда падала с «No module named pytest», хотя всё на месте.
PY=""
for cand in "${VIRTUAL_ENV:-}/bin/python" /data/venvs/graphcast/bin/python \
            "$(command -v python3 || true)"; do
  [[ -x "$cand" ]] && { PY="$cand"; break; }
done
[[ -z "$PY" ]] && { echo "питон не найден"; exit 1; }

if ! "$PY" -c "import pytest" 2>/dev/null; then
  echo "pytest нет в $PY — ставлю (одна попытка, 120 секунд)"
  timeout 120 "$PY" -m pip install -q pytest coverage \
      --extra-index-url https://artifactory.tcsbank.ru/artifactory/api/pypi/python-all/simple \
      2>&1 | tail -3
  "$PY" -c "import pytest" 2>/dev/null || {
    echo "не поставился. Вручную: $PY -m pip install -r requirements-dev.txt"; exit 1; }
fi
echo "питон: $PY"

if [[ "${1:-}" == "--cov" ]]; then
  shift
  "$PY" -c "import coverage" 2>/dev/null || {
    echo "coverage не установлен: pip install -r requirements-dev.txt"; exit 1; }
  rm -rf .coverage .coverage.*
  PYTHONPATH=. COVERAGE_PROCESS_START="$PWD/.coveragerc" \
    "$PY" -m coverage run -m pytest tests/ -q "$@" || exit 1
  "$PY" -m coverage combine -q 2>/dev/null
  "$PY" -m coverage report --sort=cover
else
  PYTHONPATH=. "$PY" -m pytest tests/ -q "$@"
fi
