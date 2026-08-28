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

if [[ "${1:-}" == "--cov" ]]; then
  shift
  command -v python3 >/dev/null || { echo "нет python3"; exit 1; }
  python3 -c "import coverage" 2>/dev/null || {
    echo "coverage не установлен: pip install -r requirements-dev.txt"; exit 1; }
  rm -rf .coverage .coverage.*
  PYTHONPATH=. COVERAGE_PROCESS_START="$PWD/.coveragerc" \
    python3 -m coverage run -m pytest tests/ -q "$@" || exit 1
  python3 -m coverage combine -q 2>/dev/null
  python3 -m coverage report --sort=cover
else
  PYTHONPATH=. python3 -m pytest tests/ -q "$@"
fi
