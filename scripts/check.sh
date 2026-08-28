#!/usr/bin/env bash
# Полная проверка перед пушем: линтер и тесты.
#
#   bash scripts/check.sh          # линтер + тесты
#   bash scripts/check.sh --fast   # только линтер (секунды)
#
# Два уровня линтера, и это не придирка, а осознанное разделение.
#
# Первый идёт по всему коду и содержит только правила, ловящие НАСТОЯЩИЕ ошибки:
# обращение к неопределённому имени, повторное определение, изменяемое значение
# по умолчанию. Код им уже соответствует, поэтому проверка зелёная — а зелёной
# она обязана быть, иначе её начнут пропускать и она перестанет ловить что-либо.
#
# Второй строже и идёт по каталогам, которые поддерживаются и закрыты тестами.
# Именно этому уровню должен соответствовать весь НОВЫЙ код. Остальное не
# «плохое» — оно просто не приведено к этому уровню; список в pyproject.toml.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

STRICT_PATHS=(tests src/plotting.py src/postprocessing/geometry.py
              src/postprocessing/corpus_math.py scripts/postproc/record_run.py
              scripts/postproc/split_corpus.py)

if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff не установлен: pip install -r requirements-dev.txt"
  exit 1
fi

fail=0
echo "== линтер: весь код, правила настоящих ошибок =="
ruff check --config pyproject.toml src/ scripts/ tests/ || fail=1

echo
echo "== линтер: строгий, поддерживаемые каталоги =="
ruff check --config .ruff-strict.toml "${STRICT_PATHS[@]}" || fail=1

if [[ "${1:-}" != "--fast" ]]; then
  echo
  echo "== тесты =="
  bash scripts/run_tests.sh || fail=1
fi

echo
if [[ $fail -eq 0 ]]; then
  echo "всё чисто"
else
  echo "ЕСТЬ ЗАМЕЧАНИЯ — см. выше"
fi
exit $fail
