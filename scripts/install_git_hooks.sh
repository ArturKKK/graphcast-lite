#!/usr/bin/env bash
# Ставит проверку перед коммитом. Запустить один раз на каждой машине:
#   bash scripts/install_git_hooks.sh
#
# Хук гоняет только линтер (секунды) — тесты перед каждым коммитом ждать никто
# не станет, и хук просто отключат. Тесты гоняются перед пушем, руками или в CI.
#
# Обойти в разовом случае: git commit --no-verify
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
hook=.git/hooks/pre-commit
cat > "$hook" <<'HOOK'
#!/usr/bin/env bash
# Ставится scripts/install_git_hooks.sh. Обойти: git commit --no-verify
repo=$(git rev-parse --show-toplevel)
command -v ruff >/dev/null 2>&1 || { echo "[хук] ruff нет, пропускаю"; exit 0; }
if ! bash "$repo/scripts/check.sh" --fast; then
  echo
  echo "[хук] коммит остановлен линтером. Починить или: git commit --no-verify"
  exit 1
fi
HOOK
chmod +x "$hook"
echo "хук установлен: $hook"
echo "проверяю на текущем состоянии..."
bash scripts/check.sh --fast
