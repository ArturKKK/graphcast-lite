# docs/ — структура документации репозитория

После очистки корня (май 2026) вся проектная документация и артефакты выступлений лежат здесь.

## Подпапки

| Папка | Что внутри |
|---|---|
| `diploma/` | Активный исходник диплома (`diploma.tex`) и финальный PDF |
| `diploma/legacy/` | Старые версии: `.bak`, `before_update_*.tex`, ранние `.md` черновики |
| `diploma/build/` | LaTeX-артефакты сборки (`.aux .log .fls .toc .xdv` — в `.gitignore`) |
| `doclad/` | Доклад на защиту + тезисы МНСК-2026 |
| `slides/` | Все варианты презентаций: текущие + apr03 + final + editable |
| `results/` | Результаты экспериментов и план запуска: DA, regional, merge, итоговый summary |
| `prompts/` | Контексты для AI и review (architecture consultation, presentation, regional improvement, website) |
| `notes/` | Рабочие заметки, временные файлы, картинки, RU-черновики (инференсы, подбор параметров) |
| `era5_urban_problems_and_postprocessing.md` | Заметка по городскому ERA5 + MOS |
| `mos_correction.md` | Описание MOS-коррекции |

## Что лежит в корне репо (не трогали)

- `README.md`, `README_RU.MD`, `LICENSE`, `requirements.txt` — стандарт
- `mlc_preset.yaml` — конфиг MLC платформы
- `__init__.py` — корневой пакет
- `.github/`, `.vscode/`, `.gitignore`, `.gitattributes` — служебное
- **Код:** `src/`, `scripts/`, `notebooks/`, `experiments/`
- **Данные/результаты вне VCS:** `data/`, `data-for-diplom/`, `results/`, `logs/`, `aaaa/`, `live_runtime_bundle/`, `viz_variants/`, `vm_backup/`, `slides_visuals/`, `website/`, `conference/`, `archives/`

## Тяжёлые архивы

Перенесены в `/archives/` (в `.gitignore`):
- `jan2023_dataset.tar.gz` (133 MB)
- `region_krsk_61x41_19f_2010-2020_025deg.tar.gz` (1.2 GB)
- `region_krsk_cds_19f.zip` (5.4 MB)
- `city-tabakov_2026_private.zip`
