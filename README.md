# VAC

Настольное приложение для интерпретации виброакустической цементометрии.
Текущая модульная версия — база дальнейшего развития программы.

## Установка и запуск

Python **3.12, 64 бит**. Зависимости закреплены в `requirements.txt`.

Windows, PowerShell из корня проекта:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m vac
```

Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m vac
```

Для разработки: `python -m pip install -e .`. После установки `python -m vac`
или `vac` запускается из любого каталога. `VAC_python.py` и
`VAC_python_mini.py` остаются совместимыми стартовыми скриптами одной основной формы.
Отдельная mini-версия не поддерживается.

На Debian/Ubuntu Qt могут потребоваться `libgl1`, `libegl1`,
`libxkbcommon-x11-0`, `libxcb-xinerama0` и библиотеки XCB.
Проверки без рабочего стола выполняются на Linux; они не заменяют визуальную
проверку GUI, шрифтов, масштабирования и Word на Windows и Linux.
Полный снимок проверенного Linux-окружения: `docs/requirements-linux-py312.lock.txt`.
Переносимость этого снимка на Windows не проверена.

## Работа с данными

Приложение загружает LVM и TWF, обрабатывает сигналы, строит цементограммы,
поддерживает ручные исправления и TXT/PNG/DOCX-экспорт.
В меню **Архив** доступны импорт, сохранение и восстановление версий обработки.
Новый расчёт создаёт отдельную версию; открытие сохранённой версии не пересчитывает её.

Шаблон Word находится в `vac/resources/template.docx`, кривые — в
`vac/reference.py`. Эти ресурсы не зависят от текущего каталога.
Файловый экспорт создаёт и перезаписывает результаты в текущем рабочем каталоге.
Пути с пробелами заключайте в кавычки.

Старые `.vac` содержат настройки и часть интерпретации, требуют загруженных
измерений и не заменяют архив исследования. Открывайте только доверенные файлы:
их формат pickle допускает выполнение кода.

[Работа с архивом](docs/archive.md) · [Формат и API](docs/storage.md) ·
[Резервирование и миграции](docs/backup.md)

```sh
python -m vac.storage.backup backup "исследования.sqlite" "копии/исследования.sqlite"
python -m vac.storage.backup verify "копии/исследования.sqlite"
python -m vac.storage.backup restore "копии/исследования.sqlite" "восстановление/исследования.sqlite"
```

Назначение должно быть новым файлом. Для открытого SQLite используйте эту команду,
а не простое копирование файла: актуальные записи могут находиться в WAL.

## Разработка и проверки

`vac.app` создаёт окно, `vac.controller` связывает интерфейс с моделью состояния.
`vac.io` читает и экспортирует данные, `vac.processing` и `vac.calculation`
выполняют вычисления, `vac.services` предоставляет обработку и сохранение,
`vac.reporting` формирует отчёты, `vac.storage` хранит локальный SQLite-архив.
`vac.evaluation` оценивает метки интерпретации на независимой разметке:
[описание метода и оценка изменений](docs/interpretation.md).

Основная форма генерируется из `VAC_dialog.ui`, а не редактируется вручную:

```sh
python tools/generate_ui.py
python tools/generate_ui.py --check
```

Проверки выполняются стандартным `unittest`, pytest не требуется:

```sh
python -W ignore -m unittest discover -s tests/unit -v
python -W ignore -m unittest discover -s tests/integration -v
python tests/check_package.py
python tests/check_responsive_layout.py
python tools/smoke.py --scenario launch
python tools/smoke.py --scenario new
python tools/smoke.py --scenario old
python tools/check_archive_reliability.py --output work/archive-performance.json
```

Для headless-запуска интеграционных тестов задайте `QT_QPA_PLATFORM=offscreen`
и `MPLBACKEND=Agg`. Тесты архива проверяют версии, откат ошибок, сохранение ручных
правок, резервирование, восстановление и Word. Небольшой набор ожидаемых
результатов для проверки расчётов описан в [tests/fixtures](tests/fixtures/README.md).

Исторические отчёты `stage-*`, пошаговые снимки переписывания и инструменты
сравнения со старой программой удалены из рабочей структуры; они доступны
в истории Git. Рабочие исходники, миграции, шаблон и примеры измерений сохранены.
