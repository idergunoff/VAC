"""Editable Word text, stored separately from the packaged document (no Qt)."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from string import Formatter
import tempfile

import docx

from ..resources import template_path


@dataclass(frozen=True)
class TextElement:
    key: str
    label: str
    location: tuple
    tokens: tuple = ()  # (run index, placeholder name); filled by the existing renderer

    def paragraph(self, document):
        if self.location[0] == 'p':
            return document.paragraphs[self.location[1]]
        if self.location[0] == 'nested':
            return document.tables[1].cell(0, 0).tables[0].cell(0, self.location[1]).paragraphs[0]
        _, table, row, cell = self.location
        return document.tables[table].cell(row, cell).paragraphs[0]


ELEMENTS = (
    TextElement('title', 'Заключение — название метода', ('p', 0)),
    TextElement('organization', 'Заключение — организация', ('p', 1)),
    TextElement('conclusion', 'Заключение — заголовок', ('p', 3)),
    TextElement('well', 'Заключение — номер скважины', ('p', 4), ((2, 'скважина'),)),
    TextElement('client', 'Заключение — заказчик', ('p', 6), ((1, 'заказчик'),)),
    TextElement('summary', 'Заключение — вводный текст', ('p', 9),
                ((3, 'конструкция'), (5, 'скважина'), (7, 'диаметр'), (10, 'от'), (12, 'до'))),
    TextElement('intervals', 'Заключение — заголовок интервалов', ('p', 11), ((1, 'конструкция'),)),
    TextElement('chart_title', 'Цементограмма — заголовок', ('t', 0, 0, 0), ((3, 'скважина'),)),
    TextElement('pipe', 'Цементограмма — конструкция', ('t', 0, 1, 0), ((0, 'конструкция'), (3, 'диаметр'))),
    TextElement('range', 'Цементограмма — интервал исследования', ('t', 0, 1, 1), ((1, 'от'), (5, 'до'))),
    TextElement('chart_client', 'Цементограмма — заказчик', ('t', 0, 2, 0), ((1, 'заказчик'),)),
    TextElement('date', 'Цементограмма — дата', ('t', 0, 2, 1), ((1, 'дата'),)),
    TextElement('scale', 'Цементограмма — подпись масштаба', ('t', 0, 3, 1)),
    TextElement('method', 'Цементограмма — описание метода', ('t', 0, 4, 0)),
    TextElement('coefficient', 'Цементограмма — коэффициент', ('t', 0, 7, 0), ((7, 'коэффициент'),)),
    TextElement('frequency', 'Цементограмма — частота', ('t', 0, 7, 2), ((5, 'частота'),)),
    TextElement('manager_role', 'Подпись — должность руководителя', ('t', 0, 8, 0)),
    TextElement('manager_name', 'Подпись — ФИО руководителя', ('t', 0, 8, 2)),
    TextElement('interpreter_role', 'Цементограмма — подпись интерпретатора', ('t', 0, 9, 0)),
    TextElement('conclusion_interpreter_role', 'Заключение — подпись интерпретатора', ('t', 1, 1, 0)),
    TextElement('upper', 'Таблица — верхняя граница', ('nested', 0)),
    TextElement('lower', 'Таблица — нижняя граница', ('nested', 1)),
    TextElement('thickness', 'Таблица — мощность интервала', ('nested', 2)),
    TextElement('category', 'Таблица — категория дефекта', ('nested', 3)),
)
DEFAULT_LABELS = {
    'default_diameter': 'По умолчанию — диаметр',
    'default_client': 'По умолчанию — заказчик',
    'default_client2': 'По умолчанию — предприятие',
    'default_interpreter': 'По умолчанию — интерпретатор',
}


def default_texts():
    document = docx.Document(template_path())
    result = {}
    for element in ELEMENTS:
        tokens = dict(element.tokens)
        result[element.key] = ''.join(
            '{' + tokens[i] + '}' if i in tokens else run.text.replace('{', '{{').replace('}', '}}')
            for i, run in enumerate(element.paragraph(document).runs))
    table = document.tables[0]
    result.update(
        default_diameter=table.cell(1, 0).paragraphs[0].runs[3].text,
        default_client=table.cell(2, 0).paragraphs[0].runs[1].text,
        default_client2=table.cell(3, 0).text,
        default_interpreter=table.cell(9, 2).text,
    )
    return result


def validate_texts(texts):
    if not isinstance(texts, dict):
        raise ValueError('Тексты шаблона должны быть словарём')
    known = {e.key: e for e in ELEMENTS}
    for key, value in texts.items():
        if key not in known and key not in DEFAULT_LABELS:
            raise ValueError('Неизвестный элемент шаблона: ' + str(key))
        if not isinstance(value, str):
            raise ValueError('Элемент шаблона должен содержать текст: ' + key)
        if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
            raise ValueError('Недопустимый символ в тексте: ' + key)
        if key in DEFAULT_LABELS:
            continue
        element = known[key]
        try:
            fields = []
            for _, field, spec, conversion in Formatter().parse(value):
                if field is not None:
                    if spec or conversion:
                        raise ValueError('Форматирование подстановок не поддерживается')
                    fields.append(field)
            if sorted(fields) != sorted(name for _, name in element.tokens):
                raise ValueError('Сохраните каждую подстановку в фигурных скобках ровно один раз')
        except ValueError as exc:
            raise ValueError(element.label + ': ' + str(exc)) from exc


def apply_texts(document, texts):
    """Apply after computed values are filled, before copying cementogram pages."""
    validate_texts(texts)
    for element in ELEMENTS:
        if element.key not in texts:
            continue
        paragraph = element.paragraph(document)
        runs = paragraph.runs
        values = {name: runs[i].text for i, name in element.tokens}
        styles = {name: deepcopy(runs[i]._r.rPr) for i, name in element.tokens}
        base_style = deepcopy(runs[0]._r.rPr) if runs else None
        paragraph.clear()  # retain paragraph properties, use the original run styles
        for literal, field, _, _ in Formatter().parse(texts[element.key]):
            for content, style in ((literal, base_style), (values.get(field, ''), styles.get(field))):
                if content:
                    run = paragraph.add_run(content)
                    if style is not None:
                        run._r.insert(0, deepcopy(style))


class TemplateStore:
    """Versioned UTF-8 overrides with atomic replacement; errors never erase edits."""
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {}
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError('Не удалось прочитать настройки шаблона: ' + str(self.path)) from exc
        if not isinstance(payload, dict) or payload.get('version') != 1 or 'texts' not in payload:
            raise ValueError('Неизвестный формат настроек шаблона: ' + str(self.path))
        validate_texts(payload['texts'])
        return payload['texts']

    def save(self, texts):
        validate_texts(texts)
        defaults = default_texts()
        overrides = {k: v for k, v in texts.items() if v != defaults[k]}
        payload = json.dumps({'version': 1, 'texts': overrides}, ensure_ascii=False, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix=self.path.name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def template_hash(texts):
    validate_texts(texts)
    digest = sha256(template_path().read_bytes())
    digest.update(json.dumps(texts, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    return digest.hexdigest()
