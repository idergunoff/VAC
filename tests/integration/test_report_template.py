"""Persistence failures and placeholder protection for editable report text."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import docx
from vac.reporting.template import (
    ELEMENTS, TemplateStore, apply_texts, default_texts, template_hash, validate_texts,
)
from vac.resources import template_path


class TemplateTests(unittest.TestCase):
    def test_defaults_and_all_elements_preserve_dynamic_values(self):
        defaults = default_texts()
        validate_texts(defaults)
        original = docx.Document(template_path())
        edited = docx.Document(template_path())
        overrides = {}
        for element in ELEMENTS:
            # Include literal braces, Unicode, multiple lines and reversed placeholders.
            overrides[element.key] = 'Изменено {{ВАЦ}}\n' + ' / '.join(
                '{' + name + '}' for _, name in reversed(element.tokens))
        apply_texts(edited, overrides)
        for element in ELEMENTS:
            with self.subTest(key=element.key):
                before = element.paragraph(original)
                after = element.paragraph(edited)
                expected = 'Изменено {ВАЦ}\n' + ' / '.join(before.runs[i].text for i, _ in reversed(element.tokens))
                self.assertEqual(after.text, expected)
                self.assertEqual(after.alignment, before.alignment)
                for index, name in element.tokens:
                    run = next(r for r in after.runs if r.text == before.runs[index].text)
                    self.assertEqual(run.font.name, before.runs[index].font.name)
                    self.assertEqual(run.font.size, before.runs[index].font.size)
        self.assertEqual(edited.tables[0].cell(0, 3).text, original.tables[0].cell(0, 3).text)

    def test_invalid_tokens_are_rejected(self):
        for text in ('Скважина', '{скважина} {скважина}', '{неизвестно}', '{скважина!r}',
                     '{скважина:>10}', '{скважина.__class__}', '{', '\x00{скважина}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_texts({'well': text})
        validate_texts({'organization': '', 'default_client': 'ООО {Сервис}'})

    def test_save_reload_reset_and_atomic_failure(self):
        original = template_path().read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'Настройки с пробелами' / 'шаблон.json'
            store = TemplateStore(path)
            self.assertEqual(store.load(), {})
            store.save({'organization': 'Новая организация', 'default_client': 'Заказчик'})
            self.assertEqual(TemplateStore(path).load()['organization'], 'Новая организация')
            saved = path.read_bytes()
            with patch('vac.reporting.template.os.replace', side_effect=PermissionError('locked')):
                with self.assertRaises(PermissionError):
                    store.save({'organization': 'Потерянный текст'})
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])
            with self.assertRaises(ValueError):
                store.save({'well': 'Нет подстановки'})
            self.assertEqual(path.read_bytes(), saved)
            store.save(default_texts())
            self.assertEqual(store.load(), {})
            path.write_text('{bad', encoding='utf-8')
            with self.assertRaises(ValueError):
                store.load()
        self.assertEqual(template_path().read_bytes(), original)

    def test_hash_tracks_edits_and_is_order_independent(self):
        self.assertNotEqual(template_hash({}), template_hash({'title': 'Новый текст'}))
        self.assertEqual(template_hash({'title': 'Текст', 'organization': 'Фирма'}),
                         template_hash({'organization': 'Фирма', 'title': 'Текст'}))


if __name__ == '__main__':
    unittest.main()
