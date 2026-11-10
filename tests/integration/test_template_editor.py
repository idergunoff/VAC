"""Edit/save/reopen/cancel, direct DOCX export and report pagination."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import docx
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from vac.app import create_window
from vac.domain.models import SignalIdentity
from vac.gui.template_editor import TemplateEditor
from vac.reporting.template import ELEMENTS, TemplateStore
from vac.reporting.word import build_report
from vac.resources import template_path


class TemplateEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='VAC шаблон ')
        self.store = TemplateStore(Path(self.tmp.name) / 'settings' / 'template.json')
        self.patch = patch('vac.gui.template_editor.user_template_store', return_value=self.store)
        self.patch.start()
        self.window, self.c = create_window()

    def tearDown(self):
        self.window.close()
        plt.close('all')
        self.patch.stop()
        self.tmp.cleanup()

    def select(self, dialog, key):
        for row in range(dialog.elements.count()):
            if dialog.elements.item(row).data(QtCore.Qt.UserRole) == key:
                dialog.elements.setCurrentRow(row)
                return
        self.fail('Element not found: ' + key)

    def test_edit_save_restart_and_defaults_button(self):
        self.assertTrue(self.c.template_menu.action.isEnabled())
        dialog = TemplateEditor(self.store, self.window)
        self.select(dialog, 'organization')
        dialog.editor.setPlainText('Новая организация\nОтдел исследований')
        self.select(dialog, 'default_client')
        dialog.editor.setPlainText('Новый заказчик')
        dialog.save_button.click()
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Accepted)
        self.window.close()
        self.window, self.c = create_window()
        reopened = TemplateEditor(self.c.template_menu.store, self.window)
        self.select(reopened, 'organization')
        self.assertEqual(reopened.editor.toPlainText(), 'Новая организация\nОтдел исследований')
        self.c.cement_temp()
        self.assertEqual(self.c.ui.lineEdit_client.text(), 'Новый заказчик')
        reopened.reject()

    def test_cancel_validation_and_failed_save_keep_edits(self):
        dialog = TemplateEditor(self.store, self.window)
        self.select(dialog, 'well')
        dialog.editor.setPlainText('Без номера')
        dialog.save_button.click()
        self.assertIn('не сохранён', dialog.error.text())
        self.assertFalse(self.store.path.exists())
        dialog.reset_element()
        self.select(dialog, 'organization')
        dialog.editor.setPlainText('Несохранённое')
        with patch.object(self.store, 'save', side_effect=PermissionError('locked')):
            dialog.save_button.click()
        self.assertIn('locked', dialog.error.text())
        self.assertEqual(dialog.editor.toPlainText(), 'Несохранённое')
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Cancel):
            dialog.reject()
        self.assertNotEqual(dialog.texts, dialog.saved)
        with patch.object(QtWidgets.QMessageBox, 'question', return_value=QtWidgets.QMessageBox.Discard):
            dialog.reject()
        self.assertFalse(self.store.path.exists())

    def test_direct_export_uses_saved_texts(self):
        self.store.save({'organization': 'Организация для экспорта', 'chart_title': 'Скважина: {скважина}'})
        table = pd.DataFrame({'depth': [0., 1., 2., 3., 4.], 'first_sig': [1., 2., 3., 2., 1.],
                              'corr_sig': [1., 2., 3., 2., 1.], 'func': [.1, .2, .3, .2, .1], 'corr_coeff': 1.})
        self.c.state.interpretation.table = table
        self.c.state.interpretation.signal = SignalIdentity('new', 0, '1 Тест')
        self.c.state.interpretation.automatic_table = table.copy(deep=True)
        self.c.ui.lineEdit_number_skv.setText('17')
        self.c.draw_cement()
        previous = Path.cwd()
        try:
            os.chdir(self.tmp.name)
            with patch.object(plt.Figure, 'show', side_effect=AssertionError('preview')):
                with patch.object(QtWidgets.QFileDialog, 'getSaveFileName',
                                  return_value=(str(Path(self.tmp.name) / 'цементограмма_по_скв_17.docx'), 'Word (*.docx)')):
                    self.c.ui.pushButton_save_cement.click()
            document = docx.Document('цементограмма_по_скв_17.docx')
            self.assertEqual(document.paragraphs[1].text, 'Организация для экспорта')
            self.assertEqual(document.tables[0].cell(0, 0).text, 'Скважина: 17')
        finally:
            os.chdir(previous)

    def test_changed_texts_and_results_on_all_three_pages(self):
        image = Path(self.tmp.name) / 'chart.png'
        from PyQt5.QtGui import QImage
        pic = QImage(10, 10, QImage.Format_RGB32)
        pic.fill(QtCore.Qt.white)
        for suffix in ('', '1', '2'):
            pic.save(str(image.with_name('chart' + suffix + '.png')))
        table = pd.DataFrame({'depth': [0., 400., 801.], 'quality': [0, 0, 0]})
        values = {'lineEdit_number_skv.text': '17', 'comboBox_pipe_type.currentText': 'Колонна',
                  'lineEdit_pipe_diameter.text': '245', 'lineEdit_client.text': 'Заказчик',
                  'lineEdit_client2.text': 'Предприятие', 'lineEdit_interpreter.text': 'Исполнитель',
                  'dateEdit.text': '10.11.2026', 'doubleSpinBox_int_max.value': 801.,
                  'checkBox_golfstrim.checkState': 0}
        texts = {e.key: e.label + ': ' + ' / '.join('{' + name + '}' for _, name in e.tokens) for e in ELEMENTS}
        document = build_report(table, '55', values, template_path(), image, texts=texts)
        path = Path(self.tmp.name) / 'отчёт.docx'
        document.save(path)
        result = docx.Document(path)
        self.assertEqual(len(result.tables), 4)
        self.assertIn('17', result.paragraphs[9].text)
        self.assertIn('245', result.paragraphs[9].text)
        self.assertIn('колонны', result.paragraphs[9].text)
        for i, sheet in enumerate(result.tables[:3]):
            self.assertEqual(sheet.cell(0, 0).text, 'Цементограмма — заголовок: 17')
            self.assertEqual(sheet.cell(7, 2).text, 'Цементограмма — частота: 55')
            self.assertEqual(sheet.cell(8, 2).text, 'Подпись — ФИО руководителя: ')
            self.assertIn(str(i + 1), sheet.cell(0, 3).text)
            self.assertEqual(len(sheet.cell(6, 0)._element.xpath('.//w:drawing')), 1)
        self.assertEqual(result.tables[-1].cell(0, 0).tables[0].cell(0, 3).text, 'Таблица — категория дефекта: ')


if __name__ == '__main__':
    unittest.main()
