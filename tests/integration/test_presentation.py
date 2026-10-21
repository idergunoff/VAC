"""Template pagination and trusted legacy compatibility at the presentation boundary."""
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vac.io import legacy_options
from vac.reporting import word, figures
from vac.resources import template_path
from vac import reference


class PresentationTests(unittest.TestCase):
    def test_legacy_settings_exact_fields_and_pickle_bytes(self):
        for equipment in ('new', 'old'):
            path = ROOT / 'tests/fixtures/current' / equipment / 'settings.vac'
            expected = pickle.loads(path.read_bytes())  # repository-owned trusted fixture
            actual = legacy_options.load_trusted(path)
            self.assertEqual(actual, expected)
            with tempfile.TemporaryDirectory(prefix='VAC настройки с пробелами ') as tmp:
                saved = Path(tmp) / 'настройки.vac'
                legacy_options.save(saved, actual)
                self.assertEqual(saved.read_bytes(), path.read_bytes())

    def test_reference_values_lengths_and_order(self):
        expected = {
            'good_new': (5138, 'ab2a80c24522e4a584cc6d4c212f9190bb7d59b36def0c8a36a4df1db0310ae8'),
            'good_old': (4437, '6d5cad27771e2865fc23323b69deca19c1e6abe035ae9a5f078fcd690a75ccb5'),
            'bad_new': (5138, '340ab79d6ec74d333a23322aefa43e49e90cddfe6b226565f280aa5ac8200dbc'),
            'bad_old': (4437, '61a500a84c73edf0aec95c21bfc5c2e62f2401384d25bde212db89e662693162'),
            'depth_new': (5138, '3b1ae897561793259976099b3df5e949aee94c6a49cefd4f1b56e4353f9afafc'),
            'depth_old': (4437, 'b7848723a7e476939b7529dd6c90030ef899f0034758a42da3de3a34a2b3f64e'),
        }
        self.assertEqual(set(reference.__all__), set(expected))
        for name, (length, digest) in expected.items():
            values = np.asarray(getattr(reference, name), dtype='<f8')
            self.assertEqual(values.shape, (length,))
            self.assertEqual(hashlib.sha256(values.tobytes()).hexdigest(), digest)

    def test_word_and_print_pages_at_400_and_800_boundaries(self):
        # Test the actual pagination triggers, both report interval conventions,
        # and every pipe wording. Scalar values enter without a Qt window.
        values = {
            'lineEdit_number_skv.text': 'Тест 17',
            'comboBox_pipe_type.currentText': 'Кондуктор',
            'lineEdit_pipe_diameter.text': '245',
            'lineEdit_client.text': 'Заказчик',
            'lineEdit_client2.text': 'Месторождение',
            'lineEdit_interpreter.text': 'Исполнитель',
            'dateEdit.text': '2026-10-19',
            'doubleSpinBox_int_max.value': 1200.,
            'checkBox_golfstrim.checkState': 0,
        }
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(prefix='VAC отчёт с пробелами ') as tmp:
            try:
                os.chdir(tmp)
                for maximum, pages in ((400, 1), (401, 2), (800, 2), (801, 3)):
                    with self.subTest(maximum=maximum), patch.object(plt.Figure, 'show'):
                        plt.close('all')
                        depth = np.arange(maximum + 1, dtype=float)
                        table = pd.DataFrame({'depth': depth, 'corr_sig': np.sin(depth / 10),
                                              'quality': (depth.astype(int) // 50) % 4})
                        # Avoid the known legacy report bug: a new interval at
                        # the last point has no allocated row in the template.
                        table.loc[maximum, 'quality'] = table.loc[maximum - 1, 'quality']
                        before = table.copy(deep=True)
                        for image in Path('.').glob('temp*.png'):
                            image.unlink()
                        figures.cementogramma(table, -1.2, 1.2, -.3, .3, 'Тест')
                        self.assertEqual(len(list(Path('.').glob('temp*.png'))), pages)
                        self.assertEqual(len(plt.get_fignums()), pages)
                        self.assertTrue(all(len(plt.figure(n).axes) == 4 for n in plt.get_fignums()))
                        # The renderer receives already computed classes and never changes them.
                        pd.testing.assert_frame_equal(table, before)
                        for mode in (0, 2):
                            values['checkBox_golfstrim.checkState'] = mode
                            for pipe in ('Кондуктор', 'Колонна', 'Тех. колонна', 'Направление'):
                                values['comboBox_pipe_type.currentText'] = pipe
                                document = word.build_report(table, '55 кГц', values, template_path())
                                self.assertEqual(len(document.tables), pages + 1)
                                self.assertEqual(document.tables[0].rows[0].cells[0].paragraphs[0].runs[3].text, 'Тест 17')
                                self.assertIn('Заказчик', document.paragraphs[6].text)
                                for sheet in document.tables[:pages]:
                                    self.assertEqual(len(sheet.rows[6].cells[0]._element.xpath('.//w:drawing')), 1)
                        plt.close('all')
            finally:
                plt.close('all')
                os.chdir(previous)

    def test_reporting_imports_without_qt(self):
        subprocess.run([sys.executable, '-c', "import sys; import vac.reporting.word, vac.reporting.figures; assert not any(n.startswith(('PyQt5', 'pyqtgraph')) for n in sys.modules)"],
                       cwd=ROOT, env=dict(os.environ, MPLBACKEND='Agg'), check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
