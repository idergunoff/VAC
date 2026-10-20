"""Check inert imports and all public launchers from a foreign working directory.

Run with the source checkout or an installed wheel on Python's search path.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='VAC пакет с пробелами ') as tmp:
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(Path(tmp) / 'mpl'), PYTHONDONTWRITEBYTECODE='1')
    # An installed-wheel check may override PYTHONPATH to exclude the checkout.
    env['PYTHONPATH'] = os.environ.get('VAC_TEST_PACKAGE_PATH', str(ROOT))
    def check(code):
        subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp, env=env,
                       check=True, timeout=40)
    check('''
from pathlib import Path
import sys
before = set(Path('.').rglob('*'))
import vac, vac.app, vac.reference, VAC_python, VAC_python_mini, standard
assert 'PyQt5' not in sys.modules
assert set(Path('.').rglob('*')) == before
assert standard.good_new is vac.reference.good_new
''')
    for launcher in ('vac', 'VAC_python', 'VAC_python_mini'):
        check(f'''
import runpy
from pathlib import Path
from PyQt5 import QtCore, QtWidgets
app_type = QtWidgets.QApplication
assert app_type.instance() is None
import vac.legacy as legacy
assert app_type.instance() is None
assert not hasattr(legacy, 'ui')
from vac.resources import template_path
import docx
assert len(docx.Document(str(template_path())).tables) == 2
assert not Path('template.docx').exists()
called = []
def event_loop(self):
    called.append(True)
    assert legacy.MainWindow.isVisible()
    assert legacy.ui.graphicsView is not None
    assert legacy.proxy is not None
    QtCore.QTimer.singleShot(0, self.quit)
    return original()
original = app_type.exec_
app_type.exec_ = event_loop
try:
    runpy.run_module({launcher!r}, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
assert called == [True]
assert not list(Path('.').glob('*.txt'))
assert not list(Path('.').glob('*.png'))
assert not list(Path('.').glob('*.docx'))
assert not list(Path('.').glob('*.vac'))
''')
print('PASS inert imports, resources, and three launchers from foreign CWD')
