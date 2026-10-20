"""Generate the primary form with the existing responsive layout hook."""
import argparse
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
assert version('PyQt5') == '5.15.11', 'Use pinned PyQt5 5.15.11'
with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / 'VAC_dialog.py'
    subprocess.run([sys.executable, '-m', 'PyQt5.uic.pyuic', '-x',
                    'VAC_dialog.ui', '-o', str(output)], cwd=ROOT, check=True)
    source = output.read_text(encoding='utf-8')
anchor = '        QtCore.QMetaObject.connectSlotsByName(MainWindow)'
assert source.count(anchor) == 1
source = source.replace(anchor, '        from responsive_layout import apply_responsive_layout\n'
                        '        apply_responsive_layout(self, MainWindow)\n' + anchor)
target = ROOT / 'VAC_dialog.py'
if args.check:
    assert target.read_text(encoding='utf-8') == source, 'Run python tools/generate_ui.py'
else:
    target.write_text(source, encoding='utf-8')
