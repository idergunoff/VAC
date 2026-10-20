"""Replay isolated cases against documented outcomes, with bounded subprocesses."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--filter', default='', help='Run only keys containing this text')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
expected = json.loads((root / 'docs/stage-1/cases.json').read_text())
for key, reference in expected.items():
    if args.filter not in key:
        continue
    device, case = key.split(':')
    directory = out / (device + '-' + case)
    try:
        process = subprocess.run([sys.executable, str(root / 'tools/regression.py'), 'record', '--device', device, '--case', case, '--output', str(directory)], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
        if (directory / 'manifest.json').exists():
            m = json.loads((directory / 'manifest.json').read_text())
            result = {'steps': m['steps'], 'qt_errors': m['qt_errors']}
        else:
            result = {'exit': process.returncode}
    except subprocess.TimeoutExpired:
        result = {'timeout': True}
    assert result == reference, (key, reference, result)
    print('PASS expected legacy outcome', key, flush=True)
