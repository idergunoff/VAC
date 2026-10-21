"""Measure production archive operations and headless processing on both examples."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vac.storage import Archive
from vac.services.archive import import_directory, open_measurements
from vac.services.analysis import analyze
from vac.domain.models import ProcessingParameters

# Engineering budgets declared before measurement; storage targets from stage 6.
BUDGET_MS = dict(import_ms=2000, open_ms=1000, processing_ms=5000)


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - start) * 1000


def measure():
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  budgets_ms=BUDGET_MS, repeats=3,
                  timing='median; warm process/cache, includes commits; processing excludes GUI', datasets={})
    with tempfile.TemporaryDirectory(prefix='VAC reliability ') as scratch:
        for equipment, folder in [('new', 'новое оборудование'), ('old', 'старое оборудование')]:
            baseline = ROOT / 'tests/baselines/stage-1' / equipment
            settings = json.loads((baseline / '01-load/settings.json').read_text())
            suffix = '' if equipment == 'new' else '_old'
            params = ProcessingParameters(settings['doubleSpinBox_int_min'], settings['doubleSpinBox_int_max'],
                settings['doubleSpinBox_coeff_norm' + suffix], settings['doubleSpinBox_coeff_func' + suffix],
                settings['spinBox_coeff_dif_res' + suffix], settings['spinBox_mean_win' + suffix])
            order = json.loads((baseline / 'manifest.json').read_text())['read_order'][:9 if equipment == 'new' else 13]
            root = ROOT / 'пример' / folder
            directories = {}
            for name in order:
                path = Path(name)
                directories.setdefault(path.parent, []).append(path.name)
            traversal = patch('vac.io.lvm.os.walk', return_value=[(str(root / p), [], n) for p, n in directories.items()]) if equipment == 'new' else patch('vac.io.twf.os.listdir', return_value=order)
            values = {key: [] for key in BUDGET_MS}
            with Archive(Path(scratch) / (equipment + '.sqlite')) as db, traversal:
                well = db.create_well('benchmark')
                for _ in range(3):
                    imported, elapsed = timed(lambda: import_directory(db, well, root, equipment, mean=settings['spinBox_mean']))
                    values['import_ms'].append(elapsed)
                    source, elapsed = timed(lambda: open_measurements(db, imported.study_id))
                    values['open_ms'].append(elapsed)
                    result, elapsed = timed(lambda: analyze(source, params))
                    values['processing_ms'].append(elapsed)
                report['datasets'][equipment] = dict(
                    median_ms={key: round(statistics.median(v), 3) for key, v in values.items()},
                    samples_ms=values,
                    within_budget={key: statistics.median(v) <= BUDGET_MS[key] for key, v in values.items()},
                    source_count=len(imported.accepted), channels=len(source.original.columns)-1,
                    samples_per_channel=len(source.original), classification_errors=len(result.classification_errors))
    assert 'PyQt5' not in sys.modules
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = measure()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(args.output)
