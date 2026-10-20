"""Generate static function/state inventory and sample manifest without importing GUI."""
import ast
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work' / 'inventory'
OUT.mkdir(parents=True, exist_ok=True)
source = (ROOT / 'vac/controller.py').read_text(encoding='utf-8')
tree = ast.parse(source)
controller = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Controller')
functions = [n for n in controller.body if isinstance(n, ast.FunctionDef)]
names = {n.name for n in functions}
lines = ['# Карта функций и состояния', '',
         'Сгенерировано `python tools/inventory.py` из AST vac/controller.py. '
         'Состояние принадлежит экземпляру контроллера. Чтение и возможные изменения '
         'синтаксические; aliases и переданные аргументы требуют просмотра кода. '
         'Вычислительные функции находятся отдельно в vac/calculation.py.', '',
         '## Методы контроллера', '', '| Метод:строка | Назначение | Прямые вызовы методов |',
         '| --- | --- | --- |']
for f in functions:
    calls = sorted({n.func.attr for n in ast.walk(f) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == 'self' and n.func.attr in names})
    desc = (ast.get_docstring(f) or 'См. тело метода').splitlines()[0].replace('|', '/')
    lines.append(f'| `{f.name}:{f.lineno}` | {desc} | {", ".join(calls)} |')
lines += ['', '## Состояние экземпляра', '', '| Поле | Используют / возможно меняют |', '| --- | --- |']
state = {}
for f in functions:
    for n in ast.walk(f):
        if isinstance(n, ast.Attribute):
            field = ast.unparse(n)
            if field.startswith('self.state.'):
                state.setdefault(field, set()).add(f.name)
for field, methods in sorted(state.items()):
    lines.append(f'| `{field}` | {", ".join(sorted(methods))} |')
(OUT / 'function-state-map.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
manifest = []
for p in sorted((ROOT / 'пример').rglob('*')):
    if not p.is_file():
        continue
    data = p.read_bytes()
    entry = {'path': p.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if p.suffix == '.TWF':
        payload = data[130:]
        assert payload and len(payload) % 4 == 0
        values = struct.unpack('<' + str(len(payload) // 4) + 'i', payload)
        entry.update(format='TWF', samples=len(values), header_bytes=130)
    elif p.suffix == '.lvm':
        rows = [line.split('\t') for line in data.decode().splitlines() if line.strip()]
        assert rows and all(len(row) >= 3 for row in rows)
        depth = [float(row[1]) for row in rows]
        signal = [float(row[2]) for row in rows]
        entry.update(format='LVM', samples=len(rows), depth_min=min(depth), depth_max=max(depth), strictly_increasing=all(a < b for a, b in zip(depth, depth[1:])))
    manifest.append(entry)
assert sum(x.get('format') == 'LVM' for x in manifest) == 9
assert sum(x.get('format') == 'TWF' for x in manifest) == 13
(OUT / 'samples.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(f'Inventory: {len(functions)} methods, {len(state)} state attributes; {len(manifest)} sample files')
