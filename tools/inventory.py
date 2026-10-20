"""Generate static function/state inventory and sample manifest without importing GUI."""
import ast
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work' / 'inventory'
OUT.mkdir(parents=True, exist_ok=True)
source = (ROOT / 'vac/legacy.py').read_text(encoding='utf-8')
tree = ast.parse(source)
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
names = {n.name for n in functions}
state = set()
for n in tree.body:
    if isinstance(n, ast.Assign):
        state.update(t.id for t in n.targets if isinstance(t, ast.Name))
for f in functions:
    for n in ast.walk(f):
        if isinstance(n, ast.Global):
            state.update(n.names)
lines = ['# Карта функций и состояния', '',
         'Сгенерировано `python tools/inventory.py` из AST основного файла. '
         'Строки относятся к vac/legacy.py. Чтение/запись имён — синтаксические; '
         'изменение столбцов и вызовы методов включены в «возможные изменения». '
         'Это не анализ потока выполнения: aliases, сигналы Qt и переданные аргументы требуют просмотра кода.', '',
         '## Функции', '', '| Функция:строка | Назначение | Прямые вызовы функций приложения |',
         '| --- | --- | --- |']
for f in functions:
    calls = sorted({n.func.id for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names})
    desc = (ast.get_docstring(f) or 'См. тело функции').splitlines()[0].replace('|', '/')
    lines.append(f'| `{f.name}:{f.lineno}` | {desc} | {", ".join(calls)} |')
lines += ['', '## Глобальное состояние', '', '| Имя | Создание в модуле | Читают | Записывают имя / возможно меняют объект |', '| --- | --- | --- | --- |']
for name in sorted(state):
    init = [str(n.lineno) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    reads, writes = [], []
    for f in functions:
        nodes = list(ast.walk(f))
        if any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load) for n in nodes):
            reads.append(f.name)
        changed = any(isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store) for n in nodes)
        for n in nodes:
            target = n.value if isinstance(n, (ast.Subscript, ast.Attribute)) else None
            while isinstance(target, (ast.Subscript, ast.Attribute)):
                target = target.value
            if isinstance(target, ast.Name) and target.id == name:
                if isinstance(n.ctx, ast.Store) or isinstance(n, ast.Attribute):
                    changed = True
        if changed:
            writes.append(f.name)
    lines.append(f'| `{name}` | {", ".join(init) or "Возникает позже"} | {", ".join(reads)} | {", ".join(writes)} |')
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
print(f'Inventory: {len(functions)} functions, {len(state)} global names; {len(manifest)} sample files')
