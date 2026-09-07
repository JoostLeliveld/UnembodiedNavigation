"""Retain an auditable static index; report supplies reachability/lock analysis.

Assignments and calls are syntactic inventories, not a call-graph proof. Parsing
frozen files keeps line references stable while other investigations edit source.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = [
    'src/planning/planning/nodes/unicycle_planner_node.py',
    'src/planning/planning/nodes/efe_agent_node.py',
    'src/experiments/experiments/nodes/goal_mission_node.py',
]
report = {}
for name in FILES:
    tree = ast.parse((HERE / 'source_snapshot' / name).read_text())
    methods = {}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            writes, locks, calls, clocks = [], [], [], []
            for n in ast.walk(fn):
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for target in targets:
                        for leaf in ast.walk(target):
                            if isinstance(leaf, ast.Attribute) and isinstance(leaf.ctx, ast.Store):
                                text = ast.unparse(leaf)
                                if text.startswith('self.'):
                                    writes.append({'line': n.lineno, 'target': text})
                if isinstance(n, ast.With):
                    for item in n.items:
                        locks.append({'line': n.lineno, 'context': ast.unparse(item.context_expr)})
                if isinstance(n, ast.Call):
                    call = ast.unparse(n.func)
                    if call.startswith('self.'):
                        calls.append({'line': n.lineno, 'call': call})
                    if any(s in call for s in ('now', 'perf_counter', 'monotonic', 'Time.', 'Clock')):
                        clocks.append({'line': n.lineno, 'expression': ast.unparse(n)})
            methods[f'{cls.name}.{fn.name}'] = {
                'line': fn.lineno, 'end_line': fn.end_lineno,
                'decorators': [ast.unparse(n) for n in fn.decorator_list],
                'assignments': sorted(writes, key=lambda x: x['line']),
                'locks': sorted(locks, key=lambda x: x['line']),
                'calls': sorted(calls, key=lambda x: x['line']),
                'clocks': sorted(clocks, key=lambda x: x['line']),
            }
    report[name] = methods
(HERE / 'ownership_map.json').write_text(json.dumps(report, indent=2) + '\n')
for name, methods in report.items():
    print(name, len(methods), 'methods')
    for method, data in methods.items():
        if any(w['target'] in ('self.belief_m', 'self.belief_S', 'self.belief_stamp')
               for w in data['assignments']):
            print('  recursive assignment:', method, 'line', data['line'])
