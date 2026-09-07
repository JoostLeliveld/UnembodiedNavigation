"""Load the preserved investigation-08 runtime modules in an isolated probe process."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

SNAPSHOT=Path(__file__).with_name('08_source_snapshot')
manifest=json.loads((SNAPSHOT/'manifest.json').read_text())
for relative,expected in manifest['sources'].items():
    if hashlib.sha256((SNAPSHOT/relative).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('changed audit baseline: '+relative)
for name,relative in [
    ('planning.core.casadi_efe','src/planning/planning/core/casadi_efe.py'),
    ('planning.planners.base_planner','src/planning/planning/planners/base_planner.py'),
    ('planning.nodes.unicycle_planner_node','src/planning/planning/nodes/unicycle_planner_node.py'),
    ('planning.nodes.efe_agent_node','src/planning/planning/nodes/efe_agent_node.py')]:
    parent,leaf=name.rsplit('.',1)
    package=importlib.import_module(parent)
    spec=importlib.util.spec_from_file_location(name,SNAPSHOT/relative)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    setattr(package,leaf,module)
    spec.loader.exec_module(module)
