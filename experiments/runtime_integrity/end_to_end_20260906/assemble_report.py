"""Assemble audit-owned Markdown fragments into the final report."""
from __future__ import annotations

import csv
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[2]
base = (here / "report_base.md").read_text()
inventory = (here / "component_inventory.md").read_text().rstrip()
documentation = (here / "documentation.md").read_text().rstrip()

with (here / "matrix.tsv").open(newline="") as stream:
    rows = list(csv.reader(stream, delimiter="\t"))

def safe(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")

header, *body = rows
table = ["## Master acceptance matrix", "", "| " + " | ".join(map(safe, header)) + " |"]
table.append("|" + "|".join("---" for _ in header) + "|")
table.extend("| " + " | ".join(map(safe, row)) + " |" for row in body)

result = base.replace("<!-- COMPONENT_INVENTORY -->", inventory)
result = result.replace("<!-- ACCEPTANCE_MATRIX -->", "\n".join(table))
result = result.replace("<!-- DOCUMENTATION_CONTRADICTIONS -->", documentation)
if "<!--" in result:
    raise RuntimeError("unresolved report placeholder")

target = root / "docs/module_audits/15_end_to_end_acceptance.md"
target.write_text(result.rstrip() + "\n")
print(target)
