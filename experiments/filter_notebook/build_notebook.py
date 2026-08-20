#!/usr/bin/env python3
"""Build (and optionally execute) the notebook from its paired `# %%` script.

The notebook of record is written as a plain Python script with cell markers, so
that every cell can be run and debugged headlessly before it ever becomes JSON:

    # %% [markdown]      -> a markdown cell (leading '# ' stripped from each line)
    # %%                 -> a code cell

Executing here rather than shipping an empty notebook matters: a reader opening
`pp4_learning_r.ipynb` sees the figures without owning a GPU, a Gazebo install, or the
capture.

There are six notebooks -- 1 and 2 work with a single camera, 3 and 4 repeat the same
two questions with all four, and 5 estimates R as a field over the floor -- and each is
built from a `# %%` script that calls
`notebook_model.py` and `notebook_views.py`. None carries an estimator or a figure of its
own.

    python3 build_notebook.py                     # build and execute all six
    python3 build_notebook.py --no-exec           # build only
    python3 build_notebook.py pp4_1_learning_r    # just one of them
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
NOTEBOOKS = ("pp4_1_learning_r", "pp4_2_the_offset",
             "pp4_3_learning_r_four_cameras", "pp4_4_the_offset_four_cameras",
             "pp4_5_the_r_field", "pp4_6_the_observation_function",
             "camera_localisation_from_scratch")

CELL_MARKER = re.compile(r"^#\s*%%(?:\s*\[(?P<kind>markdown|md|raw)\])?\s*(?P<title>.*)$")


def split_cells(text: str) -> list[tuple[str, str]]:
    """[(kind, source)] for every `# %%` cell in the script."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    body: list[str] = []
    for line in text.splitlines():
        match = CELL_MARKER.match(line)
        if match:
            if body:
                cells.append((kind, body))
            raw = match.group("kind")
            kind = "markdown" if raw in {"markdown", "md"} else (raw or "code")
            body = []
            continue
        body.append(line)
    if body:
        cells.append((kind, body))

    out = []
    for cell_kind, lines in cells:
        if cell_kind == "markdown":
            # markdown cells are written as comments in the script; unwrap them
            unwrapped = [re.sub(r"^#\s?", "", line) for line in lines]
            source = "\n".join(unwrapped).strip("\n")
        else:
            source = "\n".join(lines).strip("\n")
        if source.strip():
            out.append((cell_kind, source))
    return out


def build_one(stem: str, execute: bool) -> int:
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell, new_raw_cell

    source, target = HERE / f"{stem}.py", HERE / f"{stem}.ipynb"
    if not source.is_file():
        print(f"missing {source}", file=sys.stderr)
        return 1
    print(f"\n=== {stem} ===")

    makers = {"code": new_code_cell, "markdown": new_markdown_cell, "raw": new_raw_cell}
    cells = [makers[kind](text) for kind, text in split_cells(source.read_text("utf-8"))]
    notebook = new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"{len(cells)} cells ({n_code} code, {len(cells) - n_code} markdown)")

    if execute:
        from nbclient import NotebookClient

        print("executing...")
        client = NotebookClient(
            notebook, timeout=1800, kernel_name="python3",
            resources={"metadata": {"path": str(HERE)}},
            allow_errors=False,
        )
        client.execute()
        print("all cells ran without raising")

    nbformat.write(notebook, target)
    print(f"wrote {target.relative_to(target.parents[2])}")
    return 0


def build(stems, execute: bool) -> int:
    return max(build_one(stem, execute) for stem in stems)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", default=None,
                        help=f"which to build (default: all of {', '.join(NOTEBOOKS)})")
    parser.add_argument("--no-exec", action="store_true", help="build JSON without running")
    args = parser.parse_args()
    chosen = args.notebooks or list(NOTEBOOKS)
    unknown = [n for n in chosen if n not in NOTEBOOKS]
    if unknown:
        parser.error(f"unknown notebook(s): {', '.join(unknown)}")
    raise SystemExit(build(chosen, execute=not args.no_exec))
