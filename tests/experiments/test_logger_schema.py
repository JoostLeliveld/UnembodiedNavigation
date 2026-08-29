"""The experiment logger's CSV header and data row must stay the same length.

``experiment_logger`` writes its header and its per-timestep row as two separate
list literals ~1700 lines apart. Insert a column into one and not the other and
nothing raises -- every column after the insertion point silently shifts, so a
whole campaign's CSVs decode into the wrong fields -- with no error, on every run of
a campaign that takes hours.

Parsed with ``ast`` rather than imported, so no ROS runtime is needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


LOGGER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "experiments" / "experiments" / "nodes" / "experiment_logger.py"
)


def writerow_literal_lengths() -> dict[str, list[tuple[int, int]]]:
    """Map writer attribute -> [(lineno, n_elements)] for literal-list writerows."""
    tree = ast.parse(LOGGER_PATH.read_text())
    out: dict[str, list[tuple[int, int]]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "writerow"
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            out.setdefault(ast.unparse(node.func.value), []).append(
                (node.lineno, len(node.args[0].elts))
            )
    return out


def test_logger_source_is_present():
    assert LOGGER_PATH.is_file(), LOGGER_PATH


@pytest.mark.parametrize(
    "writer",
    ["self.writer", "self.perception_writer", "self.plan_writer"],
)
def test_header_and_row_lengths_agree(writer):
    calls = writerow_literal_lengths().get(writer)
    assert calls, f"no literal writerow found for {writer}"
    lengths = {n for _, n in calls}
    assert len(lengths) == 1, (
        f"{writer} writes rows of differing widths {sorted(lengths)} at lines "
        f"{[ln for ln, _ in calls]} -- a column was added to the header or the "
        f"row but not both, which silently misaligns every later column."
    )


def test_experiment_csv_still_carries_the_correction_diagnostic_columns():
    """The shared correction chain's reason codes must reach the CSV.

    A refused correction must say why it was refused. Without a reason code a gate
    that is silently doing nothing and a gate that is working look identical in the
    log, which is how a broken gate survives a whole campaign.
    """
    header_src = LOGGER_PATH.read_text()
    for column in (
        "pixel_corr_reject_reason_code",
        "pixel_corr_reject_reason",
        "pixel_corr_nis",
        "pixel_corr_xy_update_norm_m",
        "pixel_corr_measurement_space",
        "pixel_corr_predict_clipped_m",
    ):
        assert f"'{column}'" in header_src, column


def test_reject_reason_decoder_covers_every_shared_chain_code():
    """The logger must not decode a live reason code as 'unknown'."""
    import sys

    sys.path.insert(0, str(LOGGER_PATH.parents[3]))   # src/planning sits alongside
    from planning.core import belief_correction as bc

    tree = ast.parse(LOGGER_PATH.read_text())
    decoded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_pixel_correction_reject_reason_name":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for key in sub.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, int):
                            decoded.add(key.value)
    assert decoded, "could not parse the reject-reason decoder"
    live = {int(code) for code in bc.REJECT_CODES.values()}
    missing = live - decoded
    assert not missing, (
        f"reject codes {sorted(missing)} are emitted by the correction chain but "
        f"decode as 'unknown' in the CSV"
    )
