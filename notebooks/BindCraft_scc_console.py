#!/usr/bin/env python3
"""Launch with: python -i notebooks/BindCraft_scc_console.py"""
import ast as _ast
from pathlib import Path as _Path
import re as _re

_CELL_FILE = _Path(__file__).with_name("BindCraft_scc_cells.py")


def _read_cells():
    # Reload each time so edits made in an editor are picked up immediately.
    blocks = _re.split(r"(?m)^# %% ", _CELL_FILE.read_text())[1:]
    return [block.partition("\n") for block in blocks]


def cells():
    """List the available cells."""
    for number, (title, _, source) in enumerate(_read_cells()):
        print(f"{number:2d}: {title}")


def run_cell(number, **overrides):
    """Execute a cell in this console, optionally overriding its assigned settings.

    Examples:
        run_cell(1)
        run_cell(3, binder_name="LOV", starting_pdb="/path/to/target.pdb")
        run_cell(6, RUN_DESIGN=True)
        run_cell(8, STOP_RUN=True)
    """
    blocks = _read_cells()
    if type(number) is not int or not 0 <= number < len(blocks):
        raise ValueError("Use a cell number printed by cells().")
    title, _, source = blocks[number]
    tree = _ast.parse(source, filename=str(_CELL_FILE))
    assigned = {target.id for node in tree.body if isinstance(node, _ast.Assign)
                for target in node.targets if isinstance(target, _ast.Name)}
    if overrides.keys() - assigned:
        raise ValueError("Overrides must name settings assigned in this cell: " +
                         str(sorted(overrides.keys() - assigned)))
    # Keep an overridden assignment from resetting a value supplied by the user.
    tree.body = [node for node in tree.body if not (
        isinstance(node, _ast.Assign) and len(node.targets) == 1 and
        isinstance(node.targets[0], _ast.Name) and node.targets[0].id in overrides)]
    globals().update(overrides)
    print("Running:", title)
    exec(compile(tree, str(_CELL_FILE), "exec"), globals())


if __name__ == "__main__":
    print("SCC cell console. Run cells in order with run_cell(1), run_cell(2), etc.")
    print("Use run_cell(6, RUN_DESIGN=True) to start; run_cell(8, STOP_RUN=True) to stop.")
    print("Use cells() to show this list again. Keep the allocation and console open.")
    cells()
