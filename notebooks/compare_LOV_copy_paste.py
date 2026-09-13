# Paste into BindCraft_local.ipynb after its setup/settings-validation cells.
# This launches a new comparison. Use show_comparison(comparison_dir) alone
# to redisplay an existing comparison without rerunning predictions.
import sys

if str(REPO_ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "notebooks"))
from compare_binders_light_dark import run_comparison, show_comparison

comparison_dir = run_comparison(
    globals(),
    results_dir=target_settings["design_path"],
    targets={
        "Light": TARGET_PDB.parent / "2V1B.pdb",
        "Dark": TARGET_PDB.parent / "2V1A.pdb",
    },
    ranks=(1, 2),
    chain="A",
    use_multimer=False,  # Two AF2 models per binder/condition: eight predictions.
    seeds=(0,),
    recycles=advanced_settings["num_recycles_validation"],
)
show_comparison(comparison_dir)
