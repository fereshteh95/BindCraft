# Paste into a new cell in BindCraft_local.ipynb after running its
# runtime, settings, and settings-validation cells.

binder_seq = "PASTE_YOUR_CHAIN_B_SEQUENCE_HERE"


def test_binder(target_pdb, name, chain="A", use_multimer=False):
    if "run_python" not in globals() or "advanced_settings" not in globals():
        raise RuntimeError("Run the notebook's runtime, settings, and validation cells first.")

    sequence = "".join(binder_seq.split()).upper()
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("Replace binder_seq with the binder's amino-acid sequence only.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Use letters, numbers, underscores, dots, or hyphens for name.")
    target = local_path(target_pdb)
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("Provide a PDB file; convert mmCIF to PDB first.")

    output = local_path(target_settings["design_path"]) / "binder_tests" / (
        f"{name}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    )
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        target_pdb=runtime_path(target), output=runtime_path(output),
        name=name, chain=chain, sequence=sequence, use_multimer=use_multimer,
        recycles=advanced_settings["num_recycles_validation"],
        data_dir=runtime_path(advanced_settings["af_params_dir"]),
    )

    print("Saving predictions to:", output, flush=True)
    print("Running in WSL/Linux. This cell stays busy; logs appear when it finishes.", flush=True)
    log = run_python(r'''
import json
import sys
from pathlib import Path
import pandas as pd
from colabdesign import mk_afdesign_model, clear_mem

cfg = json.loads(sys.argv[1])
output = Path(cfg["output"])
clear_mem()
model = mk_afdesign_model(
    protocol="binder",
    num_recycles=cfg["recycles"],
    data_dir=cfg["data_dir"],
    use_multimer=cfg["use_multimer"],
    use_initial_guess=False,
    use_initial_atom_pos=False,
)
if not model._model_names:
    raise RuntimeError("No AlphaFold model weights were loaded. Check data_dir.")
model.prep_inputs(
    pdb_filename=cfg["target_pdb"],
    chain=cfg["chain"],
    binder_len=len(cfg["sequence"]),
    rm_target_seq=False,
    rm_target_sc=False,
)

rows = []
for model_num, model_name in enumerate(model._model_names):
    print(f"Predicting {model_name}...", flush=True)
    model.predict(
        seq=cfg["sequence"], models=[model_num],
        num_recycles=cfg["recycles"], verbose=False,
    )
    metrics = model.aux["log"]
    rows.append({
        "model": model_num + 1, "model_name": model_name,
        "pLDDT": float(metrics["plddt"]),
        "pTM": float(metrics["ptm"]),
        "i_pTM": float(metrics["i_ptm"]),
        "pAE": float(metrics["pae"]),
        "i_pAE": float(metrics["i_pae"]),
    })
    model.save_pdb(str(output / f"{cfg['name']}_{model_name}.pdb"))
    table = pd.DataFrame(rows)
    table.to_csv(output / "scores.csv", index=False)
    (output / "scores.json").write_text(json.dumps(rows), encoding="utf-8")
    (output / "scores.html").write_text(
        table.to_html(index=False, float_format=lambda x: f"{x:.4f}"), encoding="utf-8"
    )
print(f"Finished: {len(rows)} predictions saved.", flush=True)
''', args=[json.dumps(config)], timeout=None)
    print(log)
    display(HTML((output / "scores.html").read_text(encoding="utf-8")))
    return json.loads((output / "scores.json").read_text(encoding="utf-8"))


# False: two AF2 models with target templates, matching your original setup.
# True: five AF-Multimer models.
results = test_binder(TARGET_PDB, "LOV_test", chain="A", use_multimer=False)
