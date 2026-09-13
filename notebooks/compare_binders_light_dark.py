"""Compare saved BindCraft binder sequences using matched target conditions.

The public helpers run in the notebook kernel; prediction and plotting run in
the notebook's existing Linux environment. Importing this file starts no work.
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _ca_residues(path, chain):
    residues = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if (line.startswith("ATOM  ") and line[21] == chain
                and line[12:16].strip() == "CA"):
            residues.setdefault(line[22:27].strip(), line[17:20].strip())
    if not residues:
        raise ValueError(f"No protein CA records for chain {chain} in {path}")
    return residues


def run_comparison(runtime, results_dir="results/LOV_local", targets=None,
                   ranks=(1, 2), chain="A", use_multimer=False,
                   seeds=(0,), recycles=3):
    """Run all selected sequences against both conditions and return output Path.

    Pass globals() from BindCraft_local.ipynb as runtime. Results and inputs are
    saved in a new directory on each call; existing design results are retained.
    """
    needed = {"local_path", "runtime_path", "linux_command", "LINUX_PYTHON", "REPO_ROOT"}
    if needed - runtime.keys():
        raise RuntimeError("Run the notebook's runtime/setup cell first.")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("seeds must contain nonnegative 32-bit integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct seeds.")
    if type(recycles) is not int or recycles < 0:
        raise ValueError("recycles must be a nonnegative integer.")
    if len(chain) != 1:
        raise ValueError("Select one PDB target chain, for example A.")
    if len(set(ranks)) != len(ranks) or not ranks:
        raise ValueError("Select distinct accepted-design ranks.")

    local_path, runtime_path = runtime["local_path"], runtime["runtime_path"]
    design_dir = local_path(results_dir)
    if (design_dir / ".bindcraft-notebook.lock").exists():
        raise RuntimeError("The main BindCraft campaign has a run lock. Check its status first.")
    scores_file = design_dir / "final_design_stats.csv"
    with scores_file.open(newline="", encoding="utf-8-sig") as handle:
        accepted = list(csv.DictReader(handle))
    binders = []
    for rank in ranks:
        matches = [row for row in accepted if int(row["Rank"]) == rank]
        if len(matches) != 1:
            raise ValueError(f"Expected one accepted design with rank {rank}; found {len(matches)}.")
        row = matches[0]
        sequence = "".join(row["Sequence"].split()).upper()
        if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"Invalid amino-acid sequence for {row['Design']}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["Design"]):
            raise ValueError("Invalid design name in the scores CSV.")
        binders.append(dict(rank=rank, design=row["Design"],
                            label=f"Binder {rank}", sequence=sequence))

    if targets is None:
        target_parent = local_path("../LOV")
        targets = {"Light": target_parent / "2V1B.pdb", "Dark": target_parent / "2V1A.pdb"}
    if set(targets) != {"Light", "Dark"}:
        raise ValueError("Provide exactly the Light and Dark target paths.")
    paths = {condition: local_path(targets[condition]) for condition in ("Light", "Dark")}
    residues = {condition: _ca_residues(path, chain) for condition, path in paths.items()}
    if residues["Light"] != residues["Dark"]:
        raise ValueError("Target residue identities/numbering differ; align equivalent residues before comparing.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output = design_dir / "light_dark_comparison" / stamp
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(scores_file, inputs / "accepted_designs.csv")
    target_inputs = {}
    for condition, source in paths.items():
        snapshot = inputs / f"{condition}_{source.name}"
        shutil.copyfile(source, snapshot)
        target_inputs[condition] = dict(
            pdb=runtime_path(snapshot), source=str(source),
            sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            residues=len(residues[condition]),
        )
    model_names = ([f"model_{n}_multimer_v3" for n in range(1, 6)] if use_multimer
                   else ["model_1_ptm", "model_2_ptm"])
    config = dict(
        output=runtime_path(output), binders=binders, targets=target_inputs,
        chain=chain, use_multimer=bool(use_multimer), model_names=model_names,
        seeds=list(seeds), recycles=recycles,
        data_dir=runtime_path(runtime["REPO_ROOT"]),
        expected_predictions=len(binders) * 2 * len(model_names) * len(seeds),
    )
    _write_json(output / "config.json", config)
    print(f"Output: {output}", flush=True)
    print(f"{len(binders)} binders x 2 conditions x {len(model_names)} models "
          f"x {len(seeds)} seeds = {config['expected_predictions']} predictions", flush=True)
    print("The cell finishes automatically after predictions, tables, and plots are saved.", flush=True)
    worker = local_path("notebooks/compare_binders_light_dark.py")
    command = runtime["linux_command"]([
        runtime["LINUX_PYTHON"], "-u", runtime_path(worker),
        "--config", runtime_path(output / "config.json"),
    ])
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        **runtime.get("HOST_PROCESS_OPTIONS", {}),
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait()
    except KeyboardInterrupt:
        (output / "STOP").touch()
        print("Stop requested. The Linux worker will exit after its current prediction; "
              f"check {output / 'status.json'} before starting another GPU job.", flush=True)
        raise
    if returncode:
        raise RuntimeError(f"Comparison exited with {returncode}. See {output / 'run.log'}")
    return output


def show_comparison(output):
    """Display the saved report without launching or rerunning predictions."""
    from IPython.display import HTML, display
    output = Path(output)
    state = json.loads((output / "status.json").read_text(encoding="utf-8"))
    print(f"Status: {state['status']} | "
          f"{state['completed']}/{state['expected']} predictions | {output}")
    report = output / "report.html"
    if report.is_file():
        display(HTML(report.read_text(encoding="utf-8")))
    else:
        print("The report is generated after all predictions finish. See run.log for progress.")


def make_report(output, cfg):
    """Summarize complete paired results; import plotting libraries only in Linux."""
    import base64
    import html
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [json.loads(path.read_text()) for path in sorted((output / "records").glob("*.json"))]
    rows = pd.DataFrame(records)
    if len(rows) != cfg["expected_predictions"]:
        raise ValueError("All expected predictions must finish before creating the comparison report.")
    order = ["rank", "condition", "model_name", "seed"]
    rows = rows.sort_values(order).reset_index(drop=True)
    metrics = ["pLDDT", "pTM", "i_pTM", "pAE", "i_pAE", "pAE_A", "i_pAE_A"]
    keys = ["rank", "binder", "design", "condition"]
    means = rows.groupby(keys, sort=True)[metrics].mean().add_suffix("_mean")
    summary = rows.groupby(keys, sort=True).size().rename("n").to_frame().join(means).reset_index()
    pair_keys = ["rank", "binder", "design", "model_name", "seed"]
    light = rows[rows.condition == "Light"].set_index(pair_keys)[metrics]
    dark = rows[rows.condition == "Dark"].set_index(pair_keys)[metrics]
    if not light.index.equals(dark.index):
        raise ValueError("Light and Dark predictions do not form complete matched pairs.")
    differences = (light - dark).add_prefix("delta_").reset_index()
    delta_columns = ["delta_" + metric for metric in metrics]
    deltas = differences.groupby(["rank", "binder", "design"])[delta_columns].mean().reset_index()
    rows.to_csv(output / "all_predictions.csv", index=False)
    summary.to_csv(output / "condition_summary.csv", index=False)
    differences.to_csv(output / "paired_light_minus_dark.csv", index=False)
    deltas.to_csv(output / "mean_light_minus_dark.csv", index=False)

    palette = {"Light": "#D99012", "Dark": "#355CA8"}
    plot_specs = [("i_pTM", "Interface pTM (higher is better)", (0, 1)),
                  ("i_pAE_A", "Interface PAE / angstrom (lower is better)", (0, 31)),
                  ("pLDDT", "Binder pLDDT (higher is better)", (0, 1)),
                  ("pTM", "Complex pTM (higher is better)", (0, 1)),
                  ("pAE_A", "Binder + interface PAE / angstrom (lower is better)", (0, 31))]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), layout="constrained")
    binder_labels = [binder["label"] for binder in cfg["binders"]]
    for ax, (metric, title, limits) in zip(axes.flat, plot_specs):
        for i, label in enumerate(binder_labels):
            subset = rows[rows.binder == label]
            paired = subset.pivot(index=["model_name", "seed"], columns="condition", values=metric)
            for _, pair in paired.iterrows():
                ax.plot([i - .19, i + .19], [pair.Light, pair.Dark], color="#727272", lw=.8, alpha=.5, zorder=3)
            for condition, offset in (("Light", -.19), ("Dark", .19)):
                values = subset[subset.condition == condition][metric]
                ax.bar(i + offset, values.mean(), width=.34, color=palette[condition], alpha=.65, zorder=2)
                ax.scatter(np.full(len(values), i + offset), values, s=32,
                           color=palette[condition], edgecolor="white", linewidth=.6, zorder=4)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(binder_labels)), binder_labels)
        ax.set_ylim(*limits)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    note_ax = axes.flat[-1]
    note_ax.axis("off")
    from matplotlib.patches import Patch
    note_ax.legend(handles=[Patch(color=palette[c], label=c + (" / 2V1B" if c == "Light" else " / 2V1A"))
                            for c in ("Light", "Dark")], loc="upper left", frameon=False)
    note_ax.text(0, .72, "Bars: mean across model/seed predictions\n"
                 "Dots: individual predictions\nLines: matched model and seed\n\n"
                 f"Models per condition: {len(cfg['model_names'])}\n"
                 f"Seeds: {cfg['seeds']} | Recycles: {cfg['recycles']}\n\n"
                 "Confidence scores do not measure affinity.\n"
                 "Protein templates represent the conditions;\n"
                 "FMN photochemistry is not modeled.", va="top", linespacing=1.5, fontsize=10)
    fig.suptitle("LOV binder predictions: light versus dark", fontsize=17)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)

    display_metrics = ["pLDDT", "pTM", "i_pTM", "pAE_A", "i_pAE_A"]
    mean_display = summary[["binder", "condition", "n"] + [m + "_mean" for m in display_metrics]]
    delta_display = deltas[["binder"] + ["delta_" + m for m in display_metrics]]
    labels = pd.DataFrame([{k: binder[k] for k in ("label", "rank", "design", "sequence")}
                           for binder in cfg["binders"]])
    fmt = lambda value: f"{value:.4f}"
    encoded = base64.b64encode((output / "comparison.png").read_bytes()).decode("ascii")
    report = ('<div class="lov-comparison" style="max-width:1200px">'
              '<h2>LOV light/dark comparison</h2>'
              '<p>Light: 2V1B.pdb; Dark: 2V1A.pdb. Target chain ' + html.escape(cfg["chain"]) + '.</p>'
              '<h3>Binder sequences</h3>' + labels.to_html(index=False)
              + '<h3>Mean prediction scores</h3>' + mean_display.to_html(index=False, float_format=fmt)
              + '<p>pLDDT, pTM, and i_pTM use a 0–1 scale. PAE_A columns are in angstroms. '
              'The raw CSV also retains normalized pAE/i_pAE values.</p>'
              + '<h3>Mean paired difference: Light minus Dark</h3>'
              + delta_display.to_html(index=False, float_format=fmt)
              + '<p>Positive delta_i_pTM means higher interface confidence for Light; '
              'negative delta_i_pAE_A means lower interface error for Light. '
              'Model/seed differences are descriptive and are not experimental replicates or affinity estimates.</p>'
              + f'<img alt="Scores for both binders in light and dark conditions" style="width:100%" '
              f'src="data:image/png;base64,{encoded}">'
              + '<h3>Individual predictions</h3>'
              + rows[["binder", "condition", "model_name", "seed"] + display_metrics].to_html(index=False, float_format=fmt)
              + '<p>These predictions use the protein coordinates from each condition as target templates. '
              'AlphaFold can change the target conformation; this workflow does not model FMN or its light-induced covalent chemistry. '
              'The full targets contain residues 403–546; the original exposed design target contains 403–520.</p>'
              '<p>State assignments: <a href="https://www.rcsb.org/structure/2V1B">2V1B (light)</a> and '
              '<a href="https://www.rcsb.org/structure/2V1A">2V1A (dark)</a>.</p></div>')
    (output / "report.html").write_text(report, encoding="utf-8")
    print("\nMean scores:\n" + mean_display.to_string(index=False, float_format=fmt), flush=True)
    print("\nLight minus Dark:\n" + delta_display.to_string(index=False, float_format=fmt), flush=True)


def predict_comparison(config_path):
    import importlib.metadata
    import time
    import numpy as np
    from colabdesign import mk_afdesign_model, clear_mem

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(cfg["output"])
    (output / "pdbs").mkdir(exist_ok=True)
    (output / "records").mkdir(exist_ok=True)
    state = dict(status="running", completed=0, expected=cfg["expected_predictions"],
                 started=datetime.now(timezone.utc).isoformat())
    _write_json(output / "status.json", state)
    versions = {name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpy", "pandas", "matplotlib", "biopython")}
    _write_json(output / "versions.json", versions)
    try:
        clear_mem()
        model = mk_afdesign_model(
            protocol="binder", num_recycles=cfg["recycles"], data_dir=cfg["data_dir"],
            use_multimer=cfg["use_multimer"], model_names=cfg["model_names"],
            use_initial_guess=False, use_initial_atom_pos=False,
        )
        if model._model_names != cfg["model_names"]:
            raise RuntimeError(f"Expected weights {cfg['model_names']}; loaded {model._model_names}")
        print("Loaded models: " + ", ".join(model._model_names), flush=True)
        for binder in cfg["binders"]:
            for condition in ("Light", "Dark"):
                for seed in cfg["seeds"]:
                    model.prep_inputs(
                        pdb_filename=cfg["targets"][condition]["pdb"], chain=cfg["chain"],
                        binder_len=len(binder["sequence"]), rm_target_seq=False, rm_target_sc=False,
                        seed=seed,
                    )
                    for model_num, model_name in enumerate(model._model_names):
                        if (output / "STOP").exists():
                            state["status"] = "stopped"
                            _write_json(output / "status.json", state)
                            print("Stopped after the previous prediction.", flush=True)
                            return 130
                        print(f"[{state['completed'] + 1}/{state['expected']}] "
                              f"{binder['label']} / {condition} / {model_name} / seed {seed}", flush=True)
                        started = time.monotonic()
                        model.predict(seq=binder["sequence"], models=[model_num],
                                      num_recycles=cfg["recycles"], seed=seed,
                                      dropout=False, verbose=False)
                        metrics = model.aux["log"]
                        row = dict(rank=binder["rank"], binder=binder["label"], design=binder["design"],
                                   condition=condition, model_name=model_name, seed=seed,
                                   pLDDT=float(metrics["plddt"]), pTM=float(metrics["ptm"]),
                                   i_pTM=float(metrics["i_ptm"]), pAE=float(metrics["pae"]),
                                   i_pAE=float(metrics["i_pae"]))
                        row.update(pAE_A=31 * row["pAE"], i_pAE_A=31 * row["i_pAE"],
                                   seconds=time.monotonic() - started)
                        if not all(np.isfinite(row[key]) for key in ("pLDDT", "pTM", "i_pTM", "pAE", "i_pAE")):
                            raise RuntimeError("Prediction returned a non-finite confidence score.")
                        stem = f"{binder['design']}_{condition}_{model_name}_seed{seed}"
                        row["pdb"] = "pdbs/" + stem + ".pdb"
                        model.save_pdb(str(output / row["pdb"]))
                        _write_json(output / "records" / (stem + ".json"), row)
                        state["completed"] += 1
                        _write_json(output / "status.json", state)
                        print(f"  i_pTM={row['i_pTM']:.4f}, i_pAE={row['i_pAE_A']:.2f} A, "
                              f"pLDDT={row['pLDDT']:.4f} ({row['seconds']:.1f} s)", flush=True)
        del model
        clear_mem()
        make_report(output, cfg)
        state.update(status="finished", finished=datetime.now(timezone.utc).isoformat())
        _write_json(output / "status.json", state)
        print(f"\nFinished: {state['completed']} predictions. Tables and plots: {output}", flush=True)
        return 0
    except BaseException as exc:
        state.update(status="failed", error=str(exc), finished=datetime.now(timezone.utc).isoformat())
        _write_json(output / "status.json", state)
        raise


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        try:
            return self.stream.write(message)
        except (BrokenPipeError, OSError):
            return len(message)

    def flush(self):
        self.log.flush()
        try:
            self.stream.flush()
        except (BrokenPipeError, OSError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(cfg["output"])
    with (output / "run.log").open("a", encoding="utf-8", buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
        try:
            if args.report_only:
                make_report(output, cfg)
            else:
                raise SystemExit(predict_comparison(args.config))
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
