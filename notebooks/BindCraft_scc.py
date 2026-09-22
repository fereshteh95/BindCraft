#!/usr/bin/env python3
"""Run BindCraft in an allocated BU SCC terminal; no Jupyter required.

Activate your BindCraft Conda environment first, then run:
    python notebooks/BindCraft_scc.py --help
    python notebooks/BindCraft_scc.py --prepare-only
    python notebooks/BindCraft_scc.py --check-only
    python notebooks/BindCraft_scc.py

The default is a short PDL1 trial. Edit DEFAULTS or use command-line options.
Relative input paths resolve from the repository, regardless of the current directory.
Ctrl+C stops the design process group. SCC's allocation wall time still applies.
"""
#%%
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import uuid
from datetime import datetime

#%%
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "env_prefix": "/projectnb/dunlop/fereshteh/.conda/envs/BindCraft",
    "work_dir": "/projectnb/dunlop/fereshteh/BindCraft",
    "pdb": "example/PDL1.pdb",
    "name": "PDL1",
    "chains": "A",
    "hotspots": None,
    "lengths": [60, 80],
    "number": 1,
    "max_trajectories": 1,
    "mpnn_sequences": 4,
    "minutes": 60,
    "advanced": "settings_advanced/default_4stage_multimer.json",
    "filters": "settings_filters/default_filters.json",
}

#%%
PREFLIGHT_SOURCE = r"""
import importlib.metadata as md
import json, os, re, subprocess, sys, zipfile
from pathlib import Path
root, target, dssp_path = map(Path, sys.argv[1:4])
os.chdir(root)
sys.path.insert(0, str(root))
print('Python:', sys.executable, sys.version.split()[0], flush=True)
for package in ['jax', 'jaxlib', 'numpy', 'pandas', 'biopython', 'colabdesign', 'pyrosetta', 'py3Dmol']:
    print(package + ': ' + md.version(package), flush=True)
weight_names = ([f'params_model_{i}_multimer_v3.npz' for i in range(1, 6)] +
                [f'params_model_{i}_ptm.npz' for i in range(1, 6)])
for name in weight_names:
    path = root / 'params' / name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError('Missing/empty AlphaFold weights: ' + str(path))
    with zipfile.ZipFile(path) as archive:
        if not archive.namelist():
            raise ValueError('Empty parameter archive: ' + str(path))
print('AlphaFold: 10 required pTM/multimer archives found (headers checked).', flush=True)
for binary in [dssp_path, root / 'functions' / 'DAlphaBall.gcc']:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError('Missing/non-executable tool: ' + str(binary))
    linked = subprocess.run(['ldd', str(binary)], capture_output=True, text=True)
    if 'not found' in linked.stdout + linked.stderr:
        raise RuntimeError('Unresolved libraries for ' + str(binary) + '\n' + linked.stdout + linked.stderr)
import jax
import jax.numpy as jnp
gpu = jax.devices('gpu')[0]
with jax.default_device(gpu):
    result = jax.jit(lambda a: a @ a)(jnp.ones((8, 8)))
    assert float(result[0, 0]) == 8.0
print('JAX GPU computation passed:', gpu.device_kind, flush=True)
import functions
import pyrosetta as pr
from Bio.PDB import PDBParser, DSSP
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {root}/functions/DAlphaBall.gcc -corrections::beta_nov16 true -relax:default_repeats 1')
pose = pr.pose_from_pdb(str(target))
print('PyRosetta target residues:', pose.total_residue(), flush=True)
version_output = subprocess.check_output([str(dssp_path), '--version'], text=True)
version = int(re.search(r'\d+', version_output).group())
command = [str(dssp_path)] + (['--output-format=dssp'] if version >= 4 else []) + [str(target)]
check = subprocess.run(command, capture_output=True, text=True, timeout=60)
if check.returncode or not check.stdout.strip():
    raise RuntimeError(f'DSSP failed: exit {check.returncode}\n' + check.stderr)
model = PDBParser(QUIET=True).get_structure('target', str(target))[0]
assignments = DSSP(model, str(target), dssp=str(dssp_path))
if len(assignments) == 0:
    raise RuntimeError('DSSP returned zero residues; do not start design.')
print('DSSP:', version_output.strip(), '| assigned residues:', len(assignments), flush=True)
print('BindCraft compute preflight passed.', flush=True)
"""


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()

#%%

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-prefix", default=DEFAULTS["env_prefix"], help="Installed BindCraft Conda prefix")
    parser.add_argument("--work-dir", default=DEFAULTS["work_dir"], help="Persistent directory for logs and default results")
    parser.add_argument("--settings", help="Existing target JSON; overrides all target-specific flags")
    parser.add_argument("--pdb", default=DEFAULTS["pdb"], help="Prepared target PDB")
    parser.add_argument("--name", default=DEFAULTS["name"], help="Binder name prefix")
    parser.add_argument("--chains", default=DEFAULTS["chains"])
    parser.add_argument("--hotspots", default=DEFAULTS["hotspots"], help='Example: "A493,A495"')
    parser.add_argument("--lengths", nargs=2, type=int, default=DEFAULTS["lengths"], metavar=("MIN", "MAX"))
    parser.add_argument("--number", type=int, default=DEFAULTS["number"], help="Desired accepted designs")
    parser.add_argument("--output", help="Results directory; default WORK_DIR/results/NAME_scc")
    parser.add_argument("--advanced", default=DEFAULTS["advanced"], help="Advanced protocol JSON")
    parser.add_argument("--filters", default=DEFAULTS["filters"], help="Filter JSON")
    parser.add_argument("--max-trajectories", type=int, default=DEFAULTS["max_trajectories"], help="Relaxed trajectory limit; 0 disables")
    parser.add_argument("--mpnn-sequences", type=int, default=DEFAULTS["mpnn_sequences"])
    parser.add_argument("--minutes", type=float, default=DEFAULTS["minutes"], help="Design wall time in minutes; 0 disables script limit")
    parser.add_argument("--dssp", help="Override bundled DSSP with a working executable")
    parser.add_argument("--animations", action="store_true", help="Save design animations")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Validate and save JSON settings without requiring an environment/GPU")
    mode.add_argument("--check-only", action="store_true", help="Run GPU/package/weights/DSSP preflight without starting design")
    args = parser.parse_args(argv)
    if args.max_trajectories < 0 or args.mpnn_sequences < 1 or not (0 <= args.minutes < float("inf")):
        parser.error("Trajectory/time limits must be finite and nonnegative; MPNN sequences must be positive.")
    return args


#%%
def prepare_settings(args):
    work = repo_path(args.work_dir)
    if args.settings:
        target = json.loads(repo_path(args.settings).read_text())
    else:
        target = dict(design_path=args.output or str(work / "results" / (args.name + "_scc")),
                      binder_name=args.name, starting_pdb=args.pdb, chains=args.chains,
                      target_hotspot_residues=args.hotspots, lengths=args.lengths,
                      number_of_final_designs=args.number)
    required = {"design_path", "binder_name", "starting_pdb", "chains", "lengths", "number_of_final_designs"}
    if required - target.keys():
        raise ValueError("Missing target keys: " + str(required - target.keys()))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target["binder_name"]):
        raise ValueError("Binder name must contain letters, digits, underscores, dots or hyphens.")
    bounds = target["lengths"]
    if not (isinstance(bounds, list) and len(bounds) == 2 and
            all(type(x) is int and x > 0 for x in bounds) and bounds[0] <= bounds[1]):
        raise ValueError("Binder lengths must be [positive minimum, positive maximum].")
    if type(target["number_of_final_designs"]) is not int or target["number_of_final_designs"] < 1:
        raise ValueError("Number of final designs must be positive.")
    pdb = repo_path(target["starting_pdb"])
    available = set()
    for line in pdb.read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            available.add(line[21])
    chains = [c.strip() for c in target["chains"].split(",")]
    if not chains or any(c not in available for c in chains):
        raise ValueError(f"Requested chains {chains}; available protein chains: {sorted(available)}")
    target.update(starting_pdb=str(pdb), design_path=str(repo_path(target["design_path"])),
                  chains=",".join(chains), target_hotspot_residues=target.get("target_hotspot_residues") or None)
    advanced = json.loads(repo_path(args.advanced).read_text())
    filters = json.loads(repo_path(args.filters).read_text())
    dalphaball = str(REPO_ROOT / "functions/DAlphaBall.gcc")
    if any(c.isspace() for c in dalphaball):
        raise ValueError("DAlphaBall requires a repository path without whitespace.")
    advanced.update(af_params_dir=str(REPO_ROOT), dssp_path=str(repo_path(args.dssp or "functions/dssp")),
                    dalphaball_path=dalphaball, max_trajectories=args.max_trajectories or False,
                    num_seqs=args.mpnn_sequences, save_design_animations=args.animations)
    return work, target, advanced, filters


def compute_environment(prefix):
    # Preserve scheduler GPU visibility and module library paths.
    env = os.environ.copy()
    env.update(CONDA_PREFIX=str(prefix), PATH=str(prefix / "bin") + ":" + env.get("PATH", ""),
               LD_LIBRARY_PATH=":".join(filter(None, [str(prefix / "lib"), env.get("LD_LIBRARY_PATH", "")])),
               XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = env.get("NSLOTS", "1")
    return env


def require_allocation():
    if not sys.platform.startswith("linux") or not os.environ.get("JOB_ID"):
        raise RuntimeError("Run this command inside an SCC GPU allocation (qrsh or qsub), not on a login node.")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1", "NoDevFiles"}:
        raise RuntimeError("No allocated GPU is visible. Request a GPU; do not set CUDA_VISIBLE_DEVICES manually.")


def stream_process(argv, env, log_path, minutes=0):
    """Stream stdout/stderr to terminal and log; stop the whole group on interrupt/timeout."""
    def forward(source, log):
        for line in iter(source.readline, ""):
            log.write(line)
            log.flush()
            try:
                print(line, end="", flush=True)
            except BrokenPipeError:
                pass

    with log_path.open("w") as log:
        process = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", start_new_session=True)
        reader = threading.Thread(target=forward, args=(process.stdout, log), daemon=True)
        reader.start()
        try:
            return process.wait(timeout=minutes * 60 if minutes else None)
        except (KeyboardInterrupt, subprocess.TimeoutExpired) as exc:
            print("\nStopping design/process group...", flush=True)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 130 if isinstance(exc, KeyboardInterrupt) else 124
        finally:
            reader.join(timeout=20)
            process.stdout.close()


def main(argv=None):
    args = parse_args(argv)
    work, target, advanced, filters = prepare_settings(args)
    prefix = repo_path(args.env_prefix)
    python = prefix / "bin/python"
    if not args.prepare_only:
        require_allocation()
        if not python.is_file():
            raise FileNotFoundError(f"BindCraft Python missing: {python}. Install the environment or set --env-prefix.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run = work / ".bindcraft-scc/runs" / stamp
    run.mkdir(parents=True)
    for name, value in [("target", target), ("advanced", advanced), ("filters", filters)]:
        (run / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    state = dict(status="prepared", hostname=socket.gethostname(), job_id=os.environ.get("JOB_ID"),
                 launcher_pid=os.getpid(), results=target["design_path"], run_dir=str(run))
    def save_state(status, **extra):
        state.update(status=status, **extra)
        temp = run / "state.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(run / "state.json")
    save_state("prepared")
    print("Run records:", run, flush=True)
    print("Results:", target["design_path"], flush=True)
    print("Target:", target["starting_pdb"], flush=True)
    if args.prepare_only:
        print("Settings prepared. No GPU work started.")
        return 0
    env = compute_environment(prefix)
    (run / "preflight.py").write_text(PREFLIGHT_SOURCE)
    print("Checking packages, weights, GPU operation, PyRosetta and DSSP...", flush=True)
    save_state("checking")
    rc = stream_process([str(python), "-u", str(run / "preflight.py"), str(REPO_ROOT),
                         target["starting_pdb"], advanced["dssp_path"]], env, run / "preflight.log", minutes=5)
    if rc:
        save_state("preflight_failed", returncode=rc)
        return rc
    if args.check_only:
        save_state("checked")
        print("Preflight passed; no design started.")
        return 0
    output = Path(target["design_path"])
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".bindcraft-notebook.lock"
    # Same exclusive lock name as the SCC notebook to avoid mixed campaigns.
    try:
        lock = lock_path.open("x")
    except FileExistsError:
        raise RuntimeError(f"Results are locked: {lock_path}. Verify the owning job before removing a stale lock.")
    try:
        with lock:
            json.dump(state, lock)
        save_state("running")
        rc = stream_process([str(python), "-u", str(REPO_ROOT / "bindcraft.py"),
                             "--settings", str(run / "target.json"), "--advanced", str(run / "advanced.json"),
                             "--filters", str(run / "filters.json")], env, run / "run.log", args.minutes)
        save_state({0: "finished", 124: "time_limit", 130: "stopped"}.get(rc, "failed"), returncode=rc)
        print(f"Run {state['status']} (exit {rc}). Results: {output}")
        return rc
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Scheduler termination follows the same cleanup path as Ctrl+C where possible.
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
