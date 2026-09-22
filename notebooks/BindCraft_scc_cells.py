# %% Read first: BU SCC cell-by-cell workflow
# Open this file in Spyder or VS Code, then run one # %% cell at a time.
# No JupyterLab server or browser is required by this file.
# Your IDE's Python console/kernel MUST run inside an SCC GPU allocation.
# A console on your laptop or an SCC login node cannot use a separate qrsh job.
# VS Code's Python Interactive window uses its Python and Jupyter extensions.
# Spyder uses its IPython console; connect it to a console/kernel in the allocation.
#
# Run setup, DSSP selection, settings, validation, then preflight in order.
# Set RUN_DESIGN = True only in the start cell when ready.
# The background run stays in the allocated session. Use status and stop cells.
# Keep that allocation alive; SCC wall time still applies.
# This file reuses the installed BindCraft environment; it installs nothing.
# Do not run the entire file when you intend to step through it interactively.

# %% 1. Paths and compute helpers
import csv
import html
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from IPython.display import HTML, display

REPO_OVERRIDE = "/project/dunlop/fereshteh/BindCraft/BindCraft"
ENV_PREFIX = Path("/projectnb/dunlop/fereshteh/.conda/envs/BindCraft")
PROJECT_WORK = Path("/projectnb/dunlop/fereshteh/BindCraft")
LINUX_PYTHON = str(ENV_PREFIX / "bin/python")
HOST_PROCESS_OPTIONS = {}

def find_repo():
    here = Path.cwd().resolve()
    candidates = [Path(REPO_OVERRIDE).expanduser()] if REPO_OVERRIDE else [here, *here.parents]
    for path in candidates:
        if (path / "bindcraft.py").is_file() and (path / "functions").is_dir():
            return path.resolve()
    raise FileNotFoundError("Set REPO_OVERRIDE to the directory containing bindcraft.py.")

REPO_ROOT = find_repo()
RUNTIME_ROOT = str(REPO_ROOT)
RUNTIME_PREFIX = str(ENV_PREFIX)
LOCAL_STATE = PROJECT_WORK / ".bindcraft-scc"

def local_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()

def runtime_path(value):
    return str(local_path(value))

def require_gpu_allocation():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run the kernel on an SCC Linux compute node.")
    if not os.environ.get("JOB_ID"):
        raise RuntimeError("No SCC JOB_ID. Start Jupyter inside a scheduled GPU allocation.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or visible in {"-1", "NoDevFiles"}:
        raise RuntimeError("No assigned GPU visible. Request a GPU for this Jupyter session; do not set CUDA_VISIBLE_DEVICES manually.")

def linux_command(argv):
    # env inherits SCC's scheduler variables, CUDA visibility, and module paths.
    updates = {
        "CONDA_PREFIX": RUNTIME_PREFIX,
        "PATH": RUNTIME_PREFIX + "/bin:" + os.environ.get("PATH", ""),
        "LD_LIBRARY_PATH": ":".join(filter(None, [RUNTIME_PREFIX + "/lib", os.environ.get("LD_LIBRARY_PATH", "")])),
        "OMP_NUM_THREADS": os.environ.get("NSLOTS", "1"),
        "OPENBLAS_NUM_THREADS": os.environ.get("NSLOTS", "1"),
        "MKL_NUM_THREADS": os.environ.get("NSLOTS", "1"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    return ["env", *[f"{key}={value}" for key, value in updates.items()], *map(str, argv)]

def run_python(source, args=(), timeout=180):
    completed = subprocess.run(
        linux_command([LINUX_PYTHON, "-u", "-", *args]),
        input=textwrap.dedent(source), capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(f"Compute process exited with {completed.returncode}:\n" + completed.stdout + completed.stderr)
    if completed.stderr.strip():
        print(completed.stderr.strip())
    return completed.stdout

print("Kernel host:", socket.gethostname(), "| Job:", os.environ.get("JOB_ID", "none"))
print("Assigned GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES", "none"))
print("Repository:", REPO_ROOT)
print("Compute Python:", LINUX_PYTHON)
print("Run records:", LOCAL_STATE)
if not Path(LINUX_PYTHON).is_file():
    raise FileNotFoundError("Install BindCraft first or correct ENV_PREFIX: " + LINUX_PYTHON)
PREFLIGHT_OK = False

# %% 2. Select DSSP
DSSP_OVERRIDE = ""  # Optional absolute path to a working mkdssp executable.
DSSP_PATH = local_path(DSSP_OVERRIDE) if DSSP_OVERRIDE else REPO_ROOT / "functions/dssp"
print("DSSP:", DSSP_PATH)

# %% 3. Edit target and protocol settings
design_path = str(PROJECT_WORK / "results/PDL1_scc")
binder_name = "PDL1"
starting_pdb = "example/PDL1.pdb"
chains = "A"
target_hotspot_residues = ""  # Example: "A56"; blank lets BindCraft choose.
lengths = [60, 80]
number_of_final_designs = 1
load_previous_target_settings = ""

design_protocol = "Default"       # Default, Beta-sheet, Peptide
prediction_protocol = "Default"   # Default, HardTarget
interface_protocol = "AlphaFold2" # AlphaFold2, MPNN
# These select repository presets; MPNN sequence generation is controlled by the preset.
template_protocol = "Default"     # Default, Masked
filter_option = "Default"         # Default, Peptide, Relaxed, Peptide_Relaxed, None
max_trajectories = 1               # Counts relaxed trajectories; False removes this limit.
mpnn_sequences = 4
save_design_animations = False
max_wall_minutes = 60             # None removes only the notebook limit, not SCC's limit.

# %% 4. Validate settings
# Resolve presets and validate target paths/chains before importing scientific packages.
if load_previous_target_settings:
    target_settings = json.loads(local_path(load_previous_target_settings).read_text(encoding="utf-8"))
else:
    target_settings = dict(
        design_path=design_path, binder_name=binder_name, starting_pdb=starting_pdb,
        chains=chains, target_hotspot_residues=target_hotspot_residues,
        lengths=lengths, number_of_final_designs=number_of_final_designs,
    )
required = {"design_path", "binder_name", "starting_pdb", "chains", "lengths", "number_of_final_designs"}
if required - target_settings.keys():
    raise ValueError("Missing target settings: " + str(required - target_settings.keys()))
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target_settings["binder_name"]):
    raise ValueError("Use a binder_name containing letters, numbers, underscores, dots or hyphens.")
bounds = target_settings["lengths"]
if not (isinstance(bounds, list) and len(bounds) == 2 and
        all(type(x) is int and x > 0 for x in bounds) and bounds[0] <= bounds[1]):
    raise ValueError("lengths must be [positive_minimum, positive_maximum].")
if type(target_settings["number_of_final_designs"]) is not int or target_settings["number_of_final_designs"] < 1:
    raise ValueError("number_of_final_designs must be a positive integer.")
if max_trajectories is not False and (type(max_trajectories) is not int or max_trajectories < 1):
    raise ValueError("max_trajectories must be a positive integer or False.")
if type(mpnn_sequences) is not int or mpnn_sequences < 1:
    raise ValueError("mpnn_sequences must be a positive integer.")
if max_wall_minutes is not None and (isinstance(max_wall_minutes, bool) or max_wall_minutes <= 0):
    raise ValueError("max_wall_minutes must be positive or None.")

TARGET_PDB = local_path(target_settings["starting_pdb"])
DESIGN_DIR = local_path(target_settings["design_path"])
if not TARGET_PDB.is_file():
    raise FileNotFoundError(TARGET_PDB)
# Read CA records from the first model without needing BioPython in the frontend.
residues = {}
for line in TARGET_PDB.read_text(encoding="utf-8").splitlines():
    if line.startswith("ENDMDL"):
        break
    if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
        residues.setdefault(line[21], set()).add(line[22:27].strip())
selected_chains = [x.strip() for x in target_settings["chains"].split(",")]
if not selected_chains or any(x not in residues for x in selected_chains):
    raise ValueError(f"Requested chains {selected_chains}; available protein chains: {list(residues)}")
target_settings["chains"] = ",".join(selected_chains)
target_settings["target_hotspot_residues"] = target_settings.get("target_hotspot_residues") or None
target_settings["starting_pdb"] = runtime_path(TARGET_PDB)
target_settings["design_path"] = runtime_path(DESIGN_DIR)

design_tags = {"Default": "default_4stage_multimer", "Beta-sheet": "betasheet_4stage_multimer", "Peptide": "peptide_3stage_multimer"}
interface_tags = {"AlphaFold2": "", "MPNN": "_mpnn"}
template_tags = {"Default": "", "Masked": "_flexible"}
prediction_tags = {"Default": "", "HardTarget": "_hardtarget"}
filter_names = {"Default": "default_filters", "Peptide": "peptide_filters", "Relaxed": "relaxed_filters", "Peptide_Relaxed": "peptide_relaxed_filters", "None": "no_filters"}
if design_protocol == "Peptide" and prediction_protocol != "Default":
    raise ValueError("The repository has no Peptide/HardTarget preset; select Default prediction.")
try:
    preset = design_tags[design_protocol] + interface_tags[interface_protocol] + template_tags[template_protocol] + prediction_tags[prediction_protocol]
    ADVANCED_SOURCE = REPO_ROOT / "settings_advanced" / (preset + ".json")
    FILTER_SOURCE = REPO_ROOT / "settings_filters" / (filter_names[filter_option] + ".json")
except KeyError as exc:
    raise ValueError(f"Unknown protocol/filter option: {exc}") from exc
advanced_settings = json.loads(ADVANCED_SOURCE.read_text(encoding="utf-8"))
filters = json.loads(FILTER_SOURCE.read_text(encoding="utf-8"))
advanced_settings.update(
    af_params_dir=RUNTIME_ROOT,  # ColabDesign looks for data_dir/params/params_model_*.npz.
    dssp_path=runtime_path(DSSP_PATH),
    dalphaball_path=runtime_path(REPO_ROOT / "functions" / "DAlphaBall.gcc"),
    max_trajectories=max_trajectories, num_seqs=mpnn_sequences,
    save_design_animations=bool(save_design_animations),
)
if any(char.isspace() for char in advanced_settings["dalphaball_path"]):
    raise ValueError("This repository passes DAlphaBall's path unquoted to Rosetta; use a checkout path without spaces.")
PREFLIGHT_OK = False
print("Target:", TARGET_PDB)
print("Selected residues:", sum(len(residues[x]) for x in selected_chains))
print("Binder length:", bounds, "| Advanced preset:", ADVANCED_SOURCE.name)
print("Filters:", FILTER_SOURCE.name, "| Results:", DESIGN_DIR)
print(json.dumps(target_settings, indent=2))

# %% 5. GPU and package preflight
PREFLIGHT_OK = False
require_gpu_allocation()
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout)
preflight_output = run_python(r"""
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
""", [RUNTIME_ROOT, target_settings["starting_pdb"], advanced_settings["dssp_path"]], timeout=240)
print(preflight_output)
PREFLIGHT_OK = True
PREFLIGHT_SIGNATURE = json.dumps([target_settings, advanced_settings, filters], sort_keys=True)

# %% 6. Start design (enable explicitly)
RUN_DESIGN = False

SUPERVISOR_SOURCE = r"""
import json, os, signal, socket, subprocess, sys, time
from pathlib import Path
run_dir = Path(sys.argv[1])
launch = json.loads((run_dir / 'launch.json').read_text())
state_path = run_dir / 'state.json'
lock_path = Path(launch['output']) / '.bindcraft-notebook.lock'
process = None
locked = False
def save_state(**changes):
    state.update(changes)
    temp = state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(state_path)
def stop(signum, frame):
    raise KeyboardInterrupt
def terminate_design():
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
state = dict(hostname=socket.gethostname(), job_id=os.environ.get('JOB_ID'), status='starting', supervisor_pid=os.getpid(), run_dir=str(run_dir))
save_state()
try:
    # Exclusive lock avoids mixing simultaneous campaigns in the same CSV files.
    with lock_path.open('x') as lock:
        json.dump(dict(supervisor_pid=os.getpid(), run_dir=str(run_dir)), lock)
    locked = True
    argv = [sys.executable, '-u', str(Path(launch['root']) / 'bindcraft.py'),
            '--settings', str(run_dir / 'target.json'),
            '--advanced', str(run_dir / 'advanced.json'),
            '--filters', str(run_dir / 'filters.json')]
    print('Launching:', argv, flush=True)
    process = subprocess.Popen(argv, cwd=launch['root'], stdin=subprocess.DEVNULL,
                               start_new_session=True)
    save_state(status='running', design_pid=process.pid, started_at=time.time())
    timeout = None if launch['max_wall_minutes'] is None else launch['max_wall_minutes'] * 60
    code = process.wait(timeout=timeout)
    save_state(status='finished' if code == 0 else 'failed', returncode=code)
except subprocess.TimeoutExpired:
    terminate_design()
    save_state(status='time_limit', returncode=process.returncode)
    print('Stopped at the notebook wall-time limit.', flush=True)
except KeyboardInterrupt:
    terminate_design()
    save_state(status='stopped', returncode=None if process is None else process.returncode)
    print('Stopped by notebook request.', flush=True)
except BaseException as exc:
    terminate_design()
    save_state(status='failed', error=repr(exc))
    raise
finally:
    if locked:
        lock_path.unlink(missing_ok=True)
    save_state(ended_at=time.time())
"""

if RUN_DESIGN:
    require_gpu_allocation()
    signature = json.dumps([target_settings, advanced_settings, filters], sort_keys=True)
    if not globals().get("PREFLIGHT_OK", False) or signature != globals().get("PREFLIGHT_SIGNATURE"):
        raise RuntimeError("Run validation and the environment preflight for these settings first.")
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    if (DESIGN_DIR / ".bindcraft-notebook.lock").exists():
        raise RuntimeError("A notebook run owns this results folder. Inspect its state and use the stop cell.")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    RUN_DIR = LOCAL_STATE / "runs" / run_name
    RUN_DIR.mkdir(parents=True)
    launch = dict(root=RUNTIME_ROOT, output=target_settings["design_path"],
                  max_wall_minutes=max_wall_minutes)
    for filename, data in [("target.json", target_settings), ("advanced.json", advanced_settings),
                           ("filters.json", filters), ("launch.json", launch)]:
        (RUN_DIR / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
    (RUN_DIR / "supervisor.py").write_text(textwrap.dedent(SUPERVISOR_SOURCE), encoding="utf-8")
    (RUN_DIR / "preflight.txt").write_text(preflight_output, encoding="utf-8")
    output = run_python(r"""
import json, subprocess, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
with (run_dir / 'run.log').open('a') as log:
    child = subprocess.Popen([sys.executable, '-u', str(run_dir / 'supervisor.py'), str(run_dir)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True)
print('Supervisor PID:', child.pid)
""", [runtime_path(RUN_DIR)])
    print(output)
    print("Run files:", RUN_DIR)
    print("Results:", DESIGN_DIR)
    print("Execute the next cell to check progress.")
else:
    print("Design has not been started. Set RUN_DESIGN = True here when ready.")

# %% 7. Monitor progress (rerun anytime)
# Re-run this cell for progress. RUN_TO_INSPECT can select a run after a kernel restart.
RUN_TO_INSPECT = ""  # Absolute path to a run directory printed by the start cell.
if RUN_TO_INSPECT:
    RUN_DIR = local_path(RUN_TO_INSPECT)
elif "RUN_DIR" not in globals():
    candidates = sorted((LOCAL_STATE / "runs").glob("*/launch.json"))
    RUN_DIR = candidates[-1].parent if candidates else None
if RUN_DIR is None:
    print("No notebook-launched run found.")
else:
    print("Inspecting:", RUN_DIR)
    state_path = RUN_DIR / "state.json"
    print(state_path.read_text(encoding="utf-8") if state_path.exists() else "Starting; check again shortly.")
    log_path = RUN_DIR / "run.log"
    if log_path.exists():
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 12000))
            print(log.read().decode("utf-8", errors="replace"))

# %% 8. Stop design (enable explicitly)
# Select the desired run in the status cell, then set this flag to stop it.
STOP_RUN = False
if STOP_RUN:
    if not globals().get("RUN_DIR"):
        raise RuntimeError("Select a run using the status cell first.")
    print(run_python(r"""
import json, os, signal, socket, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
state = json.loads((run_dir / 'state.json').read_text())
if state.get('hostname') != socket.gethostname() or state.get('job_id') != os.environ.get('JOB_ID'):
    raise RuntimeError('Stop from the original allocated host and job. No signal sent.')
pid = state['supervisor_pid']
try:
    argv = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
except FileNotFoundError:
    print('The supervisor has already exited.')
else:
    # Verify the unique run path before signaling; never signal an unrelated reused PID.
    if str(run_dir / 'supervisor.py').encode() not in argv:
        raise RuntimeError('PID does not belong to this run; no signal sent.')
    os.kill(pid, signal.SIGTERM)
    print('Stop requested. Allow up to 15 seconds, then refresh status.')
""", [runtime_path(RUN_DIR)]))
else:
    print("No stop requested.")

# %% 9. Rank completed results
def save_ranked_csv(destination, fieldnames, rows):
    """Keep complete results even if Excel or a sync client blocks replacement."""
    import time

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    staged = destination.with_name(f"{destination.stem}_reranked_{stamp}.csv")
    with staged.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # The pipeline may already have written these rankings. Reading text also
    # normalizes Windows/Unix line endings, so that difference needs no rewrite.
    try:
        identical = destination.is_file() and (
            destination.read_text(encoding="utf-8") == staged.read_text(encoding="utf-8")
        )
    except PermissionError:
        identical = False
    if identical:
        try:
            staged.unlink()
        except PermissionError:
            pass  # A sync client can briefly hold the redundant copy open.
        print("Existing final CSV already contains these rankings; reusing it.")
        return destination

    # Brief retries accommodate transient file locks without truncating the old CSV.
    for delay in (0, 0.5, 1, 2):
        if delay:
            time.sleep(delay)
        try:
            staged.replace(destination)
            return destination
        except PermissionError:
            continue

    print(f"Could not replace {destination.name}.")
    print(f"Complete rankings were saved separately to: {staged}")
    print("Close that CSV in Excel or other applications, then rerun ranking to update the standard file.")
    print("The next results cell will display the separately saved rankings.")
    return staged

RESULTS_TO_VIEW = ""  # Blank uses the output folder configured above.
RESULTS_DIR = local_path(RESULTS_TO_VIEW) if RESULTS_TO_VIEW else DESIGN_DIR
if (RESULTS_DIR / ".bindcraft-notebook.lock").exists():
    raise RuntimeError("The campaign is running; stop it before ranking its output.")
accepted = sorted((RESULTS_DIR / "Accepted").glob("*.pdb"))
mpnn_csv = RESULTS_DIR / "mpnn_design_stats.csv"
FINAL_RESULTS_CSV = RESULTS_DIR / "final_design_stats.csv"
ranked_rows = []
if not accepted or not mpnn_csv.exists():
    print("No accepted designs with MPNN statistics yet:", RESULTS_DIR)
else:
    with mpnn_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    if not {"Design", "Average_i_pTM"}.issubset(columns):
        raise ValueError("MPNN CSV is missing Design or Average_i_pTM.")
    def score(row):
        try:
            value = float(row["Average_i_pTM"])
            return value if value == value else float("-inf")
        except (ValueError, TypeError):
            return float("-inf")
    accepted_by_design = {}
    for path in accepted:
        name, separator, model = path.stem.rpartition("_model")
        if separator and model.isdigit():
            accepted_by_design.setdefault(name, path)
    matches = []
    seen = set()
    for row in sorted(rows, key=score, reverse=True):
        name = row["Design"]
        if name in accepted_by_design and name not in seen:
            matches.append((row, accepted_by_design[name]))
            seen.add(name)
    if not matches:
        raise ValueError("Accepted PDB names did not match the MPNN CSV; existing rankings were preserved.")
    ranked_dir = RESULTS_DIR / "Accepted" / "Ranked"
    if ranked_dir.exists() and any(ranked_dir.iterdir()):
        backup = ranked_dir.with_name("Ranked_previous_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        ranked_dir.rename(backup)
    ranked_dir.mkdir(exist_ok=True)
    for rank, (row, path) in enumerate(matches, start=1):
        ranked_rows.append(dict(Rank=rank, **{key: row[key] for key in columns if key != "Rank"}))
        shutil.copy2(path, ranked_dir / f"{rank}_{path.name}")
    final_csv = RESULTS_DIR / "final_design_stats.csv"
    FINAL_RESULTS_CSV = save_ranked_csv(
        final_csv, ["Rank", *[key for key in columns if key != "Rank"]], ranked_rows,
    )
    print(f"Ranked {len(ranked_rows)} accepted designs. CSV: {FINAL_RESULTS_CSV}")

# %% 10. Display top designs
# Print results in either a plain IPython console or an interactive window.
selected_csv = globals().get("FINAL_RESULTS_CSV")
final_csv = (selected_csv if selected_csv is not None and selected_csv.parent == RESULTS_DIR
             else RESULTS_DIR / "final_design_stats.csv")
if not final_csv.exists():
    print("No final statistics yet.")
else:
    with final_csv.open(newline="", encoding="utf-8") as handle:
        top_rows = list(csv.DictReader(handle))[:20]
    if not top_rows:
        print("No accepted designs yet.")
    for row in top_rows:
        print(" | ".join(f"{key}: {row[key]}" for key in
              ["Rank", "Design", "Length", "Average_i_pTM", "Average_pLDDT", "Sequence"]
              if key in row))

# %% 11. View top structure (optional rich display)
# Rich HTML viewers may not render in Spyder's plain console.
# py3Dmol runs in the existing Linux environment; only its HTML is displayed here.
# The browser needs access to 3Dmol's JavaScript CDN and must trust this notebook.
top_pdbs = sorted((RESULTS_DIR / "Accepted" / "Ranked").glob("1_*.pdb"))
if not top_pdbs:
    print("No ranked structure to display.")
else:
    view_html = run_python(r"""
import py3Dmol, sys
from pathlib import Path
view = py3Dmol.view(width=800, height=500)
view.addModel(Path(sys.argv[1]).read_text(), 'pdb')
view.setBackgroundColor('white')
view.setStyle({'chain':'A'}, {'cartoon': {'color':'#3c5b6f'}})
view.setStyle({'chain':'B'}, {'cartoon': {'color':'#B76E79'}})
view.zoomTo()
print(view._make_html())
""", [runtime_path(top_pdbs[0])])
    display(HTML(view_html))

# %% 12. View animation (optional rich display)
# Rich HTML viewers may not render in Spyder's plain console.
# Available only for designs generated with save_design_animations=True.
if not top_pdbs:
    print("No ranked design animation to display.")
else:
    top_name = top_pdbs[0].stem.split("_", 1)[1]
    trajectory_name = top_name.rsplit("_mpnn", 1)[0]
    animation = RESULTS_DIR / "Accepted" / "Animation" / (trajectory_name + ".html")
    if animation.is_file():
        display(HTML(filename=str(animation)))
    else:
        print("No animation saved for this design. Enable animations before a future run if needed.")

