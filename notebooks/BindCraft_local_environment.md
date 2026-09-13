# Local BindCraft environment

Inspected 2026-09-12. Use [BindCraft_local.ipynb](BindCraft_local.ipynb).

## How to use the notebook

Open it in VS Code or your notebook application on Windows and select **Python 3
(ipykernel)** using `C:\Users\Feres\miniconda3\python.exe`. In Jupyter, use
**Kernel > Change Kernel**; in VS Code, use **Select Kernel** at the top right.
Run the cells in order. The notebook sends compute jobs to the
existing Ubuntu environment through `wsl.exe`; Windows does not need its own
installation of JAX, ColabDesign, PyRosetta, pandas, or py3Dmol.

It also runs from Jupyter inside Ubuntu. From an Ubuntu terminal:

```bash
source /home/fereshteh/miniconda3/bin/activate BindCraft
cd /mnt/c/Users/Feres/Dropbox/Projects/BindCraft/BindCraft
python -m jupyterlab notebooks/BindCraft_local.ipynb
```

The default cells check the environment without launching a design. Edit the
target/protocol settings, run validation and preflight, then set `RUN_DESIGN = True`
in the start cell. The background process has a 60-minute time limit by default.
Use the status cell to read progress and the stop cell to terminate the run.

Results default to `results/PDL1_local/`. Each launch saves its exact JSON settings,
preflight output, process state, and log in `.bindcraft-local/runs/<run-id>/`.
The local runtime folder is ignored by Git; Dropbox may still synchronize it.

## What is installed

| Component | Observed value |
|---|---|
| Windows repository | `C:\Users\Feres\Dropbox\Projects\BindCraft\BindCraft` |
| Linux repository | `/mnt/c/Users/Feres/Dropbox/Projects/BindCraft/BindCraft` |
| Windows notebook Python | `C:\Users\Feres\miniconda3\python.exe`, Python 3.13.2 |
| Windows `cuda_env` | Python 3.12.14, CUDA NVCC 12.6.85; a separate environment |
| Linux distribution | Ubuntu 26.04 LTS, WSL2 |
| Linux environment | `/home/fereshteh/miniconda3/envs/BindCraft` |
| Linux Python | 3.10.21 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB VRAM |
| NVIDIA driver | 566.14; GPU visible in both Windows and WSL |
| WSL memory at inspection | Approximately 15 GiB RAM and 4 GiB swap |
| JAX / jaxlib | 0.6.0 / 0.6.0 |
| NumPy / pandas / SciPy | 1.26.4 / 2.3.3 / 1.15.2 |
| ColabDesign | 1.1.3 |
| PyRosetta | 2026.29 quarterly, Python 3.10 Linux build |
| BioPython / PDBFixer / OpenMM | 1.88 / 1.12.0 / 8.6.0 |
| Flax / Haiku | 0.9.0 / 0.0.17 |
| JupyterLab / ipykernel | 4.6.3 / 7.3.0 in WSL |
| py3Dmol | 2.5.4 in WSL |
| AlphaFold parameters | 15 existing `.npz` files, approximately 5.6 GB total |
| Example target | `example/PDL1.pdb`, chain A, 115 residues numbered 18–132 |

## Compatibility changes

The original notebook installs into `/content/bindcraft`, mounts Google Drive,
imports `bindcraft.functions`, and embeds the design loop. The local notebook
discovers the existing checkout, translates Windows paths with `wslpath`, and
launches this checkout's `bindcraft.py` with the existing Linux Python interpreter.
It preserves the original notebook's target/protocol/filter selections, ranking,
top-design display, and optional animation display.

The installed ColabDesign parameter loader first searches
`data_dir/params/params_model_<name>.npz`. Accordingly, the new notebook explicitly
sets `af_params_dir` to the repository root and uses the existing `params/` folder.
It does not download weights or rerun `install_bindcraft.sh`.

`functions/DAlphaBall.gcc` is a Linux ELF executable. Without the Conda library
directory on `LD_LIBRARY_PATH`, `ldd` reports `libgfortran.so.5 => not found`.
The library already exists in the BindCraft environment. The notebook supplies
that environment's `lib` and `bin` paths to each compute process and its children.

The repository's statically linked DSSP 2.0.4 executable reports its version but
exits with signal 11 when reading `example/PDL1.pdb`. A normalized PDB copy, C locale,
unlimited stack, and disabled ASLR did not resolve this. BioPython returned zero
assignments without raising an error, so checking only imports or `--version`
would miss the failure.

A separate DSSP 3.0.0 conda prefix is installed at `.bindcraft-local/dssp`, with
matching Boost 1.73 libraries. The notebook points to its `bin/mkdssp`; the bundled
binary is preserved. The optional repair cell recreates the installation using
the [SALILAB DSSP package](https://anaconda.org/channels/salilab/packages/dssp/overview).
The package cache is also kept under `.bindcraft-local/`.

JAX GPU computations use WSL because native Windows NVIDIA support is unavailable
in the [JAX platform matrix](https://docs.jax.dev/en/latest/installation.html).
The notebook disables JAX preallocation before starting compute processes. This
changes allocation behavior, not the model's memory requirements; see
[JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).

## Trial settings and limits

The local notebook starts with the repository's default multimer design and
default filters, binder lengths 60–80, one desired final design, four MPNN sequences,
and a maximum of one relaxed trajectory. Animations are off initially. It preserves
the preset's design iteration counts, recycles, and filter thresholds.

The pipeline counts `max_trajectories` using PDB files in `Trajectory/Relaxed`,
not every attempted trajectory. The notebook adds a separate wall-time limit and
terminates the full design process group when that limit is reached. A run may
stop without an accepted design. Resume using the same target/output folder and
increase the limits if needed.

The repository recommends at least 32 GB VRAM. This machine has 8 GB, so successful
imports and small GPU tests do not establish that a complete design will fit.
A full AlphaFold optimization/design trajectory was not run during notebook setup.

## Verification

- Notebook JSON/schema validation and compilation of every code cell.
- All 19 cells pass Run All in a real Jupyter kernel in WSL with the saved defaults.
  The final compute preflight also passes from the Windows notebook interpreter.
- Windows notebook Python successfully connects to the existing WSL environment
  and translates repository, input, output, and tool paths.
- BindCraft functions, JAX, ColabDesign, and PyRosetta import successfully.
- A JIT-compiled matrix operation runs on the NVIDIA GPU.
- Existing `model_1_ptm` and `model_1_multimer_v3` weights load through ColabDesign.
- The soluble `v_48_020` MPNN model initializes successfully.
- A single multimer AlphaFold model initializes and prepares PDL1 plus a 60-residue
  binder without running a forward prediction or optimization.
- PyRosetta reads the example target as 115 residues.
- DSSP 3.0.0 produces 115 secondary-structure assignments for the example target.
- Temporary-fixture checks cover accepted-only ranking by `Average_i_pTM`, prior
  ranking preservation, empty outputs, 3D viewer HTML generation, and missing
  animations.
- Temporary scripts verify successful/failed background runs, wall-time termination,
  and stopping the recorded process without signaling an unrelated PID.
- The actual start cell launches a temporary stand-in script through WSL, saves all
  configuration snapshots, captures its log, records success, and removes its lock.

The original notebook, installer, settings presets, and pipeline Python sources
were not modified. The existing installer modification and untracked `params/`
directory were present before this work.
