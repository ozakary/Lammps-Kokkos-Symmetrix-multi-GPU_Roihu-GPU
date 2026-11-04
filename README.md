# LAMMPS + MACE on Roihu-GPU (NVIDIA GH200, Multi-GPU)
This repository documents how to build and run [LAMMPS](https://www.lammps.org) with the [`pair_symmetrix`](https://github.com/wcwitt/symmetrix) interface for using MACE MLIPs on the CSC [Roihu](https://docs.csc.fi/computing/systems-roihu/) supercomputer using NVIDIA GH200 Grace Hopper superchips.

---

📄 Author **Ouail Zakary**
- 📧 [ouail.zakary@aalto.fi](mailto:ouail.zakary@aalto.fi)
- 🔗 ORCID [0000-0002-7793-3306](https://orcid.org/0000-0002-7793-3306)
- 📁 Personal Website [ozakary.github.io](https://ozakary.github.io/)

---

The approach uses [symmetrix](https://github.com/wcwitt/symmetrix), a pure C++/Kokkos implementation of MACE with no Python or PyTorch dependency at runtime. On Roihu-GPU, LAMMPS is compiled with the Kokkos CUDA backend for the Hopper GPU (compute capability 9.0) and the Grace CPU (Arm Neoverse V2, aarch64).

**This implementation has some advantages compared to the ML-IAP one**
- No Python, no PyTorch, no container needed at runtime
- Native Kokkos/CUDA multi-GPU support via MPI domain decomposition
- CUDA-aware OpenMPI is provided by the default Roihu-GPU environment
- Model weights stored as a portable `.json` file, extracted once on any machine (the same `.json` used on LUMI works here)

> **Requirement** `pair_symmetrix` requires LAMMPS >= [10 September 2025](https://github.com/lammps/lammps/releases/tag/patch_10Sep2025).

> **Architecture** Roihu-CPU nodes are x86 (AMD) and Roihu-GPU nodes are aarch64 (Grace). Binaries built on one side do not run on the other. Always build on a **Roihu-GPU login node**.

---

## Table of Contents

1. [Hardware and Partitions](#hardware-and-partitions)
2. [Module Environment](#module-environment)
3. [Step 1 Clone LAMMPS and symmetrix](#step-1-clone-lammps-and-symmetrix)
4. [Step 2 Patch LAMMPS with pair_symmetrix](#step-2-patch-lammps-with-pair_symmetrix)
5. [Step 3 Build LAMMPS](#step-3-build-lammps)
6. [Step 4 Extract Model Weights to JSON](#step-4-extract-model-weights-to-json)
7. [Step 5 LAMMPS Input File](#step-5-lammps-input-file)
8. [Step 6 Validate the Build](#step-6-validate-the-build)
9. [Step 7 Production Job Scripts](#step-7-production-job-scripts)
10. [Differences from LUMI](#differences-from-lumi)
11. [Troubleshooting](#troubleshooting)
12. [Citation](#citation)

---

## Hardware and Partitions

Each Roihu-GPU node has 4 GH200 superchips. Each reserved GPU grants 72 Grace cores and 212 GiB of memory (95 GiB HBM3 + 117 GiB LPDDR5X). CPU and memory are allocated automatically from the number of GPUs requested.

| Partition   | Time limit | Nodes  | GPUs            | Use                                            |
|-------------|------------|--------|-----------------|------------------------------------------------|
| `gputest`   | 15 min     | 1 - 2  | 1 - 4 per node  | Build validation and short tests               |
| `gpumedium` | 36 h       | 1      | 1 - 4 per job   | Production up to one full node                 |
| `gpularge`  | 36 h       | 1 - 10 | 4 per node      | Multi-node production (requires a scalability test approved by CSC) |

GPUs are requested with `--gres=gpu:gh200:N` (N = 1 to 4 per node).

---

## Module Environment

Load the default Roihu-GPU GNU + CUDA environment (as listed in the CSC documentation, November 2026):

```bash
module purge
module load gcc/14.3.0 cuda/12.9.1 openmpi/5.0.10 openblas/0.3.30
```

Key versions
- `gcc/14.3.0`
- `cuda/12.9.1`
- `openmpi/5.0.10` (CUDA-aware)
- `openblas/0.3.30`

The same modules must be loaded at runtime so the correct MPI and CUDA libraries are used. If the build or runtime fails after a system update, check `module spider gcc cuda openmpi` and compare with the versions above.

CMake >= 3.27 is required. Check with `cmake --version`. If the system CMake is older, use `module spider cmake` or install a recent one with `pip install --user cmake`.

---

## Step 1 Clone LAMMPS and symmetrix

```bash
export PROJAPPL=/projappl/<project>/<username>/packages/lammps-symmetrix
mkdir -p $PROJAPPL
cd $PROJAPPL

# Clone LAMMPS release (>= 10 Sep 2025 required)
git clone --branch release --depth 1 https://github.com/lammps/lammps

# Clone symmetrix with all submodules
git clone --recursive https://github.com/wcwitt/symmetrix
```

---

## Step 2 Patch LAMMPS with pair_symmetrix

```bash
cd $PROJAPPL/symmetrix/pair_symmetrix
./install.sh $PROJAPPL/lammps
cd $PROJAPPL
```

---

## Step 3 Build LAMMPS

### CUDA driver stub (login node only)

Roihu-GPU login nodes have no GPU driver, so `libcuda.so.1` is missing at link time. Link against the stub shipped with the CUDA toolkit. **This is for the build only. Never put the stub directory in `LD_LIBRARY_PATH` at runtime.**

```bash
module purge
module load gcc/14.3.0 cuda/12.9.1 openmpi/5.0.10 openblas/0.3.30

CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
STUB_DIR=$PROJAPPL/cuda-stub
mkdir -p $STUB_DIR
ln -sf $CUDA_ROOT/lib64/stubs/libcuda.so $STUB_DIR/libcuda.so.1

export LD_LIBRARY_PATH=$STUB_DIR:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_ROOT/lib64/stubs:$LIBRARY_PATH
export NVCC_WRAPPER_DEFAULT_COMPILER=g++
```

### Configure, build, and install

```bash
mkdir -p $PROJAPPL/lammps-symmetrix-build

cd $PROJAPPL/lammps

cmake \
    -B build-cuda \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER=gcc \
    -D CMAKE_CXX_COMPILER=$PROJAPPL/lammps/lib/kokkos/bin/nvcc_wrapper \
    -D CMAKE_Fortran_COMPILER=gfortran \
    -D CMAKE_CXX_STANDARD=20 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D CMAKE_INSTALL_PREFIX=$PROJAPPL/lammps-symmetrix-build \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_OMP=ON \
    -D BUILD_MPI=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_HOPPER90=ON \
    -D Kokkos_ARCH_ARMV9_GRACE=ON \
    -D Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION=ON \
    -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
    -D SYMMETRIX_KOKKOS=ON \
    -D SYMMETRIX_SPHERICART_CUDA=OFF \
    cmake

cmake --build build-cuda -j 16
cmake --install build-cuda
```

The installed binary will be at `$PROJAPPL/lammps-symmetrix-build/bin/lmp`.

The stub exports are needed only for building. In a new SSH session they are gone and must be set again before any rebuild. After the build, remove the stub directory so it cannot leak into a job environment

```bash
rm -rf $PROJAPPL/cuda-stub
```

On the login node, `lmp -h` prints `CUDA driver is a stub library` twice. This is expected, since the login node has no GPU.

Quick check that the CUDA backend and `pair_symmetrix` are compiled in:

```bash
$PROJAPPL/lammps-symmetrix-build/bin/lmp -h | grep -iE "KOKKOS package API|symmetrix"
```

The output should list `CUDA` among the Kokkos backends and `symmetrix/mace` and `symmetrix/mace/kk` among the pair styles.

The script [`build_lammps_symmetrix_roihu.sh`](build_lammps_symmetrix_roihu.sh) runs Steps 1 to 3 in one go.

**Compared with LUMI**
- `hipcc` is replaced by the Kokkos `nvcc_wrapper` (which calls `nvcc` for device code and `g++` for host code)
- `Kokkos_ENABLE_HIP` and `Kokkos_ARCH_VEGA90A` are replaced by `Kokkos_ENABLE_CUDA`, `Kokkos_ARCH_HOPPER90`, and `Kokkos_ARCH_ARMV9_GRACE`
- `SYMMETRIX_SPHERICART_CUDA=OFF` keeps the spherical harmonics in the symmetrix Kokkos kernels, as on LUMI. With `ON`, configure reports `Will use CUDA version of sphericart`, but the bundled sphericart is built without CUDA (`SPHERICART_ENABLE_CUDA=OFF`) and LAMMPS fails at runtime with `sphericart was not compiled with CUDA support`
- No FFT library is specified. LAMMPS falls back to KISS FFT on the host and cuFFT for Kokkos, which is irrelevant for MLIP-only simulations without `kspace_style`

---

## Step 4 Extract Model Weights to JSON

The `.json` format changes between symmetrix versions. Extract the model with the **same symmetrix commit** that was compiled into LAMMPS, otherwise LAMMPS fails with `json.exception.out_of_range ... key 'radial_spline_r_min' not found`. A `.json` extracted for an older LUMI build may not work on Roihu.

Get the commit on Roihu

```bash
cd $PROJAPPL/symmetrix && git rev-parse HEAD
```

### Install symmetrix Python package locally

```bash
git clone --recursive https://github.com/wcwitt/symmetrix
cd symmetrix
git checkout <commit from Roihu>
git submodule update --init --recursive
pip install mace-torch
pip install --force-reinstall --no-deps ./symmetrix/
```

`--no-deps` avoids reinstalling `torch` and `mace-torch` in an existing environment.

### Extract the model

```bash
symmetrix_extract_mace \
    --model your_model.model \
    --atomic-numbers Z1 Z2  # replace with atomic numbers of your elements
                              # e.g. C=6, O=8
```

This produces a file named `your_model-Z1-Z2.json`. Check that it has the new format with `grep -o '"radial_spline_r_min"' your_model-Z1-Z2.json`, then copy it to Roihu

```bash
scp your_model-Z1-Z2.json <username>@roihu-gpu.csc.fi:/scratch/<project>/<username>/your_simulation_dir/
```

> **Note on atomic number ordering** The filename reflects the order of atomic numbers as passed to `symmetrix_extract_mace`. The element list in the LAMMPS `pair_coeff` command must match your LAMMPS atom type ordering (as defined in your data file), not the filename ordering. Always verify with `grep -A5 "Masses" your_structure.data`.

---

## Step 5 LAMMPS Input File

```lammps
comm_modify   cutoff 8.0
pair_style    symmetrix/mace
pair_coeff    * * your_model-Z1.json Element1 Element2
```

The element list after the filename maps to LAMMPS atom types 1, 2, ... in order. Example for a given system (CO2) where type 1 = C and type 2 = O:

```lammps
comm_modify   cutoff 8.0
pair_style    symmetrix/mace
pair_coeff    * * fine-tuned_mace-mp-0b3-medium_vf-6.json C O
```

---

## Step 6 Validate the Build

> [!WARNING]
> On LUMI, the fallback non-GPU-aware MPI path of `pair_symmetrix` produced **wrong forces and broken dynamics** in multi-GPU runs ([symmetrix issue #41](https://github.com/wcwitt/symmetrix/issues/41)). This is a `pair_symmetrix` issue, not a LUMI issue, so the same risk exists on Roihu. **Never use `-pk kokkos gpu/aware off`**, and always confirm that multi-GPU forces match single-GPU forces before production.

### Run the test

The test simulates a (10,0) single-walled carbon nanotube (3 unit cells, 120 C atoms) for 10 ps of NVT at 300 K on one GH200, using the MACE-OMAT-0 medium foundation model.

| File | Purpose |
|------|---------|
| [`cnt_10_0_u3.data`](cnt_10_0_u3.data) | LAMMPS data file of the (10,0) SWCNT |
| [`cnt_lammps.in`](cnt_lammps.in) | LAMMPS input, 10 ps NVT at 300 K with `pair_style symmetrix/mace` |
| [`job_test_symmetrix.sh`](job_test_symmetrix.sh) | Slurm job script for 1 GPU on the `gputest` partition |

The model `.json` file is not included because of its size (about 45 MB). Generate it as described in [Step 4](#step-4-extract-model-weights-to-json), using the same symmetrix commit as the LAMMPS build.

**1. Extract the model (local machine)**

Download `mace-omat-0-medium.model` from the [MACE foundation models repository](https://github.com/ACEsuit/mace-foundations), then run

```bash
symmetrix_extract_mace \
    --model mace-omat-0-medium.model \
    --atomic-numbers 6
```

This writes `mace-omat-0-medium-6.json`. Check the format with

```bash
grep -o '"radial_spline_r_min"' mace-omat-0-medium-6.json
```

**2. Copy the files to Roihu**

```bash
mkdir -p /scratch/<project>/<username>/symmetrix_test
# copy cnt_10_0_u3.data, cnt_lammps.in, job_test_symmetrix.sh and mace-omat-0-medium-6.json here
```

**3. Edit the job script**

In `job_test_symmetrix.sh`, set `--account=<project>` and point `PROJAPPL` to your LAMMPS installation.

**4. Submit**

```bash
cd /scratch/<project>/<username>/symmetrix_test
sbatch job_test_symmetrix.sh
```

**5. Check the output**

- The `.out` file lists `GPU 0: NVIDIA GH200 120GB`
- `libcuda.so.1` resolves to `/lib64/libcuda.so.1`, not to a `stubs` directory
- The temperature fluctuates around 300 K and the total energy stays stable
- `equilibration.dump` and `equil.restart` are written in the run folder

### What a passing test looks like

- `nvidia-smi -L` in the job output lists 4 GH200 GPUs
- No `GPU-aware MPI` warning in any `log_*gpu.lammps` file
- Step-0 potential energy agrees across 1, 2, and 4 GPUs to within 1e-5 eV/atom
- Maximum force difference across GPU counts is below 1e-3 eV/Å (in practice it is usually close to machine precision). The issue #41 failure mode gives errors orders of magnitude larger
- NVE total energy shows no systematic drift over the short run
- Throughput (timesteps/s) increases from 1 to 4 GPUs for systems large enough to saturate several GPUs

`compare_runs.py` prints `PASS` or `FAIL` for each check. The thresholds can be changed at the top of the script.

---

## Step 7 Production Job Scripts

### Single GPU

```bash
#!/bin/bash
#SBATCH --job-name=lammps-symmetrix
#SBATCH --account=<project>
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72   # 1 MPI rank per GPU, 72 Grace cores per GPU
#SBATCH --gres=gpu:gh200:1                       # 1 GH200
#SBATCH --output=lammps_symmetrix_%j.out
#SBATCH --error=lammps_symmetrix_%j.err

module purge
module load gcc/14.3.0 cuda/12.9.1 openmpi/5.0.10 openblas/0.3.30

# Drop any build-time CUDA stub paths inherited from the submitting shell
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -e cuda-stub -e '/stubs' | paste -sd:)
unset LIBRARY_PATH

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

PROJAPPL=/projappl/<project>/<username>/packages/lammps-symmetrix

# Add LAMMPS and symmetrix shared libraries to the runtime library path
export LD_LIBRARY_PATH=$PROJAPPL/lammps-symmetrix-build/lib64:$PROJAPPL/lammps-symmetrix-build/lib:$LD_LIBRARY_PATH

srun $PROJAPPL/lammps-symmetrix-build/bin/lmp \
  -in your_input.in \
  -k on g 1 \
  -sf kk \
  -pk kokkos newton on neigh half
```

### Full node (4 GPUs)

```bash
#!/bin/bash
#SBATCH --job-name=lammps-symmetrix
#SBATCH --account=<project>
#SBATCH --partition=gpumedium
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 --cpus-per-task=72   # 1 MPI rank per GPU, product = 288 cores
#SBATCH --gres=gpu:gh200:4                       # 4 GH200 = full node
#SBATCH --output=lammps_symmetrix_%j.out
#SBATCH --error=lammps_symmetrix_%j.err

module purge
module load gcc/14.3.0 cuda/12.9.1 openmpi/5.0.10 openblas/0.3.30

# Drop any build-time CUDA stub paths inherited from the submitting shell
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -e cuda-stub -e '/stubs' | paste -sd:)
unset LIBRARY_PATH

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

PROJAPPL=/projappl/<project>/<username>/packages/lammps-symmetrix
export LD_LIBRARY_PATH=$PROJAPPL/lammps-symmetrix-build/lib64:$PROJAPPL/lammps-symmetrix-build/lib:$LD_LIBRARY_PATH

# Each rank sees all 4 GPUs of the node. Kokkos assigns GPU i to node-local rank i.
srun $PROJAPPL/lammps-symmetrix-build/bin/lmp \
  -in your_input.in \
  -k on g 4 \
  -sf kk \
  -pk kokkos newton on neigh half
```

### Multi-node (4 nodes, 16 GPUs)

Requires access to `gpularge`, which CSC grants after a scalability test (see [How to get access to Roihu large and gpularge partitions](https://docs.csc.fi/accounts/how-to-access-roihu-large-partition/)).

```bash
#!/bin/bash
#SBATCH --job-name=lammps-symmetrix
#SBATCH --account=<project>
#SBATCH --partition=gpularge
#SBATCH --time=36:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4 --cpus-per-task=72   # 1 MPI rank per GPU per node
#SBATCH --gres=gpu:gh200:4                       # 4 GH200 per node, 16 in total
#SBATCH --output=lammps_symmetrix_%j.out
#SBATCH --error=lammps_symmetrix_%j.err

module purge
module load gcc/14.3.0 cuda/12.9.1 openmpi/5.0.10 openblas/0.3.30

# Drop any build-time CUDA stub paths inherited from the submitting shell
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -e cuda-stub -e '/stubs' | paste -sd:)
unset LIBRARY_PATH

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

PROJAPPL=/projappl/<project>/<username>/packages/lammps-symmetrix
export LD_LIBRARY_PATH=$PROJAPPL/lammps-symmetrix-build/lib64:$PROJAPPL/lammps-symmetrix-build/lib:$LD_LIBRARY_PATH

srun $PROJAPPL/lammps-symmetrix-build/bin/lmp \
  -in your_input.in \
  -k on g 4 \
  -sf kk \
  -pk kokkos newton on neigh half
```

> **Note on `-k on g N`** On Roihu, each rank sees every GPU of its node, so `N` is the number of GPUs **per node** (1, 2, or 4), not the total. Kokkos maps node-local rank `i` to GPU `i`. With `--cpus-per-task=72`, rank `i` is placed on the Grace CPU attached to GPU `i`.

---

## Differences from LUMI

| | LUMI-G | Roihu-GPU |
|---|---|---|
| GPU | AMD MI250X (8 GCDs per node, 64 GB each) | NVIDIA GH200 (4 per node, 96 GB HBM3 each) |
| Host CPU | AMD EPYC Trento (x86) | NVIDIA Grace (aarch64) |
| Compiler | `hipcc` | `nvcc_wrapper` (`nvcc` + `g++`) |
| Kokkos backend | `HIP`, `VEGA90A` | `CUDA`, `HOPPER90`, `ARMV9_GRACE` |
| MPI | Cray MPICH | OpenMPI 5 (CUDA-aware) |
| GPU request | `--gpus-per-node=8` | `--gres=gpu:gh200:4` |
| Cores per GPU | 7 | 72 |
| `MPICH_GPU_SUPPORT_ENABLED`, `MPICH_SMP_SINGLE_COPY_MODE` | Required | Not applicable (Cray MPICH only) |
| `-k on g N` | `g 1` with `--gpu-bind=closest` | `g` = GPUs per node |

---

## Troubleshooting

### Wrong dynamics / blown-up trajectories with multiple GPUs

**Symptom** Single-GPU simulations produce correct dynamics, but multi-GPU simulations show physically unreasonable trajectories (exploding volumes, huge pressures, unstable energies).

**Cause** The fallback non-GPU-aware MPI path in `pair_symmetrix` ([symmetrix #41](https://github.com/wcwitt/symmetrix/issues/41)). LAMMPS switches to this path silently if it detects that MPI is not GPU-aware.

**Fix** Do not use `gpu/aware off`. Check the LAMMPS log for a warning mentioning `GPU-aware MPI`. If it appears, confirm that the loaded OpenMPI is CUDA-aware

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

which should return `true`. Make sure the same modules are loaded at build time and at runtime, and that `comm_modify cutoff 8.0` appears before `pair_style`.

### CUDA driver is a stub library

```
Exception: cudaGetDeviceCount(&count) error( cudaErrorStubLibrary): CUDA driver is a stub library
```

The CUDA stub from the build step is on `LD_LIBRARY_PATH` at runtime. This happens when `sbatch` is run from the shell used for building, since Slurm copies that environment into the job and `module purge` does not remove manually added paths. The job scripts above strip stub paths after `module load`. To check which driver is used, add

```bash
nvidia-smi -L
ldd $PROJAPPL/lammps-symmetrix-build/bin/lmp | grep libcuda
```

before `srun`. `libcuda.so.1` must resolve to `/lib64/libcuda.so.1`.

### sphericart was not compiled with CUDA support

```
Exception: sphericart was not compiled with CUDA support
```

LAMMPS was configured with `SYMMETRIX_SPHERICART_CUDA=ON`. Reconfigure with `-D SYMMETRIX_SPHERICART_CUDA=OFF` (with the build environment of Step 3 loaded) and rebuild. Only `libsymmetrix` is recompiled, so this takes a few minutes.

### Cannot find libcuda.so.1 at link time

```
warning: libcuda.so.1, needed by liblammps.so, not found
```

The build runs on a login node without a GPU driver. Create the stub symlink and export `LD_LIBRARY_PATH` and `LIBRARY_PATH` as shown in [Step 3](#cuda-driver-stub-login-node-only) before running `cmake --build`. Alternatively, build inside an interactive GPU session (`sinteractive` from a Roihu-GPU login node), where the real driver is present.

### Exec format error

```
cannot execute binary file: Exec format error
```

The binary was built on a Roihu-CPU (x86) login node. Rebuild on a Roihu-GPU login node.

### Missing shared libraries at runtime

If `lmp` fails with a missing `.so` error, ensure both library directories are in `LD_LIBRARY_PATH` and that the build modules are loaded

```bash
module load gcc/14.3.0 cuda/12.9.1 openmpi/5.0.10 openblas/0.3.30
export LD_LIBRARY_PATH=$PROJAPPL/lammps-symmetrix-build/lib64:$PROJAPPL/lammps-symmetrix-build/lib:$LD_LIBRARY_PATH
```

### Out of GPU memory

```
Kokkos ERROR: Cuda memory space failed to allocate X GiB
```

Each GH200 has 95 GiB of allocatable HBM3, about 1.5 times the memory of one MI250X GCD. As a rough starting point, the LUMI atoms-per-GPU guide can be scaled by about 1.5. Memory usage scales with the number of atoms per MPI rank, so increase the number of GPUs if a single rank runs out of memory. Check actual usage with `nvidia-smi` during a short test run.

### Wrong GPU assignment / all ranks on GPU 0

If `nvidia-smi` shows all ranks on one GPU, `-k on g N` does not match the number of GPUs per node. Use `g 4` for full nodes and `g 1` for single-GPU jobs.

### JSON key not found

```
Exception: [json.exception.out_of_range.403] key 'radial_spline_r_min' not found
```

The `.json` was extracted with a different symmetrix version than the one compiled into LAMMPS. Re-extract it as described in [Step 4](#step-4-extract-model-weights-to-json).

### JSON parse error

```
Exception: [json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input
```

The `.json` model file is either missing, empty, or the path in `pair_coeff` is wrong. Check that the file exists and the path is correct relative to where the job is submitted.

---

## Citation

When using `symmetrix` in publications, please cite

- D. P. Kovács et al., "MACE-OFF: Short-Range Transferable Machine Learning Force Fields for Organic Molecules," *J. Am. Chem. Soc.* **147**, 17598 (2025). [[arXiv]](https://arxiv.org/abs/2312.15211)
- I. Batatia et al., "A foundation model for atomistic materials chemistry," *J. Chem. Phys.* **163**, 184110 (2025). [[arXiv]](https://arxiv.org/abs/2401.00096)

When using the KOKKOS package in LAMMPS, cite

- https://doi.org/10.1145/3731599.3767498
