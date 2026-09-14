# 4-bin Photon-Counting CT: Reconstruction and Material Decomposition

Code for reconstructing the **four energy-threshold sinograms** of a photon-counting CT scanner
(Siemens NAEOTOM Alpha) into four energy-resolved volumes and decomposing them into **material
maps**, such as iodine, calcium hydroxyapatite and a base material. Two further stages validate
the maps on a QRM DEP-V5 phantom scan and compare image quality with the vendor's
reconstructions. Written for a master's thesis.

---

## What is not included

| Not included | Why | What to do |
|---|---|---|
| `geometry/` | The scanner's detector tables are vendor data and not redistributable | Supply your own, see [Detector geometry](#detector-geometry) |
| Scan data | Raw `.mat` sinograms and DICOM exports are not ours to distribute | Fill in the `/path/to/...` placeholders, see [Setup](#setup) |

Everything that needs no data runs anywhere. The self-tests build synthetic phantoms and check
the maths against known answers:

```bash
python -m decomposition.selftest_decomposition       # decomposition maths
python -m decomposition.selftest_decomp_validation   # phantom validation metrics
python -m reconstruction.selftest_image_quality      # NPS / TTF / NEQ / d'
python -m decomposition.research_decomposition       # conditioning tables and figures
```

---

## Setup

- Python 3.10+ with numpy, scipy, matplotlib, h5py, SimpleITK, scikit-image and PyWavelets
- [ASTRA Toolbox](https://astra-toolbox.com/) with a CUDA GPU for the reconstruction; every
  later stage runs on the CPU
- optional: pydicom for `probe_dose.py`, pandas and openpyxl for `build_mono_table.py`

Run all commands from the repository root. The batch scripts are written for SLURM. Before the
first run, fill in the placeholders in the scripts you use:

| Placeholder | Meaning |
|---|---|
| `conda activate /path/to/conda/envs/MatDecomp` | your Python environment |
| `WFBP_DIR`, `VMI_DIR`, `S2BIN_DIR` | vendor DICOM export folders, one reconstruction per folder |
| `DATA_PATH`, `DESC_PATH` (`DESC` in `batch_dose.sh`) | raw threshold sinogram (`.mat`, HDF5) and its descriptor (`.mat`) |
| `##SBATCH --nodelist=your-node` | optional: remove one `#` to run on a fixed node |

The Python scripts that read raw data have the same placeholders in the file:
`python_reconstruction.py` (`data_path`, `desc_path`), both subtraction studies (`DATA_PATH`,
`DESC_PATH`) and `mat_info.py` / `mat_structure.py` (`SINO_PATH`, `DESC_PATH`).

---

## Repository layout

```
reconstruction/
    helical_reconstruction.py               library: geometry, rebinning, MAR, FBP/SIRT/CGLS, HU calibration
    python_reconstruction.py                reconstruction driver; settings at the top of the file
    recon_invariants.py                     run-time checks of the reconstruction
    image_quality_metrics.py                NPS, TTF, NEQ, detectability index
    figure_panels.py                        shared image-panel helpers for the figures
    recon_comparison.py                     image quality: ours vs vendor WFBP vs vendor VMI
    image_subtraction_investigation.py      threshold subtraction study, image domain
    sinogram_subtraction_investigation.py   threshold subtraction study, sinogram domain
    processed_separation_investigation.py   checks whether the thresholds are cumulative
    probe_dicom_origin.py, probe_dose.py    coordinate-frame and acquisition-parameter checks
    mat_info.py, mat_structure.py           .mat format and struct inspectors
    selftest_image_quality.py               synthetic tests of the metrics

decomposition/
    material_decomposition.py               library: B = M x, the four estimators, file I/O
    material_library.py                     material signatures, the columns of M
    decomposition_modes.py                  clinical modes (question -> basis materials)
    noise_estimation.py, denoising.py       noise covariance and edge-preserving denoisers
    decompose.py                            driver: one mode over one input source
    decomp_validation.py                    validation on the QRM DEP-V5 phantom
    research_decomposition.py               conditioning analysis and ablations
    figure_signatures.py                    mass-attenuation figure
    build_mono_table.py                     builds data/mu_rho_mono.csv
    selftest_decomposition.py, selftest_decomp_validation.py
    data/                                   mass-attenuation and material tables

batch*.sh      SLURM scripts, one per stage or study
```

`output/`, `logs/` and `geometry/` are gitignored.

---

## Detector geometry

`build_geom` combines the scan descriptor with two detector tables from `geometry/` in the
repository root (`recon_comparison.py` also accepts `--geo-dir`). A JSON file names the tables,
so their file names are free:

```
geometry/
    detector_spec.json
    channel_angles.txt
    row_positions.txt
```

`detector_spec.json`, with example values:

```json
{
  "channel_angles_file": "channel_angles.txt",
  "row_positions_file": "row_positions.txt",
  "sad_mm": 5.0,
  "sdd_mm": 10.0,
  "n_channels": 8
}
```

| Key | Required | Content |
|---|---|---|
| `channel_angles_file` | yes | Text file with one fan angle per detector channel, in radians. 0 is the ray through the isocentre; the order is the channel order of the sinogram. |
| `row_positions_file` | yes | Text file with one z-position per detector row at the isocentre, in mm, in the row order of the sinogram. |
| `sad_mm`, `sdd_mm` | no | Source-to-isocentre and source-to-detector distance. The geometry check stops the run if the scan differs by more than 0.5 mm. |
| `n_channels` | no | Expected number of channels, checked against the channel table. |

- Both tables are read with `numpy.loadtxt`: numbers separated by spaces or line breaks; `#`
  starts a comment.
- The row table may list more rows than the scan uses. `build_geom` takes the active row count
  from the descriptor and selects the central block (`row_mapping="central"`; also `"first"`,
  `"last"`, `"reversed"`). The selected rows must straddle z = 0.
- The air reference is taken from both detector edges: as many channels per edge as there are
  channels whose angle is within 10 % of the largest one (at least 8).
- Any of `sad_mm`, `sdd_mm` and `n_channels` that is missing is not checked.

### Another scanner

For the Siemens data, `build_geom` fills a plain dictionary from the descriptor and the tables.
For another scanner, build this dictionary yourself. The reconstruction and its run-time checks
read only these keys:

| Key | Unit, shape | Meaning |
|---|---|---|
| `SAD`, `SDD` | mm | source-to-isocentre and source-to-detector distance |
| `proj_per_rotation`, `n_total_proj` | count | projections per rotation, and in the whole scan |
| `channel_betas` | rad, `(n_channels,)` | fan angle of each channel (the channel table) |
| `row_z_mm` | mm, `(n_rows,)` | z-position of each active row at the isocentre |
| `slice_width_mm`, `z_spacing_mm` | mm | collimated row width, and output slice spacing |
| `z_focus_per_proj` | mm, `(n_total_proj,)` | z-position of the focal spot for each projection |
| `view_angle_per_proj` | rad, `(n_total_proj,)` | view angle of each projection, in ASTRA's fan-beam convention |
| `pitch_mm` | mm | table feed per rotation |
| `n_air_channels` | count | channels per detector edge used as air reference |
| `det_alignment` | channels | centre-of-rotation offset |
| `detector_spacing_deg` | deg | channel pitch; only used with `geometry_model="flat"` |
| `spike_mask` | bool, `(n_channels,)` or `None` | defective channels, from `detect_defect_channels` |

The sinogram is a float array of shape `[n_total_proj, n_rows, n_channels]` whose values grow
with attenuation, with air close to zero at the detector edges. A minimal run, with ASTRA on a
CUDA GPU:

```python
from reconstruction.helical_reconstruction import (
    detect_defect_channels, reconstruct_helical_stack, z_targets_for_full_scan)

# geom: the dictionary above; sino: the sinogram array
geom["spike_mask"] = detect_defect_channels(sino)
z_targets, _ = z_targets_for_full_scan(geom)
volume = reconstruct_helical_stack(sino, geom, z_targets[:20], n_pixels=512)
```

`auto_hu_calibrate` and `apply_hu_calibration` in the same module convert the result to HU.

---

## Running the pipeline

### 1. Reconstruction

```bash
sbatch batch_reconstruction.sh
```

All settings are constants at the top of `reconstruction/python_reconstruction.py`.
`FAST_MODE = True` reconstructs a quick slab preview for tuning; run the driver directly for
that, because the batch script only accepts full runs (`FAST_MODE = False`).
A full run writes to `output/reconstruction/`:

- `reconstruction_thr_{A,B,C,D}_HU.nii.gz`, the input of the decomposition
- `reconstruction_thr_{A,B,C,D}.nii.gz` (raw attenuation) and a 4-D stack of all thresholds
- `invariant_log.json`, the result of the run-time checks

### 2. Material decomposition

```bash
sbatch batch.sh own                                  # own | wfbp | vmi
python -m decomposition.decompose --list-modes       # available clinical modes
python -m decomposition.decompose --source wfbp --wfbp-dir /path/to/wfbp_export --mode phantom_ca_i
```

Solves `B = M x + n` per voxel: `B` holds the energy-channel attenuations, `M` the mass
attenuation coefficients of the basis materials and `x` their densities. `--estimator` selects
`ols`, `wls`, `wls_denoise` or `wls_joint`. The number of energy channels is read from the
data. Results go to `output/decomposition/<source>/<mode>/`, together with the config that
produced them.

### 3. Phantom validation

```bash
sbatch batch_decomp_validation.sh              # self-test, then stages 0-2
sbatch batch_decomp_validation.sh --stage 2    # only metrics and figures
```

Needs a QRM DEP-V5 scan and a segmentation `Segmentation_decomposition.nrrd` in the WFBP export
folder; its label layout is described at the top of `decomposition/decomp_validation.py`.
Stage 0 places the ROIs, stage 1 runs the decompositions and skips runs that are up to date,
stage 2 computes metrics and figures. To include the scanner's own 2-bin iodine map, set
`S2BIN_DIR` (or leave it empty to skip it), run
`python -m decomposition.decomp_validation --probe-siemens /path/to/2bin_export` and set
`S2BIN_IODINE_MATCH` / `S2BIN_VNC_MATCH` from its output. Results go to
`output/research/decomp_validation/` (`metrics/`, `figures/`, `qc/`).

### 4. Image-quality comparison

```bash
sbatch batch_compare.sh                # stages 0-2
sbatch batch_compare.sh --stage 2      # only metrics and figures
```

Reconstructs the phantom's insert slab with several settings (GPU) and measures noise, NPS,
TTF, NEQ, detectability, bias and PSNR/SSIM against the vendor's WFBP and VMI images. Results go
to `output/research/recon_comparison/`.

### Other scripts

The remaining batch scripts run the standalone studies and checks listed under
[Repository layout](#repository-layout). The header of each script says what it needs and what
it writes.

---

## Critical invariants

Properties of the data and the pipeline that the code relies on. Code comments refer to them
by number.

1. **Thresholds are stored in reverse order in the HDF5** (index 0 = D, 3 = A). Read them
   through `_load_threshold`, which maps the index.
2. **Channels are flipped on load**, and `build_geom` is called with `channels_flipped=True`.
   Keep the flip; fix a mirrored image in the output instead.
3. **Threshold sinograms are never subtracted.** Each threshold is reconstructed on its own;
   energy bins are formed from the images.
4. **Defective channels are detected with `max_excess = col_max - p99`** and dilated by
   ±1 channel.
5. **The Hampel filter is off by default**, because it flattens high-contrast inserts.
6. **The selected detector rows must straddle z = 0**; if not, try another `row_mapping`.
7. **Scan-dependent values are never hardcoded.** They come from the descriptor and
   `geometry/`, see [Detector geometry](#detector-geometry).
8. **The view angle is the physical tube angle from the descriptor**, not the projection index;
   `build_geom` stops if the start angle is missing.

---

## Extending

The decomposition package follows three design principles: library functions work on arrays,
`decompose()` is the single entry point and takes a serialisable `DecompConfig`, and new
options are added through registries instead of edits to the core.

- **Material:** add a column to `data/mu_rho_binavg.csv` and a row to `data/materials.csv`
  (`material, density_g_cm3, k_edge_keV, category, description`). For monoenergetic input also
  add rows to `data/mu_rho_mono.csv` (`material, energy_keV, mu_rho_cm2_g, edge_side`).
- **Clinical mode:** one `register_mode(ModeSpec(key, display_name, [materials]))` call in
  `decomposition_modes.py`.
- **Denoiser:** a function decorated with `@denoiser("name")` in `denoising.py`.
- **Research experiment:** a function decorated with `@experiment("name")` in
  `research_decomposition.py`.

---

## Contributing

- Module docstrings start with `filename.py -- one-line summary.`, then say what the module
  does, how, and how to run it. `probe_dose.py` and `figure_signatures.py` use the summary line
  as their `--help` text.
- Docstrings and comments explain why, not what; the reasoning behind a design choice sits next
  to the code it concerns.
- Plain ASCII in anything printed to a console.

---

## Citation

If this code is useful in your work, please cite the thesis. Contact the author for the
reference while it is in preparation.
