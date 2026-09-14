"""
decomposition -- image-domain material decomposition for 4-bin PCCT.

Public API (the stable entry points a future CLI/GUI builds on):
    decompose(volumes, config, ...) -> DecompResult
    DecompConfig / DecompResult
    list_modes() / get_mode() / MODES / register_mode()
    stability_report() / mode_stability()
    material_library (available_materials, build_M, densities, ...)

See the README for the design principles this follows.
"""
from . import material_library, noise_estimation, denoising
from .decomposition_modes import MODES, ModeSpec, get_mode, list_modes, register_mode
from .material_decomposition import (
    DecompConfig, DecompResult, decompose, mode_stability, stability_report,
    solve_ols, solve_wls, wls_operator, voxel_wls,
    load_threshold_volumes, load_threshold_volumes_dicom, build_dicom_index,
    save_decomp_result, read_recon_calibration,
)

__all__ = [
    "material_library", "noise_estimation", "denoising",
    "MODES", "ModeSpec", "get_mode", "list_modes", "register_mode",
    "DecompConfig", "DecompResult", "decompose", "mode_stability", "stability_report",
    "solve_ols", "solve_wls", "wls_operator", "voxel_wls",
    "load_threshold_volumes", "load_threshold_volumes_dicom", "build_dicom_index",
    "save_decomp_result", "read_recon_calibration",
]
