"""
decomposition_modes.py -- the clinical-mode registry.

A mode maps a clinical question to its basis materials, i.e. to the columns of M.  Adding a
mode is one register_mode() call; its material names must exist in material_library
(available_materials()).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModeSpec:
    key: str
    display_name: str
    materials: List[str]                 # basis materials (columns of M)
    clinical_question: str = ""
    tissue_base: Optional[str] = None    # the ~soft-tissue reference column (calib/notes)
    notes: str = ""


MODES: Dict[str, ModeSpec] = {}


def register_mode(spec: ModeSpec) -> None:
    MODES[spec.key] = spec


def get_mode(key: str) -> ModeSpec:
    if key not in MODES:
        raise KeyError(f"Unknown mode '{key}'. Available: {list(MODES)}")
    return MODES[key]


def list_modes() -> List[str]:
    return list(MODES)


# --- Test / validation mode --------------------------------------------------
# QRM Dual-Energy Phantom V5, middle layer: soft-tissue background with matched
# calcium + iodine inserts. Best-conditioned (iodine K-edge in bin 1) and directly
# verifiable against known inserts -> the Phase-A test mode.
register_mode(ModeSpec(
    "phantom_ca_i", "Phantom Ca/I (test)",
    ["SoftTissue", "HA", "Iodine"],
    clinical_question="Validation: separate matched calcium & iodine inserts",
    tissue_base="SoftTissue",
    notes="QRM Dual-Energy Phantom V5 middle layer. Phase-A test mode (kappa~166).",
))

# Same two agents, water as the base column instead of soft tissue.  The DEP-V5 body is
# literally CTWater (0 HU at 80-140 kV) and the agents are literally CaHA and solid
# iodine, so this basis names three of the phantom's ACTUAL materials.  Running it
# alongside phantom_ca_i is what turns "the base-material choice might matter" into a
# measured number: same data, same estimator, two bases, and the difference IS the
# sensitivity.  Neither basis covers the fat-based lesions of layers A and C -- that
# residual is a known, reported bias rather than a hidden one.
register_mode(ModeSpec(
    "phantom_water_ca_i", "Phantom Ca/I on water (test)",
    ["Water", "HA", "Iodine"],
    clinical_question="Validation: separate matched calcium & iodine inserts, water base",
    tissue_base="Water",
    notes="QRM DEP-V5. Base column matches the phantom body material (CTWater).",
))

# --- Vendor product, NOT a basis we solve -------------------------------------
# The Siemens 2-bin decomposition, imported from its DICOM export (PBV = the iodine map;
# VNC = the complementary base image) so that decomp_validation can score it with exactly
# the metrics it applies to our own maps.  Nothing here ever reaches build_M(): no M is
# constructed, no solve is run, and the material name exists only to address the file.
#
# ONE material on purpose.  With N energy channels a linear decomposition estimates at most
# N materials, so two bins cannot yield three maps -- the vendor's product is iodine plus a
# lumped base, and HA does not exist in it as a separable quantity.  Listing HA here would
# invent a map the mode is mathematically barred from producing.  VNC is stored beside the
# run as an HU image (vnc_hu.nii.gz) and is never scored as a density map.
register_mode(ModeSpec(
    "siemens_2bin", "Siemens 2-bin decomposition (vendor)",
    ["Iodine"],
    clinical_question="Reference point: what the scanner's own 2-bin mode delivers",
    tissue_base=None,
    notes="IMPORTED, not solved. Iodine map from the vendor DICOM export. Rank-limited to "
          "2 materials by the channel count, which is why only Iodine is listed.",
))

# --- The 8 clinical modes of the non-contrast workflow ------------------------
register_mode(ModeSpec(
    "liver", "Liver (iron + steatosis)", ["Water", "Fat", "Iron"],
    "Hepatic iron overload + fat fraction", tissue_base="Water",
    notes="Ill-conditioned (kappa~1.3e4): Water/Fat near-collinear -> needs Phase-B remedies."))
register_mode(ModeSpec(
    "bone", "Bone (mineral density)", ["SoftTissue", "Fat", "HA"],
    "Bone mineral density", tissue_base="SoftTissue",
    notes="kappa~7.7e3 (SoftTissue/Fat collinear)."))
register_mode(ModeSpec(
    "vessels", "Vessels (calcified plaque)", ["SoftTissue", "Fat", "HA"],
    "Calcified atherosclerotic plaque", tissue_base="SoftTissue"))
register_mode(ModeSpec(
    "soft_tissue_tumour", "Soft tissue (tumours)", ["SoftTissue", "Fat", "HA"],
    "Soft-tissue tumour characterisation", tissue_base="SoftTissue"))
register_mode(ModeSpec(
    "bone_marrow", "Bone marrow (metastases)", ["Fat", "SoftTissue", "HA"],
    "Marrow metastasis / oedema", tissue_base="SoftTissue"))
register_mode(ModeSpec(
    "kidney_stone", "Kidney (stone composition)", ["Water", "Fat", "HA"],
    "Renal stone composition", tissue_base="Water", notes="kappa~6.3e3."))
register_mode(ModeSpec(
    "gout", "Joints (gout)", ["SoftTissue", "HA", "Iron"],
    "Gout vs calcification", tissue_base="SoftTissue",
    notes="Iron is a placeholder for a dense, urate-distinct third basis; revisit."))
register_mode(ModeSpec(
    "peri_implant", "Peri-implant (metal artefact)", ["SoftTissue", "HA", "Titanium"],
    "Peri-implant / metal-artefact reduction", tissue_base="SoftTissue",
    notes="Implant metal configurable (Titanium default; CoCr / Tantalum available)."))

# --- Worked example, with contrast -------------------------------------------
register_mode(ModeSpec(
    "anaemia", "Anaemia (worked example)", ["SoftTissue", "Iodine", "Iron"],
    "Blood/iron with iodine contrast", tissue_base="SoftTissue",
    notes="Best-conditioned tissue+contrast basis (kappa~166)."))
