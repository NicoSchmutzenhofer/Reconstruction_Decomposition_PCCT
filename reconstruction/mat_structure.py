"""
mat_structure.py -- print the nested struct tree of the scan descriptor.

scipy.io.loadmat returns MATLAB structs as nested object arrays; drill() walks them and prints
every field with its type and shape.  Use it to find descriptor fields such as the tube angle,
collimation, table feed and projection count, or to check a descriptor from another scanner or
mode.  Reads only.  Set SINO_PATH and DESC_PATH below.
"""

import scipy.io as sio
import numpy as np
from pathlib import Path

SINO_PATH = Path(r"/path/to/sinogram.raw.mat")       # fill in the files to inspect
DESC_PATH = Path(r"/path/to/descriptor.raw.mat")


def drill(val, name="", depth=0, max_depth=6):
    """Recursively print every field of a MATLAB struct/cell loaded by scipy."""
    indent = "  " * depth

    # numpy structured array (MATLAB struct)
    if hasattr(val, 'dtype') and val.dtype.names:
        print(f"{indent}{name}  [struct]  shape={val.shape}")
        for field in val.dtype.names:
            try:
                child = val[field]
                # squeeze away (1,1) wrappers for display
                if hasattr(child, 'shape') and child.shape in [(1,1), (1,)]:
                    child_inner = child.flat[0]
                else:
                    child_inner = child
                drill(child_inner, name=f".{field}", depth=depth+1, max_depth=max_depth)
            except Exception as e:
                print(f"{indent}  .{field}  ERROR: {e}")

    # object array (MATLAB cell array or nested struct)
    elif isinstance(val, np.ndarray) and val.dtype == object:
        print(f"{indent}{name}  [object array]  shape={val.shape}")
        if depth < max_depth:
            for idx in np.ndindex(val.shape):
                try:
                    child = val[idx]
                    drill(child, name=f"{name}{list(idx)}", depth=depth+1, max_depth=max_depth)
                except Exception as e:
                    print(f"{indent}  {idx}  ERROR: {e}")

    # numeric/string array — just print shape and a sample
    elif isinstance(val, np.ndarray):
        sample = ""
        if val.size > 0 and val.size <= 4:
            sample = f"  values={val.flatten().tolist()}"
        elif val.size > 0:
            sample = f"  min={val.min():.4g}  max={val.max():.4g}"
        print(f"{indent}{name}  shape={val.shape}  dtype={val.dtype}{sample}")

    # scalar / string
    else:
        preview = str(val)[:80]
        print(f"{indent}{name}  type={type(val).__name__}  value={preview}")


# ── Sinogram file ──────────────────────────────────────────────
print("=" * 60)
print("SINOGRAM file")
print("=" * 60)
sino_data = sio.loadmat(str(SINO_PATH), struct_as_record=True, squeeze_me=False)

print("\n--- data_sample ---")
drill(sino_data['data_sample'].flat[0], name="data_sample")

print("\n--- header_sample ---")
drill(sino_data['header_sample'].flat[0], name="header_sample")

# ── Descriptor file ───────────────────────────────────────────
print("\n" + "=" * 60)
print("DESCRIPTOR file")
print("=" * 60)
desc_data = sio.loadmat(str(DESC_PATH), struct_as_record=True, squeeze_me=False)

print("\n--- descriptor ---")
drill(desc_data['descriptor'].flat[0], name="descriptor")

print("\n=== Done ===")
