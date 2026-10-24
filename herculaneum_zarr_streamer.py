#!/usr/bin/env python3
"""
Phase 3: Zero-Disk-Bloat Micro-CT OME-Zarr Sub-Volume Streamer (`herculaneum_zarr_streamer.py`)
Streams a real 65 x 512 x 512 surface-volume slab (plus aligned inklabels.zarr and 3D mesh metadata)
from s3://vesuvius-challenge-open-data directly into Apple M5 Max 36 GB Unified Memory via `vesuvius` & `zarr`,
executes 3D Sobel Crackle / Maximum Intensity / Normal-Phase Projections on PyTorch MPS (Metal 4 GPU),
renders the 3D papyrus fiber mesh + multi-layer diagnostic plates, and saves telemetry.
"""
import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import s3fs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from vesuvius.data.utils import open_zarr

DEFAULT_SCROLL = "PHerc1667"
DEFAULT_SEGMENT = "20251212185248-w029_20251212185248662_flatboi"
DEFAULT_SURFVOL_ZARR = (
    "s3://vesuvius-challenge-open-data/PHerc1667/segments/"
    "20251212185248-w029_20251212185248662_flatboi/surface-volumes/"
    "2.399um-0.22m-78keV-volume-20251217075048.zarr/3"
)
DEFAULT_INKLABELS_ZARR = (
    "s3://vesuvius-challenge-open-data/PHerc1667/segments/"
    "20251212185248-w029_20251212185248662_flatboi/ink-labels/"
    "2.399um-volume-20251217075048/20260918/inklabels.zarr/3"
)
DEFAULT_SUPERVISION_ZARR = (
    "s3://vesuvius-challenge-open-data/PHerc1667/segments/"
    "20251212185248-w029_20251212185248662_flatboi/ink-labels/"
    "2.399um-volume-20251217075048/20260918/supervision.zarr/3"
)
DEFAULT_OFFICIAL_INK_DS8 = (
    "vesuvius-challenge-open-data/PHerc1667/segments/"
    "20251212185248-w029_20251212185248662_flatboi/ink-detection/downsampled/"
    "PHerc1667-20251212185248-2.399um-0.22m-78keV-volume-20251217075048-"
    "20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg"
)
DEFAULT_TIFXYZ_META = (
    "vesuvius-challenge-open-data/PHerc1667/segments/"
    "20251212185248-w029_20251212185248662_flatboi/mesh/"
    "20251212185248-on-20251217075048-2.399um.tifxyz/meta.json"
)

def stream_subvolume(
    surfvol_url: str = DEFAULT_SURFVOL_ZARR,
    inklabels_url: str = DEFAULT_INKLABELS_ZARR,
    supervision_url: str = DEFAULT_SUPERVISION_ZARR,
    tifxyz_meta_path: str = DEFAULT_TIFXYZ_META,
    z_start: int = 22,
    z_depth: int = 65,
    y0: int = 2176,
    x0: int = 2176,
    hw: int = 512,
):
    """Stream a 65 x 512 x 512 sub-volume slab and ground-truth labels directly into RAM/Unified Memory."""
    from PIL import Image
    from io import BytesIO
    t0 = time.perf_counter()
    print(f"[Streamer] Opening remote OME-Zarr surface volume: {surfvol_url}")
    vol_arr = open_zarr(surfvol_url, mode="r", storage_options={"anon": True})
    full_shape = vol_arr.shape
    print(f"[Streamer] Remote OME-Zarr Shape: {full_shape} | dtype={vol_arr.dtype} | chunks={vol_arr.chunks}")

    z_end = z_start + z_depth
    y_end = y0 + hw
    x_end = x0 + hw
    print(f"[Streamer] Streaming 3D slab z=[{z_start}:{z_end}] ({z_depth}), y=[{y0}:{y_end}] ({hw}), x=[{x0}:{x_end}] ({hw})...")
    slab_uint8 = np.asarray(vol_arr[z_start:z_end, y0:y_end, x0:x_end], dtype=np.uint8)
    t_vol = time.perf_counter() - t0
    mb_size = slab_uint8.nbytes / (1024 * 1024)
    print(f"[Streamer] Streamed {slab_uint8.shape} slab ({mb_size:.2f} MB) in {t_vol:.2f}s ({mb_size/max(t_vol,1e-3):.2f} MB/s)")

    # Stream aligned 2D inklabels and supervision masks
    print(f"[Streamer] Streaming aligned inklabels.zarr, supervision.zarr & official ink prediction for y=[{y0}:{y_end}], x=[{x0}:{x_end}]...")
    lbl_arr = open_zarr(inklabels_url, mode="r", storage_options={"anon": True})
    sup_arr = open_zarr(supervision_url, mode="r", storage_options={"anon": True})
    ink_gt = (np.asarray(lbl_arr[y0:y_end, x0:x_end]) > 0).astype(np.uint8)
    sup_mask = (np.asarray(sup_arr[y0:y_end, x0:x_end]) > 0).astype(np.uint8)

    # Fetch tifxyz 3D mesh metadata & official neural ink prediction
    s3 = s3fs.S3FileSystem(anon=True)
    with s3.open(tifxyz_meta_path, "r") as f:
        mesh_meta = json.load(f)
    with s3.open(DEFAULT_OFFICIAL_INK_DS8, "rb") as f:
        full_ink_pred = np.array(Image.open(BytesIO(f.read())), dtype=np.float32) / 255.0
    official_ink_patch = full_ink_pred[y0:y_end, x0:x_end]

    lateral_um = 2.399 * 8.0  # Level 3 lateral pixel size = 19.192 um (9.83 mm per 512 px), normal Z = 2.399 um
    total_time = time.perf_counter() - t0
    telemetry = {
        "scroll_id": DEFAULT_SCROLL,
        "segment_id": DEFAULT_SEGMENT,
        "voxel_size_um": 2.399,
        "lateral_pixel_um": round(lateral_um, 3),
        "beam_energy_keV": 78.0,
        "propagation_distance_m": 0.22,
        "remote_volume_shape": list(full_shape),
        "slab_shape": list(slab_uint8.shape),
        "z_window": [z_start, z_end],
        "y_window": [y0, y_end],
        "x_window": [x0, x_end],
        "physical_patch_mm": [round(hw * lateral_um / 1000.0, 3), round(hw * lateral_um / 1000.0, 3)],
        "physical_depth_um": round(z_depth * 2.399, 2),
        "slab_megabytes": round(mb_size, 3),
        "stream_seconds": round(t_vol, 3),
        "total_io_seconds": round(total_time, 3),
        "ink_gt_positive_fraction": round(float(ink_gt.mean()), 4),
        "supervision_positive_fraction": round(float(sup_mask.mean()), 4),
        "tifxyz_mesh_meta": mesh_meta,
    }
    return slab_uint8, ink_gt, sup_mask, official_ink_patch, telemetry


def compute_mps_projections(slab_uint8: np.ndarray):
    """
    Run 3D GPU-accelerated projections on Apple Silicon MPS (Metal 4):
    1. Multi-layer Maximum Intensity Projection (MIP) across the recto writing lamina
    2. 3D Sobel Crackle & Phase-Contrast Gradient Magnitude Projection
    3. Micro-Topographic Papyrus Surface Z-Heightmap & Curvature Warping Mesh
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.perf_counter()

    # Transfer (1, 1, D, H, W) tensor to Apple M5 Max Unified Memory (MPS)
    vol = torch.from_numpy(slab_uint8).to(device=device, dtype=torch.float32) / 255.0
    vol_5d = vol.unsqueeze(0).unsqueeze(0)  # (1, 1, 65, 512, 512)

    # 1. Core Recto Lamina MIP & Mean Intensity (centered around z=32 of the 65-slice slab)
    recto_slab = vol[24:41, :, :]  # 17 slices (~40.8 um around recto surface)
    mip_proj = torch.amax(recto_slab, dim=0)
    mean_proj = torch.mean(recto_slab, dim=0)

    # 2. 3D Sobel / Scharr Crackle & Normal Phase-Contrast Filter on MPS
    # Differentiates tangential crackle edges (dx, dy) and normal phase-contrast fringes (dz)
    kz = torch.tensor([-1.0, 0.0, 1.0], device=device).view(3, 1, 1)
    ky = torch.tensor([-1.0, 0.0, 1.0], device=device).view(1, 3, 1)
    kx = torch.tensor([-1.0, 0.0, 1.0], device=device).view(1, 1, 3)
    smooth = torch.tensor([1.0, 2.0, 1.0], device=device)

    sobel_z = (kz * smooth.view(1, 3, 1) * smooth.view(1, 1, 3)).unsqueeze(0).unsqueeze(0) / 16.0
    sobel_y = (smooth.view(3, 1, 1) * ky * smooth.view(1, 1, 3)).unsqueeze(0).unsqueeze(0) / 16.0
    sobel_x = (smooth.view(3, 1, 1) * smooth.view(1, 3, 1) * kx).unsqueeze(0).unsqueeze(0) / 16.0

    gz = F.conv3d(vol_5d, sobel_z, padding=1).squeeze()
    gy = F.conv3d(vol_5d, sobel_y, padding=1).squeeze()
    gx = F.conv3d(vol_5d, sobel_x, padding=1).squeeze()

    # Tangential crackle magnitude + normal phase-contrast boundary jump across recto interface (z=26..38)
    grad_tangential = torch.sqrt(gx[26:39] ** 2 + gy[26:39] ** 2 + 1e-8)
    crackle_proj = torch.amax(grad_tangential, dim=0) + 0.65 * torch.amax(torch.abs(gz[26:39]), dim=0)

    # 3. Micro-Topographic Papyrus Fiber Surface Z-Centroid Heightmap z_surf(y, x)
    # Softmax-weighted Z-coordinate across the central papyrus sheet (z=16..49)
    z_idx = torch.arange(16, 49, device=device, dtype=torch.float32).view(-1, 1, 1)
    weights = F.softmax(vol[16:49] * 8.0, dim=0)
    z_heightmap = torch.sum(weights * z_idx, dim=0)  # in voxel units (2.399 um/voxel)

    if device.type == "mps":
        torch.mps.synchronize()
    mps_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[MPS Engine] Computed 3D MIP, 3D Sobel Crackle, and Surface Heightmap on {device} in {mps_ms:.2f} ms")

    return {
        "device": str(device),
        "mps_compute_ms": round(mps_ms, 2),
        "mip": mip_proj.cpu().numpy(),
        "mean": mean_proj.cpu().numpy(),
        "crackle": crackle_proj.cpu().numpy(),
        "z_heightmap": z_heightmap.cpu().numpy(),
        "z_profile_mean": vol.mean(dim=(1, 2)).cpu().numpy(),
        "z_profile_std": vol.std(dim=(1, 2)).cpu().numpy(),
    }


def render_diagnostic_plate(
    slab_uint8: np.ndarray,
    ink_gt: np.ndarray,
    proj: dict,
    telemetry: dict,
    out_path: str,
):
    """Render a 6-panel publication-grade diagnostic plate for Phase 3."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 11), dpi=160)
    fig.suptitle(
        f"VESUVIUS CHALLENGE PHASE 3: ZERO-DISK-BLOAT OME-ZARR STREAMING & 3D MPS PROJECTIONS\n"
        f"Scroll: {telemetry['scroll_id']} (Winding w029) | Resolution: {telemetry['voxel_size_um']} µm ({telemetry['beam_energy_keV']} keV ESRF) | "
        f"Slab: {telemetry['slab_shape']} ({telemetry['slab_megabytes']} MB in {telemetry['stream_seconds']}s) | Apple M5 Max MPS: {proj['mps_compute_ms']} ms",
        fontsize=12.5,
        fontweight="bold",
        color="#f0e6d2",
        y=0.97,
    )

    voxel_um = telemetry["voxel_size_um"]
    lat_um = telemetry.get("lateral_pixel_um", voxel_um)
    extent_mm = [0, slab_uint8.shape[2] * lat_um / 1000.0, slab_uint8.shape[1] * lat_um / 1000.0, 0]

    # Panel 1: Recto Maximum Intensity Projection (MIP) showing horizontal & vertical papyrus fibers
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(proj["mip"], cmap="bone", extent=extent_mm)
    ax1.set_title("A. Multi-Layer Recto MIP (z=24..40)\nCyperus Papyrus Vascular Fiber Bundles (9.8×9.8 mm)", fontsize=10.5, color="#8ecae6")
    ax1.set_xlabel("Horizontal Sheet Axis X (mm)")
    ax1.set_ylabel("Vertical Sheet Axis Y (mm)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Normalized X-Ray Attenuation")

    # Panel 2: 3D Sobel Crackle & Normal Phase-Contrast Projection on MPS
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(proj["crackle"], cmap="inferno", extent=extent_mm)
    ax2.set_title("B. 3D Sobel Crackle & Phase-Contrast Projection\n(||∇xy I|| + |∂z I| on Apple M5 Max MPS)", fontsize=10.5, color="#ffb703")
    ax2.set_xlabel("Horizontal Sheet Axis X (mm)")
    ax2.set_ylabel("Vertical Sheet Axis Y (mm)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="3D Phase/Crackle Gradient")

    # Panel 3: 3D Papyrus Fiber Surface Mesh & Micro-Topographic Relief
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    step = 8
    Y_grid, X_grid = np.mgrid[0:512:step, 0:512:step] * (lat_um / 1000.0)
    Z_surf_um = (proj["z_heightmap"][::step, ::step] - 32.0) * voxel_um
    surf_color = proj["crackle"][::step, ::step]
    norm_c = (surf_color - surf_color.min()) / (np.ptp(surf_color) + 1e-8)
    ax3.plot_surface(
        X_grid, Y_grid, Z_surf_um,
        facecolors=plt.cm.magma(norm_c),
        rstride=1, cstride=1, linewidth=0.15, antialiased=True, shade=False
    )
    ax3.set_title("C. 3D Papyrus Surface Mesh & Crackle Relief\n(Micro-Topographic Normal Offset Δz in µm)", fontsize=10.5, color="#fb8500")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.set_zlabel("Δz (µm)")
    ax3.view_init(elev=38, azim=-55)

    # Panel 4: Orthogonal B-Scan Cross-Section (Z vs X at Y=256) showing Papyrus Lamina Thickness
    ax4 = fig.add_subplot(2, 3, 4)
    bscan = slab_uint8[:, 256, :]
    z_extent = [0, slab_uint8.shape[2] * lat_um / 1000.0, (slab_uint8.shape[0] - 32) * voxel_um, -32 * voxel_um]
    im4 = ax4.imshow(bscan, cmap="viridis", aspect="auto", extent=z_extent)
    ax4.axhline(0.0, color="#ff006e", linestyle="--", linewidth=1.2, label="Recto Surface Center (z=32)")
    ax4.plot(
        np.linspace(0, extent_mm[1], 512),
        (proj["z_heightmap"][256, :] - 32.0) * voxel_um,
        color="#00f5d4", linewidth=1.3, label="Tracked Micro-Topography z_surf(x)"
    )
    ax4.set_title("D. Volumetric Cross-Section B-Scan I(z, y=256, x)\nRecto/Verso Parenchyma & Carbon Ink Lamina", fontsize=10.5, color="#00f5d4")
    ax4.set_xlabel("Horizontal Sheet Axis X (mm)")
    ax4.set_ylabel("Normal Depth Offset Δz (µm)")
    ax4.legend(loc="upper right", fontsize=8)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="Raw CT uint8")

    # Panel 5: Normal Depth Attenuation & Phase-Contrast Profile Across the 65 Z-Slices
    ax5 = fig.add_subplot(2, 3, 5)
    z_coords_um = (np.arange(slab_uint8.shape[0]) - 32.0) * voxel_um
    ax5.plot(z_coords_um, proj["z_profile_mean"], color="#00bbf9", linewidth=2.2, label="Mean CT Attenuation μ(z)")
    ax5.fill_between(
        z_coords_um,
        proj["z_profile_mean"] - proj["z_profile_std"],
        proj["z_profile_mean"] + proj["z_profile_std"],
        color="#00bbf9", alpha=0.2, label="±1σ Fiber/Crackle Variance"
    )
    ax5.axvspan(-19.2, 19.2, color="#f15bb5", alpha=0.18, label="Ink-Bearing Recto Stratum (±19.2 µm)")
    ax5.axvline(0.0, color="#fee440", linestyle="--", linewidth=1.2)
    ax5.set_title("E. Normal Depth Profile Across 65 Z-Layers\nAttenuation & Phase-Contrast Envelope", fontsize=10.5, color="#f15bb5")
    ax5.set_xlabel("Normal Depth Offset Δz from Recto Surface (µm)")
    ax5.set_ylabel("Normalized CT Intensity")
    ax5.grid(True, alpha=0.2)
    ax5.legend(loc="upper right", fontsize=8)

    # Panel 6: Aligned Ground-Truth Ink Labels Overlay on Papyrus Fiber MIP
    ax6 = fig.add_subplot(2, 3, 6)
    rgb = np.stack([proj["mip"]]*3, axis=-1)
    rgb = (rgb - rgb.min()) / (np.ptp(rgb) + 1e-8)
    # Highlight ink_gt in gold/magenta overlay
    mask = ink_gt > 0
    rgb[mask, 0] = np.clip(rgb[mask, 0] * 0.35 + 0.65, 0, 1)
    rgb[mask, 1] = np.clip(rgb[mask, 1] * 0.35 + 0.15, 0, 1)
    rgb[mask, 2] = np.clip(rgb[mask, 2] * 0.35 + 0.50, 0, 1)
    ax6.imshow(rgb, extent=extent_mm)
    ax6.set_title(
        f"F. Aligned Ground-Truth Greek Ink (`inklabels.zarr`)\n"
        f"Multi-Line Column on PHerc. 1667 (Coverage: {telemetry['ink_gt_positive_fraction']*100:.1f}%)",
        fontsize=10.5, color="#ff99c8"
    )
    ax6.set_xlabel("Horizontal Sheet Axis X (mm)")
    ax6.set_ylabel("Vertical Sheet Axis Y (mm)")

    plt.tight_layout(rect=[0, 0.01, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[Streamer] Saved Phase 3 Diagnostic Plate to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Vesuvius Challenge OME-Zarr Sub-Volume Streamer")
    parser.add_argument("--y0", type=int, default=2176, help="Y start coordinate")
    parser.add_argument("--x0", type=int, default=2176, help="X start coordinate")
    parser.add_argument("--z-start", type=int, default=22, help="Start Z slice (centered so 22..87 = 65 slices around z=54)")
    parser.add_argument("--z-depth", type=int, default=65, help="Number of Z slices (default: 65)")
    parser.add_argument("--hw", type=int, default=512, help="Spatial patch height/width (default: 512)")
    parser.add_argument(
        "--out-npz",
        default="/Users/macstudio/Documents/Codex/vesuvius-challenge-lab/outputs/pherc1667_column_slab_65x512x512.npz",
    )
    parser.add_argument(
        "--out-plate",
        default="/Users/macstudio/Documents/Codex/vesuvius-challenge-lab/outputs/phase3_zarr_streaming_diagnostic_plate.png",
    )
    parser.add_argument(
        "--out-json",
        default="/Users/macstudio/Documents/Codex/vesuvius-challenge-lab/outputs/phase3_streaming_telemetry.json",
    )
    args = parser.parse_args()

    slab_uint8, ink_gt, sup_mask, official_ink_patch, telemetry = stream_subvolume(
        z_start=args.z_start, z_depth=args.z_depth, y0=args.y0, x0=args.x0, hw=args.hw
    )
    proj = compute_mps_projections(slab_uint8)
    telemetry["mps_device"] = proj["device"]
    telemetry["mps_compute_ms"] = proj["mps_compute_ms"]

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        slab_uint8=slab_uint8,
        ink_gt=ink_gt,
        sup_mask=sup_mask,
        official_ink_patch=official_ink_patch,
        mip=proj["mip"],
        crackle=proj["crackle"],
        z_heightmap=proj["z_heightmap"],
    )
    print(f"[Streamer] Cached 65x512x512 slab + GT arrays to {args.out_npz}")

    with open(args.out_json, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"[Streamer] Saved streaming telemetry to {args.out_json}")

    render_diagnostic_plate(slab_uint8, ink_gt, proj, telemetry, args.out_plate)


if __name__ == "__main__":
    main()
