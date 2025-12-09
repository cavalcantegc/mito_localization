"""
mito_localization.py

Per-cell mitochondrial radial localization analysis from .lif files (MitoTracker-only, without other fluorescence probes).
Designed for single .lif file processing (one file per run).

Outputs:
 - CSV summary of per-cell metrics
 - PNG plots (one per cell) showing radial distribution of mitochondrial area
 - All outputs saved in an output folder next to the input .lif file

Usage:
    python mito_localization.py --file /path/to/your_file.lif --channel 0

Dependencies:
    pip install numpy pandas matplotlib scikit-image aicsimageio tifffile openpyxl

Notes:
 - The script estimates cell outlines from the MitoTracker signal itself.
 - It performs cell segmentation (watershed) and mitochondrial segmentation,
   then computes normalized radial distances (0=center, 1=furthest cell boundary).
 - Adjust parameters (min_cell_area, inner_frac, etc.) by command-line options.
"""

from pathlib import Path
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# skimage/scipy imports (no ambiguous overwrites)
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    binary_closing,
    disk
)
from skimage.measure import label as sk_label, regionprops
from skimage.segmentation import find_boundaries, clear_border
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

try:
    from aicsimageio import AICSImage
except Exception:
    AICSImage = None

# ---------- Utility functions ----------

def read_scene_projection(img_reader, scene_name, channel_index=0):
    """
    Return 2D max-projected image for the requested scene and channel.
    """
    img_reader.set_scene(scene_name)
    data = img_reader.get_image_data("CZYX")  # common AICSImage layout
    # data can be 3D (C,Y,X) or 4D (C,Z,Y,X)
    if data.ndim == 4:
        # C, Z, Y, X
        if channel_index >= data.shape[0]:
            raise ValueError(f"channel_index {channel_index} out of range (C={data.shape[0]})")
        ch = data[channel_index]
        img2d = ch.max(axis=0)
    elif data.ndim == 3:
        # C, Y, X
        if channel_index >= data.shape[0]:
            raise ValueError(f"channel_index {channel_index} out of range (C={data.shape[0]})")
        img2d = data[channel_index]
    else:
        raise ValueError(f"Unexpected image shape: {data.shape}")
    # normalize to uint8
    img2d = img2d.astype(np.float32)
    if img2d.max() > img2d.min():
        img2d = (img2d - img2d.min()) / (img2d.max() - img2d.min())
        img2d = (img2d * 255).astype(np.uint8)
    else:
        img2d = np.zeros_like(img2d, dtype=np.uint8)
    return img2d

def segment_cells_from_mito(mito_img,
                            gaussian_sigma=2,
                            min_cell_area=5000,
                            single_cell_per_scene=False):
    """
    Segment cells from a MitoTracker image.
    Uses smoothing + Otsu + morphological cleanup + connected-components.
    Returns labeled image (0 background).
    """
    # Smooth the image (use scipy.ndimage gaussian to avoid ambiguity)
    mito_smooth = ndi.gaussian_filter(mito_img, sigma=gaussian_sigma)

    # Threshold (Otsu) with safe fallback
    try:
        thresh = threshold_otsu(mito_smooth)
    except Exception:
        thresh = float(mito_smooth.mean() + mito_smooth.std())
    cell_mask = mito_smooth > thresh

    if cell_mask.sum() == 0:
        return np.zeros_like(cell_mask, dtype=int)

    # Morphological cleanup
    cell_mask = remove_small_objects(cell_mask, min_size=int(min_cell_area))
    cell_mask = binary_closing(cell_mask, footprint=np.ones((5, 5)))
    # fill holes using scipy ndimage (ndi.binary_fill_holes)
    try:
        cell_mask = ndi.binary_fill_holes(cell_mask)
    except Exception:
        # fallback: keep cell_mask as-is
        pass

    if cell_mask.sum() == 0:
        return np.zeros_like(cell_mask, dtype=int)

    # If user requested a single object per scene, keep only the largest connected component
    if single_cell_per_scene:
        labeled = ndi.label(cell_mask)[0]
        sizes = np.bincount(labeled.ravel())
        if sizes.size <= 1:
            return labeled
        sizes[0] = 0
        largest = np.argmax(sizes)
        cleaned = (labeled == largest)
        return ndi.label(cleaned)[0]

    # Normal mode: connected components (no watershed)
    labels_seg = ndi.label(cell_mask)[0]
    return labels_seg

def segment_mitochondria(mito_img, gaussian_sigma=1.0, min_mito_area=10):
    """
    Segment mitochondrial objects (binary mask and labeled objects).
    """
    img_smooth = gaussian(mito_img, sigma=gaussian_sigma)
    try:
        thresh = threshold_otsu(img_smooth)
    except Exception:
        thresh = float(img_smooth.mean() + img_smooth.std())
    mito_mask = img_smooth > thresh
    mito_mask = remove_small_objects(mito_mask, min_size=int(min_mito_area))
    mito_labels = sk_label(mito_mask)
    return mito_labels

def save_scene_overlay(mito_img, labeled_cells, mito_labels, out_path, title=None):

    from skimage.segmentation import mark_boundaries

    # Normalize background image
    mito_norm = mito_img.astype(float)
    if mito_norm.max() > mito_norm.min():
        mito_norm = (mito_norm - mito_norm.min()) / (mito_norm.max() - mito_norm.min())
    else:
        mito_norm = mito_norm * 0.0

    overlay = mark_boundaries(
        mito_norm,
        labeled_cells > 0,
        color=(0.8, 0.8, 0.8),
        mode='thick'
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay, cmap='gray')

    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

def save_cell_overlay(mito_img, cell_mask, mito_mask_cell, radial_norm_cell, out_path, title=None):
    """
    Save a small overlay for a single cell showing radial coloring of mitochondria inside that cell.
    """
    overlay = np.zeros((*cell_mask.shape, 3), dtype=float)
    overlay[..., 0] = radial_norm_cell  # red
    overlay[..., 2] = 1.0 - radial_norm_cell  # blue
    overlay[~mito_mask_cell] = 0.0  # only show mito pixels

    plt.figure(figsize=(4,4))
    mito_norm = mito_img.astype(float)
    if mito_norm.max() > mito_norm.min():
        mito_norm = (mito_norm - mito_norm.min()) / (mito_norm.max() - mito_norm.min())
    else:
        mito_norm = mito_norm * 0.0
    plt.imshow(mito_norm, cmap='gray')
    plt.imshow(overlay, alpha=0.7)
    cb = find_boundaries(cell_mask, mode='outer')
    plt.imshow(np.ma.masked_where(~cb, cb), cmap='spring', alpha=0.9)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_radial_histogram(mito_radii, out_path, scene_id, cell_label, bins=20):
    """
    Save radial histogram (blue->red gradient).
    """
    plt.figure(figsize=(4,3))
    if mito_radii.size == 0:
        plt.text(0.5, 0.5, 'No mitochondria detected', ha='center', va='center')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('Normalized distance')
        plt.ylabel('Fraction of mito pixels')
    else:
        hist, edges = np.histogram(mito_radii, bins=bins, range=(0,1))
        if hist.sum() > 0:
            hist = hist.astype(float) / hist.sum()
        centers = (edges[:-1] + edges[1:]) / 2.0
        colors = np.zeros((len(centers), 3))
        colors[:, 0] = centers
        colors[:, 2] = 1 - centers
        plt.bar(centers, hist, width=edges[1]-edges[0], color=colors, edgecolor='k')
        plt.xlabel('Normalized distance (0=center, 1=periphery)')
        plt.ylabel('Fraction of mito pixels')
    plt.title(f"Scene {scene_id} - Cell {cell_label}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# ---------- Main analysis ----------

def analyze_scene(mito_img, scene_id, out_dir, params):
    """
    Analyze one 2D scene and return per-cell records.
    """
    records = []
    mito_img_f = mito_img.astype(np.float32)
    if mito_img_f.max() > mito_img_f.min():
        mito_img_f = (mito_img_f - mito_img_f.min()) / (mito_img_f.max() - mito_img_f.min())
    else:
        mito_img_f = mito_img_f * 0.0

    # Segment cells & mitochondria
    # NOTE: use the normalized image for segmentation
    labeled_cells = segment_cells_from_mito(
        mito_img_f,
        gaussian_sigma=params.get('cell_gaussian_sigma', 2.0),
        min_cell_area=params.get('min_cell_area', 2000),
        single_cell_per_scene=params.get('single_cell_per_scene', False)
    )
    mito_labels = segment_mitochondria(
        mito_img_f,
        gaussian_sigma=params.get('mito_gaussian_sigma', 1.0),
        min_mito_area=params.get('min_mito_area', 10)
    )

    # Save scene-level overlay
    scene_overlay_path = out_dir / f"scene_{scene_id:03d}_overlay.png"
    save_scene_overlay(mito_img_f, labeled_cells, mito_labels, scene_overlay_path,
                       title=f"Scene {scene_id}")

    # iterate each labeled cell
    if labeled_cells.max() == 0:
        logging.info(f"Scene {scene_id}: no cells found.")
        return records

    for prop in regionprops(labeled_cells):
        cell_label = prop.label
        cell_mask = (labeled_cells == cell_label)
        area_px = int(cell_mask.sum())
        if area_px < params.get('min_cell_area', 2000):
            logging.debug(f"Scene {scene_id} Cell {cell_label}: skipped (area {area_px})")
            continue

        # centroid and per-cell radial normalization
        cy, cx = prop.centroid
        coords = np.column_stack(np.nonzero(cell_mask))
        if coords.size == 0:
            continue
        dists = np.sqrt((coords[:,0] - cy)**2 + (coords[:,1] - cx)**2)
        max_dist = dists.max() if dists.size > 0 else 1.0
        radial_norm_cell = np.zeros_like(mito_img_f, dtype=float)
        radial_norm_cell[cell_mask] = dists / max_dist
        radial_norm_cell = np.clip(radial_norm_cell, 0, 1.0)

        # mitochondria inside this cell
        mito_mask_cell = (mito_labels > 0) & cell_mask
        mito_area_px = int(mito_mask_cell.sum())

        if mito_area_px == 0:
            mean_norm_dist = np.nan
            frac_inner = 0.0
            frac_outer = 0.0
            num_mito_objects = 0
            mito_radii = np.array([], dtype=float)
        else:
            mito_coords = np.column_stack(np.nonzero(mito_mask_cell))
            mito_dists_norm = np.sqrt((mito_coords[:,0] - cy)**2 + (mito_coords[:,1] - cx)**2) / max_dist
            mito_dists_norm = np.clip(mito_dists_norm, 0, 1.0)
            mean_norm_dist = float(mito_dists_norm.mean())
            inner_thresh = params.get('inner_frac', 0.5)
            frac_inner = float((mito_dists_norm < inner_thresh).sum() / mito_dists_norm.size)
            frac_outer = 1.0 - frac_inner
            mito_labels_in_cell = np.unique(mito_labels[cell_mask & (mito_labels > 0)])
            num_mito_objects = int(len(mito_labels_in_cell))
            mito_radii = mito_dists_norm

        # Save per-cell overlays and histogram
        cell_overlay_path = out_dir / f"scene_{scene_id:03d}_cell_{cell_label:03d}_overlay.png"
        save_cell_overlay(mito_img_f, cell_mask, mito_mask_cell, radial_norm_cell,
                          cell_overlay_path, title=f"Scene {scene_id} - Cell {cell_label}")

        hist_path = out_dir / f"scene_{scene_id:03d}_cell_{cell_label:03d}_radial.png"
        save_radial_histogram(mito_radii, hist_path, scene_id, cell_label, bins=params.get('hist_bins', 20))

        rec = {
            'scene': int(scene_id),
            'cell_label': int(cell_label),
            'cell_area_px': int(area_px),
            'mito_area_px': int(mito_area_px),
            'mean_norm_dist': (mean_norm_dist if not np.isnan(mean_norm_dist) else ""),
            'frac_inner': float(frac_inner),
            'frac_outer': float(frac_outer),
            'num_mito_objects': int(num_mito_objects)
        }
        records.append(rec)

    return records

def process_lif_file(lif_path, channel_index=0, params=None):
    if params is None:
        params = {}
    lif_path = Path(lif_path)
    if not lif_path.exists():
        raise FileNotFoundError(lif_path)

    if AICSImage is None:
        raise RuntimeError("aicsimageio not available. Install with: pip install aicsimageio")

    out_root = lif_path.with_suffix('')
    output_dir = out_root.parent / (out_root.name + "_mito_localization_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    img = AICSImage(str(lif_path))
    all_records = []

    for s, scene_name in enumerate(img.scenes):
        try:
            img2d = read_scene_projection(img, scene_name, channel_index=channel_index)
        except Exception as e:
            logging.warning(f"Failed reading scene '{scene_name}': {e}")
            continue

        scene_out = output_dir / f"scene_{s:03d}"
        scene_out.mkdir(parents=True, exist_ok=True)

        recs = analyze_scene(img2d, s, scene_out, params)
        for r in recs:
            r['scene_name'] = scene_name
            r['source_file'] = lif_path.name
            r['output_subfolder'] = str(scene_out.relative_to(output_dir))
            all_records.append(r)

    if len(all_records) == 0:
        logging.warning("No records produced. Check your input and segmentation parameters.")
    else:
        df = pd.DataFrame(all_records)
        csv_path = output_dir / (lif_path.stem + "_per_cell_summary.csv")
        xlsx_path = output_dir / (lif_path.stem + "_per_cell_summary.xlsx")
        df.to_csv(csv_path, index=False)
        try:
            df.to_excel(xlsx_path, index=False)
        except Exception:
            logging.debug("Unable to write Excel file (openpyxl missing?)")
        logging.info(f"Results saved to: {output_dir}")

    return output_dir

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Per-cell mitochondrial radial localization (fixed)")
    p.add_argument('--file', '-f', required=True, help="Path to .lif file")
    p.add_argument('--channel', '-c', type=int, default=0, help="Channel index for MitoTracker (default 0)")
    p.add_argument('--min-cell-area', type=int, default=2000, help="Minimum cell area in px to keep (default 2000)")
    p.add_argument('--min-mito-area', type=int, default=10, help="Minimum mitochondrial area in px (default 10)")
    p.add_argument('--inner-frac', type=float, default=0.5, help="Normalized radius threshold for 'inner' (default 0.5)")
    p.add_argument('--cell-gaussian-sigma', type=float, default=2.0)
    p.add_argument('--mito-gaussian-sigma', type=float, default=1.0)
    # kept the watershed-related args for backward compatibility but they are not used by default
    p.add_argument('--local-max-footprint', type=int, default=9, help="footprint for local maxima in watershed (odd integer)")
    p.add_argument('--watershed-compactness', type=float, default=0.001, help="watershed compactness (0->no compactness)")
    p.add_argument('--hist-bins', type=int, default=20)
    p.add_argument('--verbose', '-v', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s: %(message)s')
    params = {
        'min_cell_area': args.min_cell_area,
        'min_mito_area': args.min_mito_area,
        'inner_frac': args.inner_frac,
        'cell_gaussian_sigma': args.cell_gaussian_sigma,
        'mito_gaussian_sigma': args.mito_gaussian_sigma,
        # watershed parameters are kept in CLI for backward compatibility but not used
        'watershed_compactness': args.watershed_compactness,
        'hist_bins': args.hist_bins
    }
    out = process_lif_file(args.file, channel_index=args.channel, params=params)
    print("Done. Output folder:", out)

if __name__ == '__main__':
    main()
