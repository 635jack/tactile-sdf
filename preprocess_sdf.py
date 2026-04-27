"""
Preprocess GLB meshes into ground truth SDF data.

For each object in the grasp dataset:
1. Load the GLB mesh via trimesh
2. Normalize to unit bounding box [-1, 1]³
3. Sample query points (uniform + near-surface)
4. Compute signed distance at each query point
5. Save as .npz for fast training loading
"""
import os
import json
import argparse
import numpy as np
import trimesh
from pathlib import Path
import gc
from tqdm import tqdm


def normalize_mesh(mesh: trimesh.Trimesh):
    """Center and scale mesh to fit in [-1, 1]³. Returns (mesh, center, scale)."""
    center = mesh.bounds.mean(axis=0)
    mesh.vertices -= center
    scale = np.abs(mesh.vertices).max()
    mesh.vertices /= scale
    return mesh, center, scale


def sample_query_points(mesh: trimesh.Trimesh, n_total: int = 100_000,
                        near_surface_ratio: float = 0.5,
                        noise_std: float = 0.05, seed: int = 42):
    """
    Sample query points for SDF supervision.

    50% uniform in [-1.1, 1.1]³
    50% near the mesh surface with Gaussian noise
    """
    rng = np.random.RandomState(seed)

    n_uniform = n_total - int(n_total * near_surface_ratio)
    n_near = n_total - n_uniform

    # Uniform samples in padded bounding box
    uniform_pts = rng.uniform(-1.1, 1.1, (n_uniform, 3)).astype(np.float32)

    # Near-surface samples
    surface_pts, _ = trimesh.sample.sample_surface(mesh, n_near)
    noise = rng.randn(n_near, 3).astype(np.float32) * noise_std
    near_pts = (surface_pts + noise).astype(np.float32)

    return np.concatenate([uniform_pts, near_pts], axis=0)


def compute_sdf(mesh: trimesh.Trimesh, query_points: np.ndarray) -> np.ndarray:
    """Compute signed distance from query points to mesh surface."""
    # trimesh signed distance: negative inside, positive outside
    # We follow the convention: negative = inside
    sdf = trimesh.proximity.signed_distance(mesh, query_points)
    # trimesh returns positive inside — flip to match standard SDF convention
    # Actually trimesh.proximity.signed_distance returns positive=inside for watertight
    # We want: negative=inside, positive=outside (standard SDF)
    return -sdf.astype(np.float32)


def process_object(glb_path: str, output_path: str, n_points: int = 100_000):
    """Process a single GLB file into SDF ground truth."""
    # Load mesh
    scene = trimesh.load(glb_path, force='scene')

    if isinstance(scene, trimesh.Scene):
        meshes = []
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            print(f"  ⚠ No meshes found in {glb_path}, skipping.")
            return False
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene

    if not isinstance(mesh, trimesh.Trimesh):
        print(f"  ⚠ Could not extract trimesh from {glb_path}, skipping.")
        return False

    # Normalize
    mesh, center, scale = normalize_mesh(mesh)

    # Decimate if too large to speed up SDF calculation
    if len(mesh.faces) > 20000:
        name = Path(glb_path).stem
        tqdm.write(f"  ⚡ Decimating {name} ({len(mesh.faces):,} faces → 20,000)")
        mesh = mesh.simplify_quadric_decimation(face_count=20000)

    # Make watertight if possible
    if not mesh.is_watertight:
        # Try to fix
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

    # Sample query points
    query_points = sample_query_points(mesh, n_total=n_points)

    # Compute SDF
    sdf_values = compute_sdf(mesh, query_points)

    # Also sample surface points for contact normalization reference
    surface_pts, face_indices = trimesh.sample.sample_surface(mesh, 2048)

    # Save
    np.savez_compressed(
        output_path,
        query_points=query_points.astype(np.float32),
        sdf_values=sdf_values.astype(np.float32),
        surface_points=surface_pts.astype(np.float32),
        vertices=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.int32),
        center=center.astype(np.float64),
        scale=np.float64(scale),
    )

    # Explicit cleanup to save RAM
    del mesh
    if 'scene' in locals(): del scene
    gc.collect()

    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess GLB meshes to SDF cache")
    parser.add_argument("--glb_dir", type=str,
                        default="../grasp-dataset-gen/data/objaverse",
                        help="Directory containing .glb files")
    parser.add_argument("--dataset_dir", type=str,
                        default="../grasp-dataset-gen/output_hf",
                        help="Directory containing dataset index")
    parser.add_argument("--sdf_cache_dir", type=str,
                        default="data/sdf_cache",
                        help="Output directory for SDF .npz files")
    parser.add_argument("--n_points", type=int, default=100_000,
                        help="Number of query points per object")
    args = parser.parse_args()

    os.makedirs(args.sdf_cache_dir, exist_ok=True)

    # Load dataset index to know which objects to process
    index_path = os.path.join(args.dataset_dir, "dataset_index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)

    objects = index["objects"]
    print(f"📐 Processing {len(objects)} objects → SDF ground truth")
    print(f"   GLB dir: {args.glb_dir}")
    print(f"   Output:  {args.sdf_cache_dir}")
    print(f"   Points:  {args.n_points:,}")
    print()

    success, failed = 0, 0
    for obj in tqdm(objects, desc="Computing SDF"):
        name = obj["mesh"]
        glb_path = os.path.join(args.glb_dir, f"{name}.glb")
        output_path = os.path.join(args.sdf_cache_dir, f"{name}.npz")

        if os.path.exists(output_path):
            tqdm.write(f"  ✓ {name} (cached)")
            success += 1
            continue

        if not os.path.exists(glb_path):
            tqdm.write(f"  ✗ {name} — GLB not found at {glb_path}")
            failed += 1
            continue

        try:
            ok = process_object(glb_path, output_path, args.n_points)
            if ok:
                success += 1
                tqdm.write(f"  ✓ {name}")
            else:
                failed += 1
        except Exception as e:
            tqdm.write(f"  ✗ {name} — {e}")
            failed += 1

    print(f"\n✅ Done: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
