import os
import torch
import numpy as np
import trimesh
from skimage import measure
import argparse
import matplotlib.pyplot as plt

from model import TactileSDF
from dataset import TactileSDFDataset

def reconstruct_mesh(model, contacts, device, resolution=64):
    """
    Predict SDF on a grid and extract mesh via Marching Cubes.
    """
    model.eval()
    # Create grid
    grid_pts = np.meshgrid(
        np.linspace(-1.1, 1.1, resolution),
        np.linspace(-1.1, 1.1, resolution),
        np.linspace(-1.1, 1.1, resolution),
        indexing='ij'
    )
    grid_pts = np.stack(grid_pts, axis=-1).reshape(-1, 3).astype(np.float32)
    grid_pts = torch.from_numpy(grid_pts).to(device).unsqueeze(0)
    
    # Predict in chunks to avoid OOM
    chunk_size = 32768
    sdf_values = []
    
    with torch.no_grad():
        latent = model.encoder(contacts)
        for i in range(0, grid_pts.shape[1], chunk_size):
            chunk = grid_pts[:, i:i+chunk_size, :]
            sdf = model.decoder(chunk, latent)
            sdf_values.append(sdf.cpu().numpy())
            
    sdf_grid = np.concatenate(sdf_values, axis=1).reshape(resolution, resolution, resolution)
    
    # Marching Cubes
    try:
        verts, faces, normals, values = measure.marching_cubes(sdf_grid, level=0.0)
        # Rescale back to [-1.1, 1.1]
        verts = verts / (resolution - 1) * 2.2 - 1.1
        return trimesh.Trimesh(vertices=verts, faces=faces)
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="../grasp-dataset-gen/output_hf")
    parser.add_argument("--index", type=int, default=0, help="Index of sample in test set")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = TactileSDF().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"✅ Loaded model from {args.checkpoint}")
    
    # Load dataset
    dataset = TactileSDFDataset(dataset_dir=args.dataset_dir, split="test")
    sample = dataset[args.index]
    contacts = sample["contacts"].to(device).unsqueeze(0)
    
    print(f"🔍 Reconstructing: {sample['name']} ({sample['category']})")
    
    # Reconstruct
    mesh = reconstruct_mesh(model, contacts, device)
    
    if mesh:
        # Load GT mesh for comparison if available
        # (Normalization is key here)
        print("📊 Visualization ready. (In a real script we would open a window or save)")
        mesh.export("reconstruction.obj")
        print("💾 Exported to reconstruction.obj")
    else:
        print("❌ Marching cubes failed (surface might be outside the grid)")

if __name__ == "__main__":
    main()
