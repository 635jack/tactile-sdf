"""
PyTorch Dataset for tactile SDF reconstruction.

Each sample consists of:
- Contact features: (N_contacts, 9) — position + normal + tangent
- Query points: (N_query, 3)
- SDF values: (N_query,)
- Metadata: object name, category, strategy
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional


CATEGORIES = ["bottle", "cup", "hammer", "screwdriver", "wrench"]
FINGER_TO_IDX = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4, "palm": 5}


class TactileSDFDataset(Dataset):
    """
    Dataset pairing tactile grasp contacts with SDF ground truth.

    Args:
        dataset_dir: Path to grasp-dataset-gen output (e.g., output_hf/)
        sdf_cache_dir: Path to precomputed SDF .npz files
        split: 'train' or 'test'
        n_query_points: Number of query points to sample per item
        strategies: Which grasp strategies to include
        test_ratio: Fraction of objects per category for test
        seed: Random seed for split
    """

    def __init__(
        self,
        dataset_dir: str = "../grasp-dataset-gen/output_hf",
        sdf_cache_dir: str = "data/sdf_cache",
        split: str = "train",
        n_query_points: int = 2048,
        strategies: Optional[List[str]] = None,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.sdf_cache_dir = sdf_cache_dir
        self.split = split
        self.n_query_points = n_query_points
        self.strategies = strategies or ["front_back", "left_right", "right_left"]

        # Load dataset index
        index_path = os.path.join(dataset_dir, "dataset_index.json")
        with open(index_path, 'r') as f:
            index = json.load(f)

        # Split objects by category
        rng = np.random.RandomState(seed)
        train_objects, test_objects = set(), set()

        for cat in CATEGORIES:
            cat_objs = [o["mesh"] for o in index["objects"]
                        if o["mesh"].startswith(cat)]
            rng.shuffle(cat_objs)
            n_test = max(1, int(len(cat_objs) * test_ratio))
            test_objects.update(cat_objs[:n_test])
            train_objects.update(cat_objs[n_test:])

        selected = train_objects if split == "train" else test_objects

        # Build sample list: (object_name, strategy)
        self.samples = []
        for obj in index["objects"]:
            name = obj["mesh"]
            if name not in selected:
                continue
            # Check SDF cache exists
            sdf_path = os.path.join(sdf_cache_dir, f"{name}.npz")
            if not os.path.exists(sdf_path):
                continue
            for strat in self.strategies:
                if strat in obj["grasps"]:
                    grasp_info = obj["grasps"][strat]
                    if grasp_info["n_contacts"] >= 3:  # need enough contacts
                        self.samples.append((name, strat))

        print(f"[{split.upper()}] Loaded {len(self.samples)} samples "
              f"({len(selected)} objects × {len(self.strategies)} strategies)")

        # Cache for loaded SDF data (limited to 32 objects to save RAM)
        self._sdf_cache: Dict[str, dict] = {}
        self._cache_order: List[str] = []
        self.max_cache_size = 32

    def _get_category(self, name: str) -> str:
        """Extract category from object name like 'cup_01_68e4c9'."""
        for cat in CATEGORIES:
            if name.startswith(cat):
                return cat
        return "unknown"

    def _load_sdf(self, name: str) -> dict:
        """Load and cache SDF data for an object with LRU policy."""
        if name not in self._sdf_cache:
            # Check cache size
            if len(self._sdf_cache) >= self.max_cache_size:
                oldest = self._cache_order.pop(0)
                del self._sdf_cache[oldest]

            path = os.path.join(self.sdf_cache_dir, f"{name}.npz")
            data = np.load(path)
            self._sdf_cache[name] = {
                "query_points": data["query_points"],
                "sdf_values": data["sdf_values"],
                "center": data["center"],
                "scale": float(data["scale"]),
                "vertices": data["vertices"],
                "faces": data["faces"],
            }
            self._cache_order.append(name)
        return self._sdf_cache[name]

    def _load_contacts(self, name: str, strategy: str) -> np.ndarray:
        """Load contact points for an object/strategy pair."""
        npz_path = os.path.join(self.dataset_dir, name, f"grasp_{strategy}.npz")
        data = np.load(npz_path, allow_pickle=True)

        positions = data["positions"]   # (N, 3)
        normals = data["normals"]       # (N, 3)
        tangents = data["tangents"]     # (N, 3)

        # Stack into (N, 9)
        contacts = np.hstack([positions, normals, tangents]).astype(np.float32)
        return contacts

    def _normalize_contacts(self, contacts: np.ndarray,
                            center: np.ndarray, scale: float) -> np.ndarray:
        """Normalize contact positions to match mesh normalization."""
        normalized = contacts.copy()
        # Normalize position (first 3 columns)
        normalized[:, :3] = (normalized[:, :3] - center) / scale
        # Normals and tangents are already unit vectors, no normalization needed
        return normalized

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        name, strategy = self.samples[idx]

        # Load SDF ground truth
        sdf_data = self._load_sdf(name)

        # Load contacts
        contacts = self._load_contacts(name, strategy)

        # Data Augmentation: Add small jitter to contact positions (first 3 columns)
        # This helps prevent overfitting on specific contact coordinates
        jitter = np.random.normal(0, 0.005, size=(contacts.shape[0], 3)).astype(np.float32)
        contacts[:, :3] += jitter
        contacts = self._normalize_contacts(
            contacts, sdf_data["center"], sdf_data["scale"]
        )

        # Pad/truncate contacts to fixed size (6)
        max_contacts = 6
        if len(contacts) < max_contacts:
            pad = np.zeros((max_contacts - len(contacts), 9), dtype=np.float32)
            contacts = np.concatenate([contacts, pad], axis=0)
        else:
            contacts = contacts[:max_contacts]

        # Random subsample of query points
        n_total = len(sdf_data["query_points"])
        indices = np.random.choice(n_total, self.n_query_points, replace=False)
        query_pts = sdf_data["query_points"][indices]
        sdf_vals = sdf_data["sdf_values"][indices]

        return {
            "contacts": torch.from_numpy(contacts),                # (6, 9)
            "query_points": torch.from_numpy(query_pts),           # (N, 3)
            "sdf_values": torch.from_numpy(sdf_vals),              # (N,)
            "name": name,
            "category": self._get_category(name),
            "strategy": strategy,
        }


def get_dataloaders(batch_size: int = 8, n_query: int = 2048,
                    num_workers: int = 0, **kwargs):
    """Create train and test dataloaders."""
    train_ds = TactileSDFDataset(split="train", n_query_points=n_query, **kwargs)
    test_ds = TactileSDFDataset(split="test", n_query_points=n_query, **kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    return train_loader, test_loader
