"""
Training loop for PointNet + SIREN tactile SDF reconstruction.

Trains the model with:
- L1 SDF loss on random query points
- Contact constraint loss (SDF ≈ 0 at contact positions)
- Eikonal regularization (|∇SDF| ≈ 1 near surface)

Logs training dynamics (loss curves, metrics) and saves checkpoints.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from model import TactileSDF
from dataset import TactileSDFDataset, get_dataloaders, CATEGORIES


def compute_iou(pred_sdf: torch.Tensor, gt_sdf: torch.Tensor,
                threshold: float = 0.0) -> float:
    """
    Compute volumetric IoU between predicted and GT SDF.
    Inside = SDF < threshold.
    """
    pred_inside = (pred_sdf < threshold).float()
    gt_inside = (gt_sdf < threshold).float()
    intersection = (pred_inside * gt_inside).sum()
    union = ((pred_inside + gt_inside) > 0).float().sum()
    if union < 1:
        return 1.0
    return (intersection / union).item()


def contact_loss(model: TactileSDF, contacts: torch.Tensor) -> torch.Tensor:
    """
    SDF should be ≈ 0 at contact positions.
    contacts: (B, N, 9) — first 3 columns are positions
    """
    contact_pos = contacts[:, :, :3]  # (B, N, 3)
    latent = model.encoder(contacts)
    sdf_at_contacts = model.decoder(contact_pos, latent)  # (B, N, 1)
    return sdf_at_contacts.abs().mean()


def eikonal_loss(model: TactileSDF, contacts: torch.Tensor,
                 query_points: torch.Tensor) -> torch.Tensor:
    """
    Eikonal regularization: |∇SDF| should be ≈ 1 everywhere.
    We compute it on a subset of query points near the surface.
    """
    query_points = query_points.detach().requires_grad_(True)
    latent = model.encoder(contacts)
    sdf = model.decoder(query_points, latent)

    grad = torch.autograd.grad(
        outputs=sdf,
        inputs=query_points,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True,
        retain_graph=True,
    )[0]

    # |∇SDF| - 1 should be 0
    grad_norm = grad.norm(dim=-1)
    return ((grad_norm - 1.0) ** 2).mean()


def train_one_epoch(model, train_loader, optimizer, device, epoch,
                    lambda_contact=0.1, lambda_eikonal=0.01):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_sdf_loss = 0
    total_contact_loss = 0
    total_eikonal_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
    for batch in pbar:
        contacts = batch["contacts"].to(device)        # (B, 6, 9)
        query_pts = batch["query_points"].to(device)    # (B, N, 3)
        sdf_gt = batch["sdf_values"].to(device)         # (B, N)

        optimizer.zero_grad()

        # Forward pass
        sdf_pred, latent = model(contacts, query_pts)
        sdf_pred = sdf_pred.squeeze(-1)  # (B, N)

        # L1 SDF loss
        loss_sdf = F.l1_loss(sdf_pred, sdf_gt)

        # Contact constraint
        loss_contact = contact_loss(model, contacts)

        # Eikonal (on a small subset for speed)
        n_eik = min(256, query_pts.shape[1])
        eik_idx = torch.randperm(query_pts.shape[1])[:n_eik]
        loss_eikonal = eikonal_loss(model, contacts, query_pts[:, eik_idx, :])

        # Total loss
        loss = loss_sdf + lambda_contact * loss_contact + lambda_eikonal * loss_eikonal

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_sdf_loss += loss_sdf.item()
        total_contact_loss += loss_contact.item()
        total_eikonal_loss += loss_eikonal.item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "sdf": f"{loss_sdf.item():.4f}",
            "cntct": f"{loss_contact.item():.4f}",
        })

    return {
        "loss": total_loss / n_batches,
        "sdf_loss": total_sdf_loss / n_batches,
        "contact_loss": total_contact_loss / n_batches,
        "eikonal_loss": total_eikonal_loss / n_batches,
    }


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    total_iou = 0
    n_batches = 0
    per_category = defaultdict(lambda: {"iou": [], "loss": []})

    for batch in test_loader:
        contacts = batch["contacts"].to(device)
        query_pts = batch["query_points"].to(device)
        sdf_gt = batch["sdf_values"].to(device)
        categories = batch["category"]

        sdf_pred, _ = model(contacts, query_pts)
        sdf_pred = sdf_pred.squeeze(-1)

        loss = F.l1_loss(sdf_pred, sdf_gt)
        total_loss += loss.item()

        # Per-sample IoU
        for i in range(sdf_pred.shape[0]):
            iou = compute_iou(sdf_pred[i], sdf_gt[i])
            total_iou += iou
            cat = categories[i]
            per_category[cat]["iou"].append(iou)
            per_category[cat]["loss"].append(
                F.l1_loss(sdf_pred[i], sdf_gt[i]).item()
            )

        n_batches += 1

    n_samples = sum(len(v["iou"]) for v in per_category.values())
    cat_metrics = {}
    for cat, vals in per_category.items():
        cat_metrics[cat] = {
            "iou": float(np.mean(vals["iou"])),
            "loss": float(np.mean(vals["loss"])),
        }

    return {
        "loss": total_loss / max(n_batches, 1),
        "iou": total_iou / max(n_samples, 1),
        "per_category": cat_metrics,
    }


def plot_training_curves(history: dict, output_dir: str):
    """Generate training dynamics plots."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Dynamics — PointNet + SIREN Tactile SDF",
                 fontsize=14, fontweight="bold")

    # 1) Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train", color="#4C72B0",
            linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val", color="#DD8452",
            linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 2) Component losses
    ax = axes[0, 1]
    ax.plot(epochs, history["sdf_loss"], label="SDF L1", color="#4C72B0",
            linewidth=2)
    ax.plot(epochs, history["contact_loss"], label="Contact", color="#55A868",
            linewidth=2)
    ax.plot(epochs, history["eikonal_loss"], label="Eikonal", color="#C44E52",
            linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components (Train)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 3) IoU
    ax = axes[1, 0]
    ax.plot(epochs, history["val_iou"], label="Val IoU", color="#8172B2",
            linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Volumetric IoU (Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 4) Per-category IoU (last recorded)
    ax = axes[1, 1]
    if history["per_category_iou"]:
        last_cat = history["per_category_iou"][-1]
        cats = sorted(last_cat.keys())
        ious = [last_cat[c] for c in cats]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
        bars = ax.bar(cats, ious, color=colors[:len(cats)], alpha=0.85)
        ax.set_ylabel("IoU")
        ax.set_title("Per-Category IoU (Final)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, ious):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150,
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "training_curves.pdf"),
                bbox_inches='tight')
    plt.close()
    print(f"📊 Saved training curves to {output_dir}/training_curves.png")


def plot_per_category_evolution(history: dict, output_dir: str):
    """Plot IoU evolution per category over training."""
    if not history["per_category_iou"]:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"bottle": "#4C72B0", "cup": "#DD8452", "hammer": "#55A868",
              "screwdriver": "#C44E52", "wrench": "#8172B2"}

    all_cats = set()
    for entry in history["per_category_iou"]:
        all_cats.update(entry.keys())

    for cat in sorted(all_cats):
        ious = []
        for entry in history["per_category_iou"]:
            ious.append(entry.get(cat, 0.0))
        ax.plot(range(1, len(ious)+1), ious, label=cat,
                color=colors.get(cat, "#333333"), linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Per-Category IoU Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_category_iou.png"), dpi=150,
                bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "per_category_iou.pdf"),
                bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Tactile SDF model")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_query", type=int, default=2048)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_siren_layers", type=int, default=4)
    parser.add_argument("--omega_0", type=float, default=30.0)
    parser.add_argument("--lambda_contact", type=float, default=0.1)
    parser.add_argument("--lambda_eikonal", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--dataset_dir", type=str,
                        default="../grasp-dataset-gen/output_hf")
    parser.add_argument("--sdf_cache_dir", type=str,
                        default="data/sdf_cache")
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🔥 Using CUDA")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    # Output directory
    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["device"] = str(device)
    config["run_dir"] = run_dir
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # Data
    print("\n📂 Loading dataset...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        n_query=args.n_query,
        dataset_dir=args.dataset_dir,
        sdf_cache_dir=args.sdf_cache_dir,
    )

    if len(train_loader) == 0:
        print("❌ No training data! Run preprocess_sdf.py first.")
        sys.exit(1)

    # Model
    model = TactileSDF(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_siren_layers=args.n_siren_layers,
        omega_0=args.omega_0,
    ).to(device)

    n_params = TactileSDF.count_parameters(model)
    print(f"\n🧠 Model: {n_params:,} trainable parameters")
    print(f"   Latent dim: {args.latent_dim}")
    print(f"   SIREN layers: {args.n_siren_layers} × {args.hidden_dim}")
    print(f"   ω₀: {args.omega_0}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training history
    history = {
        "train_loss": [], "val_loss": [], "val_iou": [],
        "sdf_loss": [], "contact_loss": [], "eikonal_loss": [],
        "lr": [], "per_category_iou": [],
    }

    best_iou = 0.0
    print(f"\n🚀 Training for {args.epochs} epochs")
    print(f"   Batch size: {args.batch_size}")
    print(f"   LR: {args.lr} → 1e-6 (cosine)")
    print(f"   Output: {run_dir}")
    print()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            lambda_contact=args.lambda_contact,
            lambda_eikonal=args.lambda_eikonal,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["sdf_loss"].append(train_metrics["sdf_loss"])
        history["contact_loss"].append(train_metrics["contact_loss"])
        history["eikonal_loss"].append(train_metrics["eikonal_loss"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == 1:
            val_metrics = evaluate(model, test_loader, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_iou"].append(val_metrics["iou"])
            cat_iou = {c: v["iou"] for c, v in val_metrics["per_category"].items()}
            history["per_category_iou"].append(cat_iou)

            # Log
            cat_str = " | ".join(f"{c}: {v:.3f}" for c, v in sorted(cat_iou.items()))
            print(f"  [Epoch {epoch:03d}] Train L1={train_metrics['sdf_loss']:.4f} | "
                  f"Val L1={val_metrics['loss']:.4f} | "
                  f"IoU={val_metrics['iou']:.3f}")
            if cat_str:
                print(f"              {cat_str}")

            # Save best
            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                torch.save(model.state_dict(),
                           os.path.join(run_dir, "best_model.pt"))
                print(f"              ★ New best IoU: {best_iou:.3f}")
        else:
            # Fill with last value for plotting
            if history["val_loss"]:
                history["val_loss"].append(history["val_loss"][-1])
                history["val_iou"].append(history["val_iou"][-1])
                history["per_category_iou"].append(
                    history["per_category_iou"][-1])
            else:
                history["val_loss"].append(train_metrics["loss"])
                history["val_iou"].append(0.0)
                history["per_category_iou"].append({})

        scheduler.step()

        # Save checkpoint periodically
        if epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, os.path.join(run_dir, f"checkpoint_{epoch:04d}.pt"))

    # Final save
    torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pt"))

    # Save history as JSON
    with open(os.path.join(run_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # Plot
    plot_training_curves(history, run_dir)
    plot_per_category_evolution(history, run_dir)

    # Also copy plots to report/figures for LaTeX
    import shutil
    fig_dir = os.path.join(os.path.dirname(__file__), "report", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for fname in ["training_curves.pdf", "per_category_iou.pdf",
                   "training_curves.png", "per_category_iou.png"]:
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(fig_dir, fname))

    print(f"\n✅ Training complete!")
    print(f"   Best IoU: {best_iou:.3f}")
    print(f"   Run dir: {run_dir}")
    print(f"   Figures copied to report/figures/")


if __name__ == "__main__":
    main()
