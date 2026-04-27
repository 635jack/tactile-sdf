"""
Streamlit interactive dashboard for Tactile SDF reconstruction.

Tabs:
1. Training Dynamics — animated loss curves, LR schedule, per-category IoU
2. 3D Reconstruction — side-by-side GT mesh vs predicted isosurface
3. Comparison Grid — all test objects at a glance
"""
import os
import sys
import json
import glob
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from model import TactileSDF
from dataset import TactileSDFDataset, CATEGORIES

# ─────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tactile SDF — PointNet + SIREN",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 10px 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p { margin: 0; opacity: 0.85; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

FINGER_COLORS = {
    "thumb": "#e74c3c",
    "index": "#3498db",
    "middle": "#2ecc71",
    "ring": "#f39c12",
    "pinky": "#9b59b6",
    "palm": "#1abc9c",
}


# ─────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────

@st.cache_data
def load_history(run_dir: str) -> dict:
    """Load training history."""
    path = os.path.join(run_dir, "history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_config(run_dir: str) -> dict:
    """Load training config."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_model(run_dir: str, config: dict, device: str = "cpu"):
    """Load best model from run directory."""
    model = TactileSDF(
        latent_dim=config.get("latent_dim", 256),
        hidden_dim=config.get("hidden_dim", 256),
        n_siren_layers=config.get("n_siren_layers", 4),
        omega_0=config.get("omega_0", 30.0),
    )
    best_path = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(best_path):
        best_path = os.path.join(run_dir, "final_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    return model


def get_run_dirs() -> list:
    """Find all training runs."""
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(runs_dir):
        return []
    dirs = sorted(glob.glob(os.path.join(runs_dir, "*")))
    return [d for d in dirs if os.path.isdir(d) and
            os.path.exists(os.path.join(d, "history.json"))]


def extract_mesh_from_sdf(sdf_grid: np.ndarray, bounds: float = 1.1):
    """Extract mesh via marching cubes from SDF grid."""
    from skimage.measure import marching_cubes
    try:
        verts, faces, normals, values = marching_cubes(sdf_grid, level=0.0)
        # Scale from grid coordinates to world coordinates
        res = sdf_grid.shape[0]
        verts = verts / (res - 1) * 2 * bounds - bounds
        return verts, faces
    except Exception:
        return None, None


def create_mesh_figure(vertices, faces, name="Mesh", color="#4C72B0",
                       opacity=0.7):
    """Create Plotly mesh3d figure."""
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        flatshading=True,
        lighting=dict(ambient=0.4, diffuse=0.6, specular=0.3, roughness=0.5),
        lightposition=dict(x=100, y=200, z=300),
    )


def create_contacts_scatter(contacts: np.ndarray, finger_labels=None):
    """Create scatter3d for contact points with finger colors."""
    if finger_labels is None:
        finger_labels = ["thumb", "index", "middle", "ring", "pinky", "palm"]

    traces = []
    for i, finger in enumerate(finger_labels[:len(contacts)]):
        color = FINGER_COLORS.get(finger, "#ffffff")
        traces.append(go.Scatter3d(
            x=[contacts[i, 0]], y=[contacts[i, 1]], z=[contacts[i, 2]],
            mode='markers',
            marker=dict(size=8, color=color, symbol='circle',
                        line=dict(width=2, color='white')),
            name=f"📍 {finger}",
            showlegend=True,
        ))

        # Normal arrow
        if contacts.shape[1] >= 6:
            normal = contacts[i, 3:6] * 0.15
            traces.append(go.Scatter3d(
                x=[contacts[i, 0], contacts[i, 0] + normal[0]],
                y=[contacts[i, 1], contacts[i, 1] + normal[1]],
                z=[contacts[i, 2], contacts[i, 2] + normal[2]],
                mode='lines',
                line=dict(width=4, color=color),
                showlegend=False,
            ))

    return traces


# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────

st.sidebar.title("🤖 Tactile SDF")
st.sidebar.markdown("**PointNet + SIREN**")
st.sidebar.markdown("---")

run_dirs = get_run_dirs()
if not run_dirs:
    st.error("❌ No training runs found in `runs/`. Run `python train.py` first!")
    st.stop()

run_names = [os.path.basename(d) for d in run_dirs]
selected_run_idx = st.sidebar.selectbox(
    "Training Run", range(len(run_names)),
    format_func=lambda i: f"🔬 {run_names[i]}"
)
selected_run = run_dirs[selected_run_idx]

config = load_config(selected_run)
history = load_history(selected_run)

if history is None:
    st.error("Could not load training history.")
    st.stop()

# Sidebar metrics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Summary")
best_iou = max(history.get("val_iou", [0]))
final_loss = history["train_loss"][-1] if history["train_loss"] else 0
n_epochs = len(history["train_loss"])

st.sidebar.metric("Best IoU", f"{best_iou:.3f}")
st.sidebar.metric("Final Train Loss", f"{final_loss:.4f}")
st.sidebar.metric("Epochs", n_epochs)
st.sidebar.metric("Parameters",
                   f"{config.get('latent_dim', 256)*4 + 256*256*4:.0f}")


# ─────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────

st.title("🧠 Tactile Shape Reconstruction")
st.markdown("### PointNet Encoder + SIREN Decoder — from sparse grasp contacts to 3D shape")

tab1, tab2, tab3 = st.tabs([
    "📈 Training Dynamics",
    "🔮 3D Reconstruction",
    "📊 Comparison Grid",
])


# ─────────────────────────────────────────────────────────
# TAB 1: Training Dynamics
# ─────────────────────────────────────────────────────────
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{best_iou:.3f}</h2>
            <p>Best IoU</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h2>{final_loss:.4f}</h2>
            <p>Final Train Loss</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h2>{n_epochs}</h2>
            <p>Epochs</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        lr_final = history["lr"][-1] if history["lr"] else 0
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h2>{lr_final:.1e}</h2>
            <p>Final LR</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Animated epoch slider
    epoch_range = st.slider(
        "Epoch range", 1, n_epochs, (1, n_epochs), key="epoch_slider"
    )
    e_start, e_end = epoch_range
    epochs_slice = list(range(e_start, e_end + 1))
    idx_start, idx_end = e_start - 1, e_end

    # Loss curves
    col_left, col_right = st.columns(2)

    with col_left:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=epochs_slice,
            y=history["train_loss"][idx_start:idx_end],
            name="Train Loss", line=dict(color="#4C72B0", width=2.5),
            fill='tozeroy', fillcolor='rgba(76,114,176,0.1)',
        ))
        fig.add_trace(go.Scatter(
            x=epochs_slice,
            y=history["val_loss"][idx_start:idx_end],
            name="Val Loss", line=dict(color="#DD8452", width=2.5),
        ))
        fig.update_layout(
            title="📉 Training & Validation Loss",
            xaxis_title="Epoch", yaxis_title="L1 Loss",
            yaxis_type="log", template="plotly_dark",
            height=400, margin=dict(t=50, b=40),
            legend=dict(x=0.7, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs_slice,
            y=history["val_iou"][idx_start:idx_end],
            name="Val IoU", line=dict(color="#8172B2", width=3),
            fill='tozeroy', fillcolor='rgba(129,114,178,0.15)',
        ))
        fig.update_layout(
            title="📐 Volumetric IoU",
            xaxis_title="Epoch", yaxis_title="IoU",
            yaxis_range=[0, 1], template="plotly_dark",
            height=400, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Component losses
    st.markdown("#### Loss Components")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs_slice,
            y=history["sdf_loss"][idx_start:idx_end],
            name="SDF L1", line=dict(color="#4C72B0", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=epochs_slice,
            y=history["contact_loss"][idx_start:idx_end],
            name="Contact", line=dict(color="#55A868", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=epochs_slice,
            y=history["eikonal_loss"][idx_start:idx_end],
            name="Eikonal", line=dict(color="#C44E52", width=2),
        ))
        fig.update_layout(
            title="Loss Components (Train)",
            xaxis_title="Epoch", yaxis_title="Loss",
            yaxis_type="log", template="plotly_dark",
            height=350, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        # Per-category IoU evolution
        if history["per_category_iou"]:
            fig = go.Figure()
            cat_colors = {"bottle": "#4C72B0", "cup": "#DD8452",
                          "hammer": "#55A868", "screwdriver": "#C44E52",
                          "wrench": "#8172B2"}
            all_cats = set()
            for entry in history["per_category_iou"]:
                all_cats.update(entry.keys())
            for cat in sorted(all_cats):
                ious = [e.get(cat, 0) for e in
                        history["per_category_iou"][idx_start:idx_end]]
                fig.add_trace(go.Scatter(
                    x=epochs_slice[:len(ious)], y=ious,
                    name=cat.capitalize(),
                    line=dict(color=cat_colors.get(cat, "#999"), width=2.5),
                ))
            fig.update_layout(
                title="Per-Category IoU Evolution",
                xaxis_title="Epoch", yaxis_title="IoU",
                yaxis_range=[0, 1], template="plotly_dark",
                height=350, margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    # LR schedule
    with st.expander("📐 Learning Rate Schedule"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history["lr"]) + 1)),
            y=history["lr"],
            line=dict(color="#f39c12", width=2),
        ))
        fig.update_layout(
            title="Cosine Annealing LR",
            xaxis_title="Epoch", yaxis_title="LR",
            yaxis_type="log", template="plotly_dark",
            height=250,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# TAB 2: 3D Reconstruction
# ─────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Select an object to visualize reconstruction")

    # Load test dataset
    @st.cache_resource
    def get_test_dataset():
        try:
            ds = TactileSDFDataset(
                split="test",
                n_query_points=2048,
                dataset_dir=config.get("dataset_dir",
                                        "../grasp-dataset-gen/output_hf"),
                sdf_cache_dir=config.get("sdf_cache_dir", "data/sdf_cache"),
            )
            return ds
        except Exception as e:
            st.error(f"Could not load test dataset: {e}")
            return None

    test_ds = get_test_dataset()

    if test_ds is not None and len(test_ds) > 0:
        # Object selector
        sample_names = [(s[0], s[1]) for s in test_ds.samples]
        unique_objects = sorted(set(s[0] for s in sample_names))

        col_obj, col_strat, col_res = st.columns(3)
        with col_obj:
            selected_obj = st.selectbox("Object", unique_objects)
        with col_strat:
            obj_strategies = [s[1] for s in sample_names if s[0] == selected_obj]
            selected_strat = st.selectbox("Strategy", obj_strategies)
        with col_res:
            grid_res = st.slider("Grid resolution", 32, 128, 64, step=16)

        # Find sample index
        sample_idx = None
        for i, (name, strat) in enumerate(sample_names):
            if name == selected_obj and strat == selected_strat:
                sample_idx = i
                break

        if sample_idx is not None:
            sample = test_ds[sample_idx]

            # Load model and predict
            model = load_model(selected_run, config)
            contacts = sample["contacts"].unsqueeze(0)  # (1, 6, 9)

            with st.spinner(f"🔮 Computing SDF grid ({grid_res}³)..."):
                sdf_grid = model.predict_grid(contacts, resolution=grid_res)

            # Extract predicted mesh
            pred_verts, pred_faces = extract_mesh_from_sdf(sdf_grid)

            # Load GT mesh
            sdf_data = test_ds._load_sdf(selected_obj)
            gt_verts = sdf_data["vertices"]
            gt_faces = sdf_data["faces"]

            # Contact points
            contact_np = sample["contacts"].numpy()

            # Create side-by-side 3D views
            col_gt, col_pred = st.columns(2)

            with col_gt:
                st.markdown("##### 🎯 Ground Truth")
                fig_gt = go.Figure()
                fig_gt.add_trace(create_mesh_figure(
                    gt_verts, gt_faces, name="GT Mesh", color="#4C72B0"
                ))
                for trace in create_contacts_scatter(contact_np):
                    fig_gt.add_trace(trace)
                fig_gt.update_layout(
                    scene=dict(
                        xaxis=dict(range=[-1.2, 1.2], showbackground=False),
                        yaxis=dict(range=[-1.2, 1.2], showbackground=False),
                        zaxis=dict(range=[-1.2, 1.2], showbackground=False),
                        aspectmode='cube',
                        bgcolor='#1a1a2e',
                    ),
                    template="plotly_dark",
                    height=550,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(x=0.02, y=0.98),
                )
                st.plotly_chart(fig_gt, use_container_width=True)

            with col_pred:
                st.markdown("##### 🔮 Predicted (SIREN)")
                fig_pred = go.Figure()
                if pred_verts is not None:
                    fig_pred.add_trace(create_mesh_figure(
                        pred_verts, pred_faces, name="Predicted",
                        color="#e74c3c", opacity=0.65,
                    ))
                else:
                    st.warning("⚠️ Could not extract isosurface (marching cubes failed)")
                for trace in create_contacts_scatter(contact_np):
                    fig_pred.add_trace(trace)
                fig_pred.update_layout(
                    scene=dict(
                        xaxis=dict(range=[-1.2, 1.2], showbackground=False),
                        yaxis=dict(range=[-1.2, 1.2], showbackground=False),
                        zaxis=dict(range=[-1.2, 1.2], showbackground=False),
                        aspectmode='cube',
                        bgcolor='#1a1a2e',
                    ),
                    template="plotly_dark",
                    height=550,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(x=0.02, y=0.98),
                )
                st.plotly_chart(fig_pred, use_container_width=True)

            # SDF cross-section
            with st.expander("🔬 SDF Cross-Section (Z-slice)"):
                z_idx = st.slider("Z slice", 0, grid_res - 1, grid_res // 2)
                fig_slice = px.imshow(
                    sdf_grid[:, :, z_idx].T,
                    color_continuous_scale="RdBu_r",
                    origin="lower",
                    zmin=-0.5, zmax=0.5,
                    labels=dict(color="SDF"),
                )
                fig_slice.update_layout(
                    title=f"SDF slice at z={z_idx}",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig_slice, use_container_width=True)
    else:
        st.info("No test data available. Run preprocessing and training first.")


# ─────────────────────────────────────────────────────────
# TAB 3: Comparison Grid
# ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### All Test Objects — GT vs Predicted")

    if test_ds is not None and len(test_ds) > 0:
        model = load_model(selected_run, config)
        compare_res = st.slider("Grid resolution (comparison)", 32, 64, 48,
                                step=8, key="compare_res")

        unique_test = sorted(set(s[0] for s in test_ds.samples))

        # Per-category organization
        for cat in CATEGORIES:
            cat_objs = [o for o in unique_test if o.startswith(cat)]
            if not cat_objs:
                continue

            st.markdown(f"### {cat.capitalize()}")
            cols = st.columns(min(len(cat_objs), 3))

            for i, obj_name in enumerate(cat_objs):
                with cols[i % len(cols)]:
                    # Find first strategy for this object
                    for si, (sn, ss) in enumerate(test_ds.samples):
                        if sn == obj_name:
                            sample = test_ds[si]
                            break

                    contacts = sample["contacts"].unsqueeze(0)
                    with st.spinner(f"Computing {obj_name}..."):
                        sdf_grid = model.predict_grid(
                            contacts, resolution=compare_res
                        )

                    pred_verts, pred_faces = extract_mesh_from_sdf(sdf_grid)
                    sdf_data = test_ds._load_sdf(obj_name)
                    gt_verts = sdf_data["vertices"]
                    gt_faces = sdf_data["faces"]

                    # Combined figure
                    fig = go.Figure()
                    fig.add_trace(create_mesh_figure(
                        gt_verts, gt_faces, "GT", "#4C72B0", 0.4
                    ))
                    if pred_verts is not None:
                        fig.add_trace(create_mesh_figure(
                            pred_verts, pred_faces, "Pred", "#e74c3c", 0.5
                        ))
                    for trace in create_contacts_scatter(
                            sample["contacts"].numpy()):
                        fig.add_trace(trace)

                    fig.update_layout(
                        title=dict(text=obj_name, font=dict(size=12)),
                        scene=dict(
                            xaxis=dict(range=[-1.2, 1.2], showbackground=False,
                                       showticklabels=False),
                            yaxis=dict(range=[-1.2, 1.2], showbackground=False,
                                       showticklabels=False),
                            zaxis=dict(range=[-1.2, 1.2], showbackground=False,
                                       showticklabels=False),
                            aspectmode='cube',
                            bgcolor='#1a1a2e',
                        ),
                        template="plotly_dark",
                        height=350,
                        margin=dict(l=0, r=0, t=40, b=0),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
    else:
        st.info("No test data available.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Made with ❤️ for ISIR\n\n"
    "**Architecture**: PointNet + SIREN\n\n"
    "**Dataset**: grasp-dataset-gen (50 Objaverse objects)"
)
