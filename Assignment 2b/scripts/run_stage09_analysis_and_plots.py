"""Stage 09: Unified Comparative Analysis and Research-Quality Visualisations.

Strictly read-only over all 27 frozen experiments (`results/speech/`,
`results/text/`, `results/fusion/`). Generates:
1. Unified comparison tables (CSV, Markdown, JSON) under `results/fusion/summaries/`.
2. Required 3-panel t-SNE visualisation and all diagnostic/comparative plots
   (300 DPI) under `results/plots/`.
3. Automated validation of every generated table, plot, and frozen checkpoint hash.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, log_loss

from models.fusion_pipeline.model import ID_TO_EMOTION, NUM_EMOTION_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "Results"
SUMMARIES_DIR = RESULTS_ROOT / "fusion" / "summaries"
PLOTS_DIR = RESULTS_ROOT / "plots"

EMOTION_NAMES: List[str] = [ID_TO_EMOTION[i] for i in range(NUM_EMOTION_CLASSES)]
EMOTION_LABELS_PRETTY: Dict[str, str] = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "pleasant_surprise": "Pleasant Surprise",
    "sad": "Sad",
}
# Colorblind-friendly Okabe-Ito-inspired palette for 7 emotions
EMOTION_COLORS: Dict[str, str] = {
    "angry": "#D55E00",
    "disgust": "#009E73",
    "fear": "#CC79A7",
    "happy": "#E69F00",
    "neutral": "#7F7F7F",
    "pleasant_surprise": "#56B4E9",
    "sad": "#0072B2",
}

SEEDS: Tuple[int, ...] = (42, 123, 999)

# Verified actual trainable parameter counts (Stage 04, 07A, 08A)
ARCH_TRAINABLE_PARAMS: Dict[str, int] = {
    "wavlm_bigru_attn": 857_863,
    "mfcc_cnn_bilstm": 1_076_279,
    "distilbert_bilstm_attn": 1_055_239,
    "gated_fusion": 296_967,
    "concat_fusion": 165_895,
    "late_fusion": 0,
}

ARCH_DISPLAY_NAMES: Dict[str, str] = {
    "wavlm_bigru_attn": "WavLM + BiGRU + Attn (Speech)",
    "mfcc_cnn_bilstm": "MFCC + CNN-BiLSTM (Speech)",
    "distilbert_bilstm_attn": "DistilBERT + BiLSTM + Attn (Text)",
    "gated_fusion": "Gated Multimodal Fusion",
    "concat_fusion": "Concatenation Fusion",
    "late_fusion": "Late-Logit Fusion",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_frozen_hashes() -> Dict[str, str]:
    """Snapshot SHA-256 of all 24 best_model.pt + 3 late_fusion selected_lambda.json."""
    targets = sorted(RESULTS_ROOT.rglob("best_model.pt")) + sorted(
        (RESULTS_ROOT / "fusion" / "late_fusion").rglob("selected_lambda.json")
    )
    return {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in targets}


def _compute_weighted_f1_and_log_loss(pred_csv: Path) -> Tuple[float, float]:
    df = pd.read_csv(pred_csv)
    y_true = df["true_label_id"].to_numpy(dtype=np.int64)
    y_pred = df["pred_label_id"].to_numpy(dtype=np.int64)
    prob_cols = [f"prob_{e}" for e in EMOTION_NAMES]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    wf1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    ll = float(log_loss(y_true, probs, labels=list(range(NUM_EMOTION_CLASSES))))
    return round(wf1, 6), round(ll, 6)


def collect_all_27_runs() -> List[Dict[str, Any]]:
    """Read exact saved metrics from all 27 completed experiment directories."""
    runs: List[Dict[str, Any]] = []

    # 1. Speech (12 runs)
    for split_col in ("word_indep_split", "cross_speaker_split"):
        for arch in ("wavlm_bigru_attn", "mfcc_cnn_bilstm"):
            for seed in SEEDS:
                d = RESULTS_ROOT / "speech" / split_col / arch / f"seed_{seed}"
                m = json.load(open(d / "test_metrics.json", "r", encoding="utf-8"))
                wf1, ll = _compute_weighted_f1_and_log_loss(d / "test_predictions.csv")
                runs.append(
                    {
                        "modality": "speech",
                        "architecture": arch,
                        "display_name": ARCH_DISPLAY_NAMES[arch],
                        "split_type": split_col,
                        "seed": seed,
                        "num_test_samples": int(m["num_test_samples"]),
                        "accuracy": float(m["accuracy"]),
                        "macro_precision": float(m["macro_precision"]),
                        "macro_recall": float(m["macro_recall"]),
                        "macro_f1": float(m["macro_f1"]),
                        "weighted_f1": wf1,
                        "log_loss": ll,
                        "trainable_params": ARCH_TRAINABLE_PARAMS[arch],
                        "per_class_f1": m["per_class_f1"],
                        "metrics_source": str((d / "test_metrics.json").relative_to(PROJECT_ROOT)),
                    }
                )

    # 2. Text (6 runs)
    for split_col, text_dir in (
        ("word_indep_split", "word_independent"),
        ("cross_speaker_split", "cross_speaker"),
    ):
        arch = "distilbert_bilstm_attn"
        for seed in SEEDS:
            d = RESULTS_ROOT / "text" / arch / text_dir / f"seed_{seed}"
            m = json.load(open(d / "test_metrics.json", "r", encoding="utf-8"))
            wf1, ll = _compute_weighted_f1_and_log_loss(d / "test_predictions.csv")
            runs.append(
                {
                    "modality": "text",
                    "architecture": arch,
                    "display_name": ARCH_DISPLAY_NAMES[arch],
                    "split_type": split_col,
                    "seed": seed,
                    "num_test_samples": int(m["num_test_samples"]),
                    "accuracy": float(m["accuracy"]),
                    "macro_precision": float(m["macro_precision"]),
                    "macro_recall": float(m["macro_recall"]),
                    "macro_f1": float(m["macro_f1"]),
                    "weighted_f1": wf1,
                    "log_loss": ll,
                    "trainable_params": ARCH_TRAINABLE_PARAMS[arch],
                    "per_class_f1": m["per_class_f1"],
                    "metrics_source": str((d / "test_metrics.json").relative_to(PROJECT_ROOT)),
                }
            )

    # 3. Fusion (9 completed word_indep_split runs)
    for arch in ("gated_fusion", "concat_fusion", "late_fusion"):
        split_col = "word_indep_split"
        for seed in SEEDS:
            d = RESULTS_ROOT / "fusion" / arch / split_col / f"seed_{seed}"
            m = json.load(open(d / "metrics.json", "r", encoding="utf-8"))
            entry: Dict[str, Any] = {
                "modality": "fusion",
                "architecture": arch,
                "display_name": ARCH_DISPLAY_NAMES[arch],
                "split_type": split_col,
                "seed": seed,
                "num_test_samples": int(m["num_test_samples"]),
                "accuracy": float(m["accuracy"]),
                "macro_precision": float(m["macro_precision"]),
                "macro_recall": float(m["macro_recall"]),
                "macro_f1": float(m["macro_f1"]),
                "weighted_f1": float(m["weighted_f1"]),
                "log_loss": float(m["log_loss"]),
                "trainable_params": ARCH_TRAINABLE_PARAMS[arch],
                "per_class_f1": m["per_class_f1"],
                "metrics_source": str((d / "metrics.json").relative_to(PROJECT_ROOT)),
            }
            if arch == "late_fusion":
                entry["selected_lambda"] = float(m["selected_lambda"])
            if arch == "gated_fusion" and "gate_stats" in m:
                entry["gate_mean"] = float(m["gate_stats"]["mean_gate"])
                entry["gate_std"] = float(m["gate_stats"]["std_gate"])
            runs.append(entry)

    assert len(runs) == 27, f"Expected 27 completed runs, found {len(runs)}"
    return runs


def aggregate_group(group_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and sample std (ddof=1) across seeds for a group of runs."""
    ddof = 1 if len(group_runs) > 1 else 0
    first = group_runs[0]
    out: Dict[str, Any] = {
        "split_type": first["split_type"],
        "modality": first["modality"],
        "architecture": first["architecture"],
        "display_name": first["display_name"],
        "num_seeds": len(group_runs),
        "seeds": ",".join(str(r["seed"]) for r in group_runs),
        "trainable_params": first["trainable_params"],
    }
    for metric in (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "log_loss",
    ):
        vals = np.array([r[metric] for r in group_runs], dtype=np.float64)
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=ddof))
        out[f"{metric}_mean"] = round(mean_v, 6)
        out[f"{metric}_std"] = round(std_v, 6)
        if metric == "log_loss":
            out[f"{metric}_mean_std"] = f"{mean_v:.4f} ± {std_v:.4f}"
        else:
            out[f"{metric}_mean_std"] = f"{mean_v * 100:.2f} ± {std_v * 100:.2f}%"

    if "selected_lambda" in first:
        lams = [r["selected_lambda"] for r in group_runs]
        out["selected_lambda_values"] = ",".join(f"{x:.2f}" for x in lams)
    if "gate_mean" in first:
        g_means = np.array([r["gate_mean"] for r in group_runs], dtype=np.float64)
        out["gate_mean_across_seeds"] = round(float(np.mean(g_means)), 6)
        out["gate_std_across_seeds"] = round(float(np.std(g_means, ddof=ddof)), 6)

    for emo in EMOTION_NAMES:
        vals = np.array([r["per_class_f1"][emo] for r in group_runs], dtype=np.float64)
        out[f"f1_{emo}_mean"] = round(float(np.mean(vals)), 6)
        out[f"f1_{emo}_std"] = round(float(np.std(vals, ddof=ddof)), 6)

    return out


def build_comparison_tables(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate Tables 1–5 in CSV, Markdown, and machine-readable JSON."""
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    ordered_groups = [
        ("word_indep_split", "wavlm_bigru_attn"),
        ("word_indep_split", "mfcc_cnn_bilstm"),
        ("word_indep_split", "distilbert_bilstm_attn"),
        ("word_indep_split", "gated_fusion"),
        ("word_indep_split", "concat_fusion"),
        ("word_indep_split", "late_fusion"),
        ("cross_speaker_split", "wavlm_bigru_attn"),
        ("cross_speaker_split", "mfcc_cnn_bilstm"),
        ("cross_speaker_split", "distilbert_bilstm_attn"),
    ]

    overall_rows: List[Dict[str, Any]] = []
    for split_col, arch in ordered_groups:
        sub = [r for r in runs if r["split_type"] == split_col and r["architecture"] == arch]
        assert len(sub) == 3, f"Expected 3 seeds for {split_col}/{arch}, got {len(sub)}"
        overall_rows.append(aggregate_group(sub))

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(SUMMARIES_DIR / "overall_comparison.csv", index=False)

    # Table 2: Speech comparison
    speech_df = overall_df[overall_df["modality"] == "speech"].reset_index(drop=True)
    speech_df.to_csv(SUMMARIES_DIR / "speech_comparison.csv", index=False)

    # Table 3: Text comparison (per-seed + summary rows)
    text_runs_df = pd.DataFrame(
        [
            {
                "split_type": r["split_type"],
                "architecture": r["architecture"],
                "seed": r["seed"],
                "num_test_samples": r["num_test_samples"],
                "accuracy": r["accuracy"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "macro_f1": r["macro_f1"],
                "weighted_f1": r["weighted_f1"],
                "log_loss": r["log_loss"],
                "trainable_params": r["trainable_params"],
            }
            for r in runs
            if r["modality"] == "text"
        ]
    )
    text_runs_df.to_csv(SUMMARIES_DIR / "text_comparison.csv", index=False)

    # Table 4: Multimodal fusion comparison
    fusion_df = overall_df[overall_df["modality"] == "fusion"].reset_index(drop=True)
    fusion_df.to_csv(SUMMARIES_DIR / "fusion_comparison.csv", index=False)

    # Table 5: Cross-speaker comparison (existing speech & text runs only)
    cs_df = overall_df[overall_df["split_type"] == "cross_speaker_split"].reset_index(drop=True)
    cs_df.to_csv(SUMMARIES_DIR / "cross_speaker_comparison.csv", index=False)

    # Markdown report for Table 1–5
    md_lines = [
        "# Assignment 2B — Unified Comparative Tables (27 Frozen Experiments)",
        "",
        "All metrics are computed directly from saved experiment artifacts across 3 random seeds (`42`, `123`, `999`) with sample standard deviation (`ddof=1`). Weighted-F1 and Log-Loss for speech and text runs are computed directly from saved `test_predictions.csv`.",
        "",
        "## Table 1 — Overall Modality Comparison (Mean ± Std across 3 Seeds)",
        "",
        "| Split | Architecture | Trainable Params | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Log-Loss |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in overall_rows:
        md_lines.append(
            f"| `{r['split_type']}` | {r['display_name']} | `{r['trainable_params']:,}` | "
            f"{r['accuracy_mean_std']} | {r['macro_precision_mean_std']} | "
            f"{r['macro_recall_mean_std']} | {r['macro_f1_mean_std']} | "
            f"{r['weighted_f1_mean_std']} | {r['log_loss_mean_std']} |"
        )

    md_lines += [
        "",
        "## Table 2 — Speech Architecture Comparison (WavLM vs. MFCC)",
        "",
        "| Split | Speech Architecture | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Log-Loss |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in speech_df.iterrows():
        md_lines.append(
            f"| `{r['split_type']}` | {r['display_name']} | {r['accuracy_mean_std']} | "
            f"{r['macro_precision_mean_std']} | {r['macro_recall_mean_std']} | "
            f"{r['macro_f1_mean_std']} | {r['weighted_f1_mean_std']} | {r['log_loss_mean_std']} |"
        )

    md_lines += [
        "",
        "## Table 3 — Text-Only Results (`DistilBERT + BiLSTM + Attention`)",
        "",
        "Every one of the 200 TESS target words is spoken identically across all 7 emotions inside the carrier phrase `\"Say the word <word>\"`. Consequently, lexical transcripts carry zero mutual information with acted emotion ($I(\\text{Text}; \\text{Emotion}) = 0$), yielding exact $1/7 \\approx 14.29\\%$ chance accuracy and near-uniform predicted probabilities (~`0.1429` per class, log-loss $\\approx \\ln(7) \\approx 1.9459$) across every seed and split.",
        "",
        "| Split | Seed | Test Samples | Accuracy | Macro-F1 | Weighted-F1 | Log-Loss |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in text_runs_df.iterrows():
        md_lines.append(
            f"| `{r['split_type']}` | `{int(r['seed'])}` | `{int(r['num_test_samples'])}` | "
            f"`{r['accuracy']*100:.2f}%` | `{r['macro_f1']*100:.2f}%` | "
            f"`{r['weighted_f1']*100:.2f}%` | `{r['log_loss']:.4f}` |"
        )

    md_lines += [
        "",
        "## Table 4 — Multimodal Fusion Comparison (`word_indep_split`, 3 Seeds)",
        "",
        "> *Note:* Gate activation statistics for `GatedMultimodalFusion` describe empirical layer outputs on test embeddings and are not interpreted as definitive causal modality importance.",
        "",
        "| Fusion Architecture | Trainable Params | Accuracy | Macro-F1 | Weighted-F1 | Log-Loss | Extra Calibration / Gate Stats |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for _, r in fusion_df.iterrows():
        extra = "—"
        if r["architecture"] == "late_fusion":
            extra = f"Selected $\\lambda$: `{r['selected_lambda_values']}`"
        elif r["architecture"] == "gated_fusion":
            extra = f"Mean gate across seeds: `{r['gate_mean_across_seeds']:.4f} ± {r['gate_std_across_seeds']:.4f}`"
        md_lines.append(
            f"| {r['display_name']} | `{int(r['trainable_params']):,}` | {r['accuracy_mean_std']} | "
            f"{r['macro_f1_mean_std']} | {r['weighted_f1_mean_std']} | {r['log_loss_mean_std']} | {extra} |"
        )

    md_lines += [
        "",
        "## Table 5 — Cross-Speaker Comparison (`OAF -> YAF`, Existing Unimodal Runs Only)",
        "",
        "*Cross-speaker multimodal fusion experiments were not executed (optional) and are omitted rather than fabricated.*",
        "",
        "| Architecture | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Log-Loss |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in cs_df.iterrows():
        md_lines.append(
            f"| {r['display_name']} | {r['accuracy_mean_std']} | {r['macro_precision_mean_std']} | "
            f"{r['macro_recall_mean_std']} | {r['macro_f1_mean_std']} | "
            f"{r['weighted_f1_mean_std']} | {r['log_loss_mean_std']} |"
        )

    (SUMMARIES_DIR / "overall_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    unified_json = {
        "total_completed_runs": len(runs),
        "seeds": list(SEEDS),
        "overall_comparison": overall_rows,
        "all_27_runs": runs,
    }
    with open(SUMMARIES_DIR / "stage09_unified_summary.json", "w", encoding="utf-8") as f:
        json.dump(unified_json, f, indent=2)

    return unified_json


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_tsne_three_panel() -> Dict[str, Any]:
    """Generate mandatory 3-panel t-SNE plot: Temporal vs. Contextual vs. Fusion."""
    sp_path = RESULTS_ROOT / "speech/word_indep_split/wavlm_bigru_attn/seed_42/test_embeddings.pt"
    tx_path = RESULTS_ROOT / "text/distilbert_bilstm_attn/word_independent/seed_42/test_embeddings.pt"
    fu_path = RESULTS_ROOT / "fusion/gated_fusion/word_indep_split/seed_42/test_embeddings.npz"

    sp_pt = torch.load(sp_path, map_location="cpu", weights_only=False)
    tx_pt = torch.load(tx_path, map_location="cpu", weights_only=False)
    fu_npz = np.load(fu_path)

    assert sp_pt.get("block") == "temporal_modelling_block"
    assert tx_pt.get("block") == "contextual_modelling_block"

    sp_ids = list(sp_pt["sample_ids"])
    tx_ids = list(tx_pt["sample_ids"])
    fu_ids = fu_npz["sample_ids"].tolist()
    assert sp_ids == tx_ids == fu_ids and len(sp_ids) == 420

    sp_y = sp_pt["labels"].numpy()
    tx_y = tx_pt["labels"].numpy()
    fu_y = fu_npz["labels"]
    assert np.array_equal(sp_y, tx_y) and np.array_equal(sp_y, fu_y)

    panels = [
        ("Temporal Modelling Block\n(WavLM + BiGRU + Attention)", sp_pt["embeddings"].numpy()),
        ("Contextual Modelling Block\n(DistilBERT + BiLSTM + Attention)", tx_pt["embeddings"].numpy()),
        ("Fusion Block\n(Gated Multimodal Fusion)", fu_npz["embeddings"]),
    ]

    tsne_params = {
        "n_components": 2,
        "perplexity": 30,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": 42,
        "max_iter": 1000,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    handles, labels = [], []

    for ax, (title, emb) in zip(axes, panels):
        assert emb.shape == (420, 256) and np.isfinite(emb).all()
        proj = TSNE(**tsne_params).fit_transform(emb)
        for cls_id, emo in enumerate(EMOTION_NAMES):
            mask = sp_y == cls_id
            sc = ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                c=EMOTION_COLORS[emo],
                label=EMOTION_LABELS_PRETTY[emo],
                s=22,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.3,
            )
            if ax is axes[0]:
                handles.append(sc)
                labels.append(EMOTION_LABELS_PRETTY[emo])
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.grid(alpha=0.2, linestyle="--")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        markerscale=1.4,
    )
    fig.suptitle(
        "Emotion Cluster Separability Across Learned Representations (Word-Independent Test Set, N = 420, Seed 42)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )

    out_path = PLOTS_DIR / "tsne_three_panel_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    config_info = {
        "output_plot": str(out_path.relative_to(PROJECT_ROOT)),
        "num_samples": 420,
        "split_type": "word_indep_split",
        "seed": 42,
        "sample_ids_aligned": True,
        "labels_aligned": True,
        "tsne_parameters": tsne_params,
        "sources": {
            "temporal": str(sp_path.relative_to(PROJECT_ROOT)),
            "contextual": str(tx_path.relative_to(PROJECT_ROOT)),
            "fusion": str(fu_path.relative_to(PROJECT_ROOT)),
        },
    }
    with open(PLOTS_DIR / "tsne_config_and_sources.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)
    return config_info


def plot_confusion_matrices() -> None:
    """Plot confusion matrices for Word-Independent (6 models) and Cross-Speaker (3 models)."""
    short_labels = ["Ang", "Dis", "Fea", "Hap", "Neu", "PS", "Sad"]

    # 1. Word-Independent (Seed 42)
    wi_specs = [
        ("WavLM Speech-Only (99.76%)", RESULTS_ROOT / "speech/word_indep_split/wavlm_bigru_attn/seed_42/confusion_matrix.csv"),
        ("MFCC Speech-Only (99.76%)", RESULTS_ROOT / "speech/word_indep_split/mfcc_cnn_bilstm/seed_42/confusion_matrix.csv"),
        ("DistilBERT Text-Only (14.29%)", RESULTS_ROOT / "text/distilbert_bilstm_attn/word_independent/seed_42/confusion_matrix.csv"),
        ("Gated Fusion (99.76%)", RESULTS_ROOT / "fusion/gated_fusion/word_indep_split/seed_42/confusion_matrix.csv"),
        ("Concatenation Fusion (99.76%)", RESULTS_ROOT / "fusion/concat_fusion/word_indep_split/seed_42/confusion_matrix.csv"),
        ("Late-Logit Fusion (99.76%)", RESULTS_ROOT / "fusion/late_fusion/word_indep_split/seed_42/confusion_matrix.csv"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), constrained_layout=True)
    for ax, (title, csv_path) in zip(axes.flat, wi_specs):
        cm = pd.read_csv(csv_path, index_col=0).to_numpy()
        assert cm.shape == (7, 7) and cm.sum() == 420
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=short_labels, yticklabels=short_labels, ax=ax,
            linewidths=0.4, linecolor="#E0E0E0",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(
        "Confusion Matrices — Word-Independent Split (Representative Seed 42, N = 420)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.savefig(PLOTS_DIR / "confusion_matrices_word_independent.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Cross-Speaker (Seed 42)
    cs_specs = [
        ("WavLM Speech-Only (Seed 42: 31.00%)", RESULTS_ROOT / "speech/cross_speaker_split/wavlm_bigru_attn/seed_42/confusion_matrix.csv"),
        ("MFCC Speech-Only (Seed 42: 41.79%)", RESULTS_ROOT / "speech/cross_speaker_split/mfcc_cnn_bilstm/seed_42/confusion_matrix.csv"),
        ("DistilBERT Text-Only (Seed 42: 14.29%)", RESULTS_ROOT / "text/distilbert_bilstm_attn/cross_speaker/seed_42/confusion_matrix.csv"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (title, csv_path) in zip(axes.flat, cs_specs):
        cm = pd.read_csv(csv_path, index_col=0).to_numpy()
        assert cm.shape == (7, 7) and cm.sum() == 1400
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Oranges", cbar=False,
            xticklabels=short_labels, yticklabels=short_labels, ax=ax,
            linewidths=0.4, linecolor="#E0E0E0",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle(
        "Confusion Matrices — Cross-Speaker Split (OAF -> YAF, Representative Seed 42, N = 1,400)",
        fontsize=12.5, fontweight="bold", y=1.04,
    )
    fig.savefig(PLOTS_DIR / "confusion_matrices_cross_speaker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves() -> None:
    """Plot Train Loss, Val Loss, and Val Macro-F1 curves for trainable models (Seed 42)."""
    specs = [
        ("WavLM Speech", RESULTS_ROOT / "speech/word_indep_split/wavlm_bigru_attn/seed_42/training_history.csv", "#0072B2"),
        ("MFCC Speech", RESULTS_ROOT / "speech/word_indep_split/mfcc_cnn_bilstm/seed_42/training_history.csv", "#56B4E9"),
        ("DistilBERT Text", RESULTS_ROOT / "text/distilbert_bilstm_attn/word_independent/seed_42/training_history.csv", "#7F7F7F"),
        ("Gated Fusion", RESULTS_ROOT / "fusion/gated_fusion/word_indep_split/seed_42/training_history.csv", "#009E73"),
        ("Concat Fusion", RESULTS_ROOT / "fusion/concat_fusion/word_indep_split/seed_42/training_history.csv", "#D55E00"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for label, csv_path, color in specs:
        df = pd.read_csv(csv_path)
        axes[0].plot(df["epoch"], df["train_loss"], label=label, color=color, linewidth=1.8)
        axes[1].plot(df["epoch"], df["val_loss"], label=label, color=color, linewidth=1.8)
        axes[2].plot(df["epoch"], df["val_macro_f1"], label=label, color=color, linewidth=1.8)

    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Validation Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[2].set_title("Validation Macro-F1", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(-0.02, 1.05)

    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
    axes[2].legend(loc="center right", frameon=True, framealpha=0.9)
    fig.suptitle(
        "Training & Validation Curves — Word-Independent Split (Seed 42; Late Fusion is Parameter-Free)",
        fontsize=12.5, fontweight="bold", y=1.03,
    )
    fig.savefig(PLOTS_DIR / "training_validation_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(overall_rows: List[Dict[str, Any]]) -> None:
    """Plot 3-seed mean per-class F1 across all 7 emotions on word_indep_split."""
    wi_rows = [r for r in overall_rows if r["split_type"] == "word_indep_split"]
    x = np.arange(len(EMOTION_NAMES))
    width = 0.13
    colors = ["#0072B2", "#56B4E9", "#7F7F7F", "#009E73", "#D55E00", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(13.5, 5.0), constrained_layout=True)
    for idx, (row, color) in enumerate(zip(wi_rows, colors)):
        means = [row[f"f1_{e}_mean"] for e in EMOTION_NAMES]
        offset = (idx - (len(wi_rows) - 1) / 2.0) * width
        ax.bar(x + offset, means, width, label=row["display_name"], color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([EMOTION_LABELS_PRETTY[e] for e in EMOTION_NAMES])
    ax.set_ylabel("3-Seed Mean F1 Score")
    ax.set_ylim(0.0, 1.08)
    ax.set_title(
        "Per-Class F1 Comparison Across All Six Architectures (Word-Independent Split, 3-Seed Mean)",
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.savefig(PLOTS_DIR / "per_class_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gated_fusion_analysis() -> None:
    """Plot descriptive gate activation distribution and per-emotion mean ± std across 3 seeds."""
    all_gates: List[np.ndarray] = []
    per_emo_seed_means: Dict[str, List[float]] = {e: [] for e in EMOTION_NAMES}

    for seed in SEEDS:
        d = RESULTS_ROOT / "fusion/gated_fusion/word_indep_split" / f"seed_{seed}"
        g = np.load(d / "gate_activations.npz")["gates"]
        assert g.shape == (420, 256) and np.isfinite(g).all()
        all_gates.append(g.ravel())
        gs = json.load(open(d / "gate_summary.json", "r", encoding="utf-8"))
        for emo in EMOTION_NAMES:
            per_emo_seed_means[emo].append(float(gs["per_emotion_mean_gate"][emo]))

    pooled_gates = np.concatenate(all_gates)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)

    # Panel 1: Histogram of gate activations
    axes[0].hist(pooled_gates, bins=40, color="#009E73", alpha=0.82, edgecolor="white", density=True)
    axes[0].axvline(pooled_gates.mean(), color="#D55E00", linestyle="--", linewidth=1.8, label=f"Mean = {pooled_gates.mean():.3f} ± {pooled_gates.std():.3f}")
    axes[0].set_title("Empirical Distribution of Gate Activations\n(3 Seeds Pooled, 420 × 256 × 3 Values)", fontweight="bold")
    axes[0].set_xlabel("Gate Activation Value g ∈ [0, 1]")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=True)
    axes[0].grid(alpha=0.25, linestyle="--")

    # Panel 2: Per-emotion mean gate ± std across seeds
    emo_means = [float(np.mean(per_emo_seed_means[e])) for e in EMOTION_NAMES]
    emo_stds = [float(np.std(per_emo_seed_means[e], ddof=1)) for e in EMOTION_NAMES]
    bar_colors = [EMOTION_COLORS[e] for e in EMOTION_NAMES]
    axes[1].bar(
        [EMOTION_LABELS_PRETTY[e] for e in EMOTION_NAMES],
        emo_means,
        yerr=emo_stds,
        capsize=4,
        color=bar_colors,
        alpha=0.88,
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Per-Emotion Mean Gate Activation\n(3-Seed Mean ± Std; Descriptive Statistics)", fontweight="bold")
    axes[1].set_ylabel("Mean Gate Value")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(
        "Gated Multimodal Fusion — Saved Gate Activation Statistics (Word-Independent Split)",
        fontsize=12.5, fontweight="bold", y=1.03,
    )
    fig.savefig(PLOTS_DIR / "gated_fusion_gate_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_late_fusion_calibration() -> None:
    """Plot Validation Macro-F1 and Validation Log-Loss vs. lambda across all 3 seeds."""
    seed_colors = {42: "#0072B2", 123: "#009E73", 999: "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), constrained_layout=True)

    for seed in SEEDS:
        d = RESULTS_ROOT / "fusion/late_fusion/word_indep_split" / f"seed_{seed}"
        df = pd.read_csv(d / "lambda_search.csv")
        sel = json.load(open(d / "selected_lambda.json", "r", encoding="utf-8"))
        sel_lam = float(sel["selected_lambda"])
        sel_row = df[np.isclose(df["lambda_weight"], sel_lam)].iloc[0]

        axes[0].plot(df["lambda_weight"], df["val_macro_f1"], marker="o", markersize=4, label=f"Seed {seed} (λ*={sel_lam:.2f})", color=seed_colors[seed], linewidth=1.6)
        axes[0].scatter([sel_lam], [sel_row["val_macro_f1"]], s=90, color=seed_colors[seed], edgecolors="black", zorder=5)

        axes[1].plot(df["lambda_weight"], df["val_log_loss"], marker="o", markersize=4, label=f"Seed {seed} (λ*={sel_lam:.2f})", color=seed_colors[seed], linewidth=1.6)
        axes[1].scatter([sel_lam], [sel_row["val_log_loss"]], s=90, color=seed_colors[seed], edgecolors="black", zorder=5)

    axes[0].set_title("Validation Macro-F1 vs. λ (21 Candidates)", fontweight="bold")
    axes[0].set_xlabel("Speech Weight λ (Text Weight = 1 - λ)")
    axes[0].set_ylabel("Validation Macro-F1")
    axes[0].set_ylim(-0.02, 1.06)
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].grid(alpha=0.25, linestyle="--")

    axes[1].set_title("Validation Log-Loss Tie-Breaker vs. λ", fontweight="bold")
    axes[1].set_xlabel("Speech Weight λ (Text Weight = 1 - λ)")
    axes[1].set_ylabel("Validation Log-Loss (lower is better)")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(alpha=0.25, linestyle="--")

    fig.suptitle(
        "Late-Logit Fusion Validation Calibration Across Seeds 42, 123, 999 (Word-Independent Split)",
        fontsize=12.5, fontweight="bold", y=1.03,
    )
    fig.savefig(PLOTS_DIR / "late_fusion_lambda_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cross_speaker_comparison(overall_rows: List[Dict[str, Any]]) -> None:
    """Compare WavLM vs. MFCC on cross_speaker_split across Accuracy, Macro-F1, and per-class F1."""
    wavlm = next(r for r in overall_rows if r["split_type"] == "cross_speaker_split" and r["architecture"] == "wavlm_bigru_attn")
    mfcc = next(r for r in overall_rows if r["split_type"] == "cross_speaker_split" and r["architecture"] == "mfcc_cnn_bilstm")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), constrained_layout=True)

    # Panel 1: Accuracy & Macro-F1
    metrics_x = np.arange(2)
    w = 0.34
    axes[0].bar(
        metrics_x - w / 2,
        [wavlm["accuracy_mean"] * 100, wavlm["macro_f1_mean"] * 100],
        w,
        yerr=[wavlm["accuracy_std"] * 100, wavlm["macro_f1_std"] * 100],
        capsize=4,
        label="WavLM + BiGRU + Attn",
        color="#0072B2",
    )
    axes[0].bar(
        metrics_x + w / 2,
        [mfcc["accuracy_mean"] * 100, mfcc["macro_f1_mean"] * 100],
        w,
        yerr=[mfcc["accuracy_std"] * 100, mfcc["macro_f1_std"] * 100],
        capsize=4,
        label="MFCC + CNN-BiLSTM",
        color="#E69F00",
    )
    axes[0].set_xticks(metrics_x)
    axes[0].set_xticklabels(["Accuracy (%)", "Macro-F1 (%)"])
    axes[0].set_ylim(0, 70)
    axes[0].set_ylabel("Score (%) — 3-Seed Mean ± Std")
    axes[0].set_title("Cross-Speaker Overall Performance (OAF -> YAF)", fontweight="bold")
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    # Panel 2: Per-class F1
    emo_x = np.arange(len(EMOTION_NAMES))
    axes[1].bar(
        emo_x - w / 2,
        [wavlm[f"f1_{e}_mean"] * 100 for e in EMOTION_NAMES],
        w,
        yerr=[wavlm[f"f1_{e}_std"] * 100 for e in EMOTION_NAMES],
        capsize=3,
        label="WavLM + BiGRU + Attn",
        color="#0072B2",
    )
    axes[1].bar(
        emo_x + w / 2,
        [mfcc[f"f1_{e}_mean"] * 100 for e in EMOTION_NAMES],
        w,
        yerr=[mfcc[f"f1_{e}_std"] * 100 for e in EMOTION_NAMES],
        capsize=3,
        label="MFCC + CNN-BiLSTM",
        color="#E69F00",
    )
    axes[1].set_xticks(emo_x)
    axes[1].set_xticklabels([EMOTION_LABELS_PRETTY[e] for e in EMOTION_NAMES], rotation=22)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Per-Class F1 (%) — 3-Seed Mean ± Std")
    axes[1].set_title("Cross-Speaker Per-Class F1 Comparison", fontweight="bold")
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle(
        "Cross-Speaker Speech Generalization: WavLM vs. MFCC (N_test = 1,400, 3 Seeds)",
        fontsize=12.5, fontweight="bold", y=1.03,
    )
    fig.savefig(PLOTS_DIR / "cross_speaker_speech_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_error_analysis_and_plot() -> pd.DataFrame:
    """Select 5 concrete misclassified samples from saved predictions, export CSV and plot."""
    case_specs = [
        (
            "Gated Fusion (WI, s42)",
            "word_indep_split",
            42,
            RESULTS_ROOT / "fusion/gated_fusion/word_indep_split/seed_42/test_predictions.csv",
            "OAF_hall_pleasant_surprise",
        ),
        (
            "MFCC Speech (WI, s123)",
            "word_indep_split",
            123,
            RESULTS_ROOT / "speech/word_indep_split/mfcc_cnn_bilstm/seed_123/test_predictions.csv",
            "OAF_rush_disgust",
        ),
        (
            "DistilBERT Text (WI, s42)",
            "word_indep_split",
            42,
            RESULTS_ROOT / "text/distilbert_bilstm_attn/word_independent/seed_42/test_predictions.csv",
            "OAF_bite_angry",
        ),
        (
            "WavLM Speech (CS, s42)",
            "cross_speaker_split",
            42,
            RESULTS_ROOT / "speech/cross_speaker_split/wavlm_bigru_attn/seed_42/test_predictions.csv",
            "YAF_back_happy",
        ),
        (
            "MFCC Speech (CS, s123)",
            "cross_speaker_split",
            123,
            RESULTS_ROOT / "speech/cross_speaker_split/mfcc_cnn_bilstm/seed_123/test_predictions.csv",
            "YAF_back_fear",
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for model_label, split_col, seed, csv_path, sample_id in case_specs:
        df = pd.read_csv(csv_path)
        match = df[df["sample_id"] == sample_id]
        assert len(match) == 1, f"Expected 1 row for {sample_id} in {csv_path}"
        r = match.iloc[0]
        assert r["true_emotion"] != r["pred_emotion"], f"Sample {sample_id} is not misclassified!"
        item: Dict[str, Any] = {
            "sample_id": sample_id,
            "model": model_label,
            "split_type": split_col,
            "seed": seed,
            "true_emotion": str(r["true_emotion"]),
            "pred_emotion": str(r["pred_emotion"]),
            "confidence": round(float(r["confidence"]), 6),
        }
        for emo in EMOTION_NAMES:
            item[f"prob_{emo}"] = round(float(r[f"prob_{emo}"]), 6)
        rows.append(item)

    err_df = pd.DataFrame(rows)
    err_df.to_csv(SUMMARIES_DIR / "error_analysis_five_cases.csv", index=False)

    # Plot 5-panel horizontal probability bar chart
    fig, axes = plt.subplots(5, 1, figsize=(10.5, 11.5), constrained_layout=True)
    for ax, (_, r) in zip(axes, err_df.iterrows()):
        probs = [float(r[f"prob_{e}"]) for e in EMOTION_NAMES]
        bar_colors = []
        for emo in EMOTION_NAMES:
            if emo == r["true_emotion"]:
                bar_colors.append("#009E73")  # True emotion (green)
            elif emo == r["pred_emotion"]:
                bar_colors.append("#D55E00")  # Predicted emotion (vermilion)
            else:
                bar_colors.append("#CCCCCC")  # Other classes
        y_pos = np.arange(len(EMOTION_NAMES))
        ax.barh(y_pos, probs, color=bar_colors, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([EMOTION_LABELS_PRETTY[e] for e in EMOTION_NAMES])
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("Predicted Probability")
        ax.set_title(
            f"{r['sample_id']}  |  {r['model']}  |  "
            f"True: {EMOTION_LABELS_PRETTY[r['true_emotion']]} (green)  vs.  "
            f"Pred: {EMOTION_LABELS_PRETTY[r['pred_emotion']]} (red, p={r['confidence']:.3f})",
            fontweight="bold",
            fontsize=9.8,
        )
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        for idx_p, val in enumerate(probs):
            if val >= 0.03:
                ax.text(val + 0.012, idx_p, f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle(
        "Error Analysis: Class Probability Distributions for Five Misclassified Test Samples",
        fontsize=12.5, fontweight="bold", y=1.01,
    )
    fig.savefig(PLOTS_DIR / "error_analysis_five_cases.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return err_df


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    hashes_before = snapshot_frozen_hashes()
    runs = collect_all_27_runs()
    unified = build_comparison_tables(runs)

    tsne_info = plot_tsne_three_panel()
    plot_confusion_matrices()
    plot_training_curves()
    plot_per_class_f1(unified["overall_comparison"])
    plot_gated_fusion_analysis()
    plot_late_fusion_calibration()
    plot_cross_speaker_comparison(unified["overall_comparison"])
    err_df = build_error_analysis_and_plot()

    # Validate all generated plots are non-empty
    expected_plots = [
        "tsne_three_panel_comparison.png",
        "confusion_matrices_word_independent.png",
        "confusion_matrices_cross_speaker.png",
        "training_validation_curves.png",
        "per_class_f1_comparison.png",
        "gated_fusion_gate_analysis.png",
        "late_fusion_lambda_calibration.png",
        "cross_speaker_speech_comparison.png",
        "error_analysis_five_cases.png",
    ]
    print("=== GENERATED PLOTS VERIFICATION ===")
    for p_name in expected_plots:
        p = PLOTS_DIR / p_name
        assert p.exists() and p.stat().st_size > 10_000, f"Plot missing or empty: {p}"
        print(f"  OK: {p.relative_to(PROJECT_ROOT)} ({p.stat().st_size / 1024:.1f} KB)")

    hashes_after = snapshot_frozen_hashes()
    assert hashes_before == hashes_after, "CRITICAL: Frozen artifact hash changed!"
    print(f"\nALL {len(hashes_after)} FROZEN EXPERIMENT ANCHOR SHA-256 HASHES VERIFIED UNCHANGED!")
    print(f"Error analysis cases exported: {len(err_df)}")
    print(f"t-SNE verified aligned: {tsne_info['sample_ids_aligned']} (N={tsne_info['num_samples']})")


if __name__ == "__main__":
    main()
